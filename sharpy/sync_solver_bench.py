#!/usr/bin/env python3
"""Head-to-head sync eigensolvers on the synthetic connection Laplacian:
power vs eigsh vs inverse-iteration (DIRECT splu and ITERATIVE cg/minres), plus
invit-as-INITIALIZER for warm power.

Reports MATVECS (hardware-independent: each = one O(nnz) sparse apply), wall-ms, and
alignment with the true phase (1.0 = synchronized). The whole point is the varying
low-frequency phase regime (cycles>=1), where power plateaus.

  G=64 CONTRAST=0.85 EPS=1e-4 python sync_solver_bench.py
"""
import os, time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, splu, cg, minres, LinearOperator, ArpackNoConvergence

G        = int(os.environ.get("G", 64)); SIGMA, R = 1.6, 4
CONTRAST = float(os.environ.get("CONTRAST", 0.85))
EPS      = float(os.environ.get("EPS", 1e-4))
N = G * G


def low_freq_phase(G, cyc):
    xs, ys = np.meshgrid(np.linspace(-1, 1, G), np.linspace(-1, 1, G), indexing="ij")
    return (cyc * 2 * np.pi * (xs ** 2 + ys ** 2) / 2.0).ravel()

def frame_weights(G, c):
    if c <= 0: return np.ones(G * G)
    xs, ys = np.meshgrid(np.linspace(0, 1, G), np.linspace(0, 1, G), indexing="ij")
    b = np.zeros((G, G))
    for cx, cy, a in [(0.5, 0.5, 0.9), (0.3, 0.7, 0.7), (0.75, 0.35, 0.6)]:
        b += a * np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / 0.02))
    return np.clip(1.0 - c * np.clip(b, 0, 1), 0.1, 1.0).ravel()

def build(G, cyc, contrast):
    n = G * G; s = frame_weights(G, contrast); th = low_freq_phase(G, cyc)
    xs, ys = np.meshgrid(np.arange(G), np.arange(G), indexing="ij"); xs = xs.ravel(); ys = ys.ravel()
    ri, ci, vv = [], [], []
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            d2 = dx * dx + dy * dy
            if d2 == 0 or d2 > R * R: continue
            xn, yn = xs + dx, ys + dy
            ok = (xn >= 0) & (xn < G) & (yn >= 0) & (yn < G)
            i = np.where(ok)[0]; j = xn[ok] + yn[ok] * G
            w = s[i] * s[j] * np.exp(-d2 / (SIGMA * SIGMA))
            ri.append(i); ci.append(j); vv.append(w * np.exp(1j * (th[i] - th[j])))
    H = sparse.csr_matrix((np.concatenate(vv), (np.concatenate(ri), np.concatenate(ci))),
                          shape=(n, n)).astype(np.complex128)
    d = np.asarray(abs(H).sum(1)).ravel(); d = np.maximum(d, 1e-30)
    dm12 = sparse.diags(1 / np.sqrt(d))
    Hn = (dm12 @ H @ dm12).tocsr()
    Lsym = (sparse.identity(n, dtype=complex, format="csr") - Hn).tocsr()
    return Hn, Lsym, np.exp(1j * th)

def align(w, phi):
    wn = w.ravel() / (np.abs(w.ravel()) + 1e-30)
    return float(np.abs(np.vdot(phi, wn)) / phi.size)

class Counter:
    def __init__(self, A): self.A, self.n = A, 0
    def mv(self, x): self.n += 1; return self.A @ x
    def op(self): return LinearOperator(self.A.shape, matvec=self.mv, dtype=self.A.dtype)

def _solve(fn, op, b, tol, maxiter):
    try: return fn(op, b, rtol=tol, maxiter=maxiter)
    except TypeError: return fn(op, b, tol=tol, maxiter=maxiter)

def power(Hn, it, v0=None):
    c = Counter(Hn)
    v = (np.ones(N, complex) if v0 is None else v0.ravel().astype(complex)); v /= np.linalg.norm(v)
    for _ in range(it):
        v = c.mv(v); v /= np.linalg.norm(v)
    return v, c.n

def invit_direct(Lsym, eps, steps):
    M = (Lsym + eps * sparse.identity(N, dtype=complex, format="csr")).tocsc()
    lu = splu(M); x = np.ones(N, complex)
    for _ in range(steps):
        x = lu.solve(x); x /= np.linalg.norm(x)
    return x

def invit_iter(Lsym, eps, steps, tol, fn):
    M = (Lsym + eps * sparse.identity(N, dtype=complex, format="csr")).tocsr()
    c = Counter(M); op = c.op(); x = np.ones(N, complex)
    for _ in range(steps):
        x, _ = _solve(fn, op, x, tol, 10000); x /= np.linalg.norm(x)
    return x, c.n

def timed(fn):
    t0 = time.perf_counter(); out = fn(); return out, (time.perf_counter() - t0) * 1e3


for cyc in (1.0, 2.0):
    Hn, Lsym, phi = build(G, cyc, CONTRAST)
    lam = np.sort(eigsh(Hn, k=4, which="LA", maxiter=20000, tol=0, return_eigenvectors=False))[::-1]
    print(f"\n========== cycles={cyc}  N={N}  eig2/eig1={lam[1]/lam[0]:.4f}  "
          f"|<ones,phi>|^2/N={abs(np.vdot(np.ones(N,complex),phi))**2/N:.0f} ==========")
    print(f"  {'method':<34}{'matvecs':>9}{'ms':>9}{'align':>8}")

    def row(name, v, mv, ms): print(f"  {name:<34}{mv:>9}{ms:>9.1f}{align(v, phi):>8.3f}")

    for it in (50, 500, 2000):
        (v, mv), ms = timed(lambda it=it: power(Hn, it)); row(f"power(ones,{it})", v, mv, ms)

    for ncv, mi in ((3, 200), (20, 2000)):
        try:
            (v), ms = timed(lambda ncv=ncv, mi=mi: eigsh(Hn, k=1, which="LA", v0=np.ones(N),
                                                          ncv=ncv, maxiter=mi, tol=0)[1][:, 0])
            row(f"eigsh(Hn,ncv={ncv},mi={mi})", v, "-", ms)
        except ArpackNoConvergence:
            print(f"  {'eigsh(Hn,ncv=%d,mi=%d)'%(ncv,mi):<34}{'-':>9}{'-':>9}{'NoConverge':>8}")

    (v), ms = timed(lambda: eigsh(Lsym, k=1, sigma=-1e-5, which="LM")[1][:, 0])  # shift-invert at ~0 (library invit)
    row("eigsh shift-invert(Lsym,~0)", v, "fact", ms)

    (v), ms = timed(lambda: invit_direct(Lsym, EPS, 2)); row(f"invit DIRECT splu (eps={EPS:g},2)", v, "fact", ms)

    for eps, st, tol, fn, nm in ((EPS, 1, 1e-8, cg, "cg"), (1e-2, 2, 1e-8, cg, "cg"),
                                 (EPS, 1, 1e-8, minres, "minres")):
        try:
            (v, mv), ms = timed(lambda eps=eps, st=st, tol=tol, fn=fn: invit_iter(Lsym, eps, st, tol, fn))
            row(f"invit ITER {nm}(eps={eps:g},{st}st)", v, mv, ms)
        except Exception as e:
            print(f"  {'invit ITER %s(eps=%g)' % (nm, eps):<34}{'-':>9}{'-':>9}  {type(e).__name__}: {str(e)[:30]}")

    # invit as INITIALIZER for warm power: 1 cheap iterative invit step, then K power matvecs
    (x0, mv0), _ = timed(lambda: invit_iter(Lsym, 1e-2, 1, 1e-3, cg))   # cheap, loose
    for K in (5, 20):
        (v, mvp), ms = timed(lambda K=K: power(Hn, K, v0=x0))
        row(f"invit-init(cg,{mv0}mv)+power({K})", v, mv0 + mvp, ms)
    (vc, mvc), msc = timed(lambda: power(Hn, mv0 + 20))   # cold power, SAME total matvec budget
    row(f"  [vs cold power({mv0+20})]", vc, mvc, msc)

print("\nQ: matvecs are the currency. Watch (a) does power EVER reach align~1 in the budget,")
print("(b) invit DIRECT vs ITER cg/minres matvec cost (~sqrt(kappa)~sqrt(2/eps)), (c) whether")
print("a cheap invit start + a few warm-power matvecs converges where cold power does not.")
