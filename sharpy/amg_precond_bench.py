#!/usr/bin/env python3
"""Phase A: does AMG preconditioning give O(1) CG iters for the inverse-iteration sync
solve, while plain CG grows with frame count? Compares plain CG vs AMG-CG (PyAMG) vs
ILU-CG on (Lsym+eps I) across scan sizes.

AMG/ILU here are PRECONDITIONERS for CG on the COMPLEX Hermitian-PD system; the AMG
hierarchy is built on the REAL magnitude Laplacian M_real = I - |Hn| + eps I (same graph
connectivity / Fiedler structure, real SPD -> standard AMG), then applied to Re/Im parts
separately (M^-1 is real -> M^-1(xr+i xi)=M^-1 xr + i M^-1 xi). This mirrors exactly the
GPU AMGX wiring planned for Eigensolver_invit.

  CONTRAST=0.85 EPS=1e-4 python amg_precond_bench.py
"""
import os, time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, spilu, LinearOperator
import pyamg

SIGMA, R = 1.6, 4
CONTRAST = float(os.environ.get("CONTRAST", 0.85))
EPS = float(os.environ.get("EPS", 1e-4))
CYC = float(os.environ.get("CYC", 1.0))


def frame_weights(G, c):
    if c <= 0: return np.ones(G * G)
    xs, ys = np.meshgrid(np.linspace(0, 1, G), np.linspace(0, 1, G), indexing="ij")
    b = np.zeros((G, G))
    for cx, cy, a in [(0.5, 0.5, 0.9), (0.3, 0.7, 0.7), (0.75, 0.35, 0.6)]:
        b += a * np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / 0.02))
    return np.clip(1.0 - c * np.clip(b, 0, 1), 0.1, 1.0).ravel()


def build(G, cyc, contrast):
    n = G * G; s = frame_weights(G, contrast)
    xs0, ys0 = np.meshgrid(np.linspace(-1, 1, G), np.linspace(-1, 1, G), indexing="ij")
    th = (cyc * 2 * np.pi * (xs0 ** 2 + ys0 ** 2) / 2.0).ravel()
    xs, ys = np.meshgrid(np.arange(G), np.arange(G), indexing="ij"); xs = xs.ravel(); ys = ys.ravel()
    ri, ci, av, mv = [], [], [], []
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            d2 = dx * dx + dy * dy
            if d2 == 0 or d2 > R * R: continue
            xn, yn = xs + dx, ys + dy
            ok = (xn >= 0) & (xn < G) & (yn >= 0) & (yn < G)
            i = np.where(ok)[0]; j = xn[ok] + yn[ok] * G
            w = s[i] * s[j] * np.exp(-d2 / (SIGMA * SIGMA))
            ri.append(i); ci.append(j); av.append(w * np.exp(1j * (th[i] - th[j]))); mv.append(w)
    ri = np.concatenate(ri); ci = np.concatenate(ci)
    A = sparse.csr_matrix((np.concatenate(av), (ri, ci)), shape=(n, n)).astype(complex)
    Amag = sparse.csr_matrix((np.concatenate(mv), (ri, ci)), shape=(n, n))
    d = np.asarray(Amag.sum(1)).ravel(); d = np.maximum(d, 1e-30); dm12 = sparse.diags(1 / np.sqrt(d))
    I = sparse.identity(n, dtype=complex, format="csr")
    Hn = (dm12 @ A @ dm12).tocsr()
    absHn = (dm12 @ Amag @ dm12).tocsr()
    M = (I - Hn + EPS * I).tocsr()                                  # complex Hermitian PD
    M_real = (sparse.identity(n, format="csr") - absHn + EPS * sparse.identity(n, format="csr")).tocsr()
    return M, M_real, np.exp(1j * th)


def align(w, phi):
    wn = w.ravel() / (np.abs(w.ravel()) + 1e-30)
    return float(np.abs(np.vdot(phi, wn)) / phi.size)


class Counter:
    def __init__(s, A): s.A, s.n = A, 0
    def mv(s, x): s.n += 1; return s.A @ x
    def op(s): return LinearOperator(s.A.shape, matvec=s.mv, dtype=s.A.dtype)


def _cg(opM, b, M, tol, x0):
    try: return cg(opM, b, M=M, x0=x0, rtol=tol, maxiter=20000)
    except TypeError: return cg(opM, b, M=M, x0=x0, tol=tol, maxiter=20000)


def solve(M, precond, phi):
    n = M.shape[0]; c = Counter(M); b = np.ones(n, complex)
    x, _ = _cg(c.op(), b, precond, 1e-8, b.copy()); x /= np.linalg.norm(x)
    return c.n, align(x, phi)


def amg_precond(M_real):
    ml = pyamg.smoothed_aggregation_solver(M_real, max_coarse=200)
    P = ml.aspreconditioner(cycle="V")
    return LinearOperator(M_real.shape, matvec=lambda r: P.matvec(r.real) + 1j * P.matvec(r.imag),
                          dtype=complex), ml


def ilu_precond(M):
    ilu = spilu(M.tocsc(), drop_tol=1e-3, fill_factor=10)
    return LinearOperator(M.shape, matvec=ilu.solve, dtype=complex)


print(f"AMG-preconditioned CG scaling  (cyc={CYC}, contrast={CONTRAST}, eps={EPS})")
print(f"  {'G':>4}{'frames':>8} | {'plainCG it':>11}{'ms':>7} | {'AMG setup':>10}{'AMG it':>7}{'ms':>7} | "
      f"{'ILU setup':>10}{'ILU it':>7}{'ms':>7} | {'align(amg)':>11}")
for G in (64, 96, 128, 160):
    M, M_real, phi = build(G, CYC, CONTRAST)
    t = time.perf_counter(); it_p, al_p = solve(M, None, phi); ms_p = (time.perf_counter() - t) * 1e3
    t = time.perf_counter(); Pamg, _ = amg_precond(M_real); s_amg = (time.perf_counter() - t) * 1e3
    t = time.perf_counter(); it_a, al_a = solve(M, Pamg, phi); ms_a = (time.perf_counter() - t) * 1e3
    t = time.perf_counter(); Pilu = ilu_precond(M); s_ilu = (time.perf_counter() - t) * 1e3
    t = time.perf_counter(); it_i, al_i = solve(M, Pilu, phi); ms_i = (time.perf_counter() - t) * 1e3
    print(f"  {G:>4}{G*G:>8} | {it_p:>11}{ms_p:>7.0f} | {s_amg:>10.0f}{it_a:>7}{ms_a:>7.0f} | "
          f"{s_ilu:>10.0f}{it_i:>7}{ms_i:>7.0f} | {al_a:>11.3f}")

print("\nGoal: AMG 'it' stays ~flat (O(1)) as frames grow, while plain CG 'it' climbs.")
print("AMG setup amortizes in-loop (build once, reuse); compare AMG solve-ms vs plain ms.")
