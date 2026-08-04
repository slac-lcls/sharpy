#!/usr/bin/env python3
"""Portable GPU eigensolver micro-benchmark (cupy/cupyx ONLY -- no sharpy/wrap_ops),
so the identical code runs on any GPU. Times the sync eigensolvers per solve on the
synthetic connection Gramian (varying-low-freq phase, the regime where sync matters):
power, eigsh (under-converged like in-loop, and converged), invit-CG, invit-direct
(inverse-power via cupyx splu). Reports per-solve ms + alignment with the true phase.

Multiply per-solve ms by ~300 (+ ~0.6s AP overhead) to estimate the in-loop number.

  CONTRAST=0.85 EPS=1e-4 CYC=1.0 python gpu_eig_bench.py
"""
import os, time
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as csl

SIGMA, R = 1.6, 4
CONTRAST = float(os.environ.get("CONTRAST", 0.85))
EPS = float(os.environ.get("EPS", 1e-4))
CYC = float(os.environ.get("CYC", 1.0))
Gs = [int(g) for g in os.environ.get("GS", "64 128 160").split()]


def frame_weights(G, c):
    if c <= 0: return np.ones(G * G)
    xs, ys = np.meshgrid(np.linspace(0, 1, G), np.linspace(0, 1, G), indexing="ij")
    b = np.zeros((G, G))
    for cx, cy, a in [(0.5, 0.5, 0.9), (0.3, 0.7, 0.7), (0.75, 0.35, 0.6)]:
        b += a * np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / 0.02))
    return np.clip(1.0 - c * np.clip(b, 0, 1), 0.1, 1.0).ravel()


def build(G, cyc, contrast):
    """Connection adjacency Hn (normalized), M=Lsym+epsI, truth phi -- all on GPU."""
    n = G * G; s = frame_weights(G, contrast)
    x0, y0 = np.meshgrid(np.linspace(-1, 1, G), np.linspace(-1, 1, G), indexing="ij")
    th = (cyc * 2 * np.pi * (x0 ** 2 + y0 ** 2) / 2.0).ravel()
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
    ri = cp.asarray(np.concatenate(ri)); ci = cp.asarray(np.concatenate(ci))
    av = cp.asarray(np.concatenate(av)); mv = cp.asarray(np.concatenate(mv))
    A = csp.csr_matrix((av, (ri, ci)), shape=(n, n))
    Amag = csp.csr_matrix((mv, (ri, ci)), shape=(n, n))
    d = cp.asarray(Amag.sum(axis=1)).ravel(); d = cp.maximum(d, 1e-30)
    dm12 = csp.diags(1.0 / cp.sqrt(d))
    I = csp.identity(n, dtype=cp.complex128, format="csr")
    Hn = (dm12 @ A @ dm12).tocsr()
    M = (I - Hn + EPS * I).tocsr()
    return Hn, M, cp.asarray(np.exp(1j * th))


def align(w, phi):
    wn = w.ravel() / (cp.abs(w.ravel()) + 1e-30)
    return float(cp.abs(cp.vdot(phi, wn)) / phi.size)


def sync(): cp.cuda.Stream.null.synchronize()

def timed(fn):
    sync(); t = time.perf_counter(); out = fn(); sync(); return out, (time.perf_counter() - t) * 1e3


def power(Hn, it):
    v = cp.ones(Hn.shape[0], cp.complex128); v /= cp.linalg.norm(v)
    for _ in range(it):
        v = Hn @ v; v /= cp.linalg.norm(v)
    return v

def eigsh_(Hn, ncv, mi):
    w, V = csl.eigsh(Hn, k=1, which="LM", ncv=ncv, maxiter=mi); return V[:, int(cp.argmax(w))]

def cg_solve(M):
    b = cp.ones(M.shape[0], cp.complex128)
    try: x, _ = csl.cg(M, b, rtol=1e-8, maxiter=20000)
    except TypeError: x, _ = csl.cg(M, b, tol=1e-8, maxiter=20000)
    return x

def splu_solve(M, steps=3):
    lu = csl.splu(M.tocsc()); x = cp.ones(M.shape[0], cp.complex128)
    for _ in range(steps):
        x = lu.solve(x); x /= cp.linalg.norm(x)
    return x


print("GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
      "| cupy", cp.__version__, "| cyc", CYC, "contrast", CONTRAST)
for G in Gs:
    Hn, M, phi = build(G, CYC, CONTRAST)
    print(f"\n=== {G*G} frames (G={G}) ===  {'method':<26}{'ms/solve':>9}{'align':>8}")
    for nm, fn in [
        ("power(50)",            lambda: power(Hn, 50)),
        ("power(200)",           lambda: power(Hn, 200)),
        ("eigsh ncv3 mi20",      lambda: eigsh_(Hn, 3, 20)),
        ("eigsh ncv20 mi2000",   lambda: eigsh_(Hn, 20, 2000)),
        ("invit-CG",             lambda: cg_solve(M)),
        ("invit-direct splu(3)", lambda: splu_solve(M, 3)),
    ]:
        try:
            v, ms = timed(fn); print(f"   {' ':<0}{nm:<26}{ms:>9.1f}{align(v, phi):>8.3f}")
        except Exception as e:
            print(f"   {nm:<26}{'-':>9}  {type(e).__name__}: {str(e)[:40]}")

print("\nPer-solve ms x300 (+~0.6s AP overhead) ~ in-loop time. Compare across GPUs.")
