"""
Does the COUPLED position solver beat the DIAGONAL one when position errors
are CORRELATED (a slow drift during sequential measurement)?

Hypothesis: the coupled solver's off-diagonal overlap terms link overlapping
neighbors, so it should resolve smooth/correlated drift modes faster than the
per-frame diagonal solver, while for uncorrelated (i.i.d.) errors the neighbor
coupling carries little extra information.

Setup: smooth probe (so coupled is well-conditioned and stable), consistent
noise-free data, true image fixed, recover shifts from zero. Two error fields
of the SAME global-shift-invariant RMS:
  - uncorrelated: i.i.d. Gaussian per frame
  - drift:        2D random walk along the raster acquisition order (smooth)

Reports Delta_r (Fienup Eq. 15) vs iteration for diag and coupled on each.

Run:  python position_drift_test.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

if config.GPU:
    import cupy as cp

    xp = cp
else:
    xp = np

from Operators import make_probe, map_frames, Splitc
from position_retrieval import (
    probe_derivatives,
    taylor_shift_probe,
    position_solve_diag,
    position_solve_coupled,
    position_plan,
    shift_rmse,
)


def make_errors(nnx, nny, target_rms, seed):
    """Return (uncorrelated, drift) shift fields, each zero-mean, same Delta_r."""
    rng = np.random.default_rng(seed)
    nframes = nnx * nny

    # uncorrelated i.i.d.
    ux = rng.standard_normal(nframes)
    uy = rng.standard_normal(nframes)

    # correlated drift: random walk along the raster acquisition order
    dx = np.cumsum(rng.standard_normal(nframes))
    dy = np.cumsum(rng.standard_normal(nframes))

    def norm(x, y):
        x = x - x.mean()
        y = y - y.mean()
        rms = np.sqrt(x.var() + y.var())
        return x / rms * target_rms, y / rms * target_rms

    return norm(ux, uy), norm(dx, dy)


def build(xi_x_np, xi_y_np, seed=0):
    rng = np.random.default_rng(seed)
    nx = ny = 32
    nnx = nny = 12
    step = 4
    Nx = Ny = nnx * step

    probe = make_probe(nx, ny)  # smooth broad probe -> coupled is stable
    if isinstance(probe, tuple):
        probe = probe[0]
    probe = xp.asarray(probe / xp.abs(probe).max(), dtype=xp.complex64)

    a = rng.standard_normal((Nx, Ny)) + 1j * rng.standard_normal((Nx, Ny))
    A = np.fft.fft2(a)
    fx, fy = np.meshgrid(np.fft.fftfreq(Nx), np.fft.fftfreq(Ny), indexing="ij")
    A *= np.exp(-(fx ** 2 + fy ** 2) / (2 * 0.10 ** 2))
    truth = xp.asarray(np.fft.ifft2(A).astype(np.complex64))

    tx, ty = np.meshgrid(np.arange(nnx) * step, np.arange(nny) * step, indexing="ij")
    translations_x = xp.asarray(tx.ravel().astype(np.float64))
    translations_y = xp.asarray(ty.ravel().astype(np.float64))
    nframes = translations_x.size

    xi_x = xp.asarray(xi_x_np)
    xi_y = xp.asarray(xi_y_np)

    dp = probe_derivatives(probe)
    probe_shifted = taylor_shift_probe(dp, xi_x, xi_y)["O"]
    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)
    frames = Splitc(truth, mapid) * probe_shifted

    plan = position_plan(translations_x, translations_y, nframes, nx, ny, Nx, Ny)
    return dict(frames=frames, dp=dp, truth=truth, mapid=mapid,
                Nx=Nx, Ny=Ny, nframes=nframes, xi_x=xi_x, xi_y=xi_y, plan=plan)


def converge(p, method, niter):
    hx = xp.zeros(p["nframes"])
    hy = xp.zeros(p["nframes"])
    curve = [shift_rmse(p["xi_x"], p["xi_y"], hx, hy)]
    for _ in range(niter):
        if method == "diag":
            hx, hy = position_solve_diag(p["frames"], p["dp"], p["truth"], p["mapid"],
                                         p["Nx"], p["Ny"], hx, hy, max_step=0.5)
        else:
            hx, hy = position_solve_coupled(p["frames"], p["dp"], p["truth"], p["mapid"],
                                            p["Nx"], p["Ny"], hx, hy, p["plan"], max_step=0.5)
        curve.append(shift_rmse(p["xi_x"], p["xi_y"], hx, hy))
    return np.array(curve)


if __name__ == "__main__":
    nnx = nny = 12
    target_rms = 0.5
    niter = 15
    (ux, uy), (dxv, dyv) = make_errors(nnx, nny, target_rms, seed=7)

    results = {}
    for label, (ex, ey) in (("uncorrelated", (ux, uy)), ("drift", (dxv, dyv))):
        p = build(ex, ey, seed=0)
        for method in ("diag", "coupled"):
            results[(label, method)] = converge(p, method, niter)

    print(f"Delta_r (px), true RMS={target_rms}, {nnx*nny} frames, smooth probe\n")
    print(f"{'iter':>4} | {'uncorr diag':>11} {'uncorr coup':>11} | {'drift diag':>11} {'drift coup':>11}")
    print("-" * 60)
    for k in range(niter + 1):
        print(f"{k:>4} | {results[('uncorrelated','diag')][k]:>11.3e} "
              f"{results[('uncorrelated','coupled')][k]:>11.3e} | "
              f"{results[('drift','diag')][k]:>11.3e} "
              f"{results[('drift','coupled')][k]:>11.3e}")

    # advantage = how much faster coupled reaches a threshold than diag
    def iters_to(curve, thr):
        below = np.where(curve < thr)[0]
        return int(below[0]) if below.size else None
    thr = 1e-3 * target_rms
    print(f"\niters to reach Delta_r < {thr:.1e}:")
    for label in ("uncorrelated", "drift"):
        d = iters_to(results[(label, "diag")], thr)
        c = iters_to(results[(label, "coupled")], thr)
        print(f"  {label:>12}: diag={d}  coupled={c}")

    # Hypothesis check: how much does each method GAIN from correlated drift?
    # ratio = Delta_r(uncorrelated) / Delta_r(drift) at a few early iterations.
    print("\ngain from correlation  =  Delta_r(uncorrelated) / Delta_r(drift)")
    print("  (>1 means the method exploits the drift correlation)")
    for method in ("diag", "coupled"):
        ratios = [results[("uncorrelated", method)][k] / results[("drift", method)][k]
                  for k in (2, 3, 4)]
        print(f"  {method:>8}: iters 2-4 -> " + ", ".join(f"{r:.1f}x" for r in ratios))
