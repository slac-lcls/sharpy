"""
Fast verification that position_solve_coupled runs and converges on the active
backend (CPU numpy or GPU cupy via config.GPU).  Purpose-built to exercise the
cupyx sparse `.real` fix (_sparse_real) and the downstream cupyx sparse path
(bmat / diags / spsolve).

The full drift test is impractical on GPU: the coupled solver's neighbor-pair
kernel is pure Python (numba is CPU-only), so every pair is a tiny GPU launch
with a host sync, and a 144-frame solve runs for minutes.  So we use a SMALL
but well-conditioned geometry (frame <= image, high overlap) and print per
iteration with a flush + timing, so progress is visible incrementally.

Run:  python verify_coupled_gpu.py
"""

import os
import sys
import time

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
    position_solve_coupled,
    position_plan,
    shift_rmse,
)

# Small, well-conditioned: frame (nx) <= image (Nx=nnx*step), high overlap.
NNX = NNY = 6
STEP = 6
NX = NY = 32
RMS = 0.5
NITER = 6


def build(seed=0):
    """Same construction as position_drift_test.build, parametrized small."""
    rng = np.random.default_rng(seed)
    Nx = Ny = NNX * STEP

    probe = make_probe(NX, NY)
    if isinstance(probe, tuple):
        probe = probe[0]
    probe = xp.asarray(probe / xp.abs(probe).max(), dtype=xp.complex64)

    a = rng.standard_normal((Nx, Ny)) + 1j * rng.standard_normal((Nx, Ny))
    A = np.fft.fft2(a)
    fx, fy = np.meshgrid(np.fft.fftfreq(Nx), np.fft.fftfreq(Ny), indexing="ij")
    A *= np.exp(-(fx ** 2 + fy ** 2) / (2 * 0.10 ** 2))
    truth = xp.asarray(np.fft.ifft2(A).astype(np.complex64))

    tx, ty = np.meshgrid(np.arange(NNX) * STEP, np.arange(NNY) * STEP, indexing="ij")
    translations_x = xp.asarray(tx.ravel().astype(np.float64))
    translations_y = xp.asarray(ty.ravel().astype(np.float64))
    nframes = translations_x.size

    ex = rng.standard_normal(nframes)
    ey = rng.standard_normal(nframes)
    ex -= ex.mean()
    ey -= ey.mean()
    r = np.sqrt(ex.var() + ey.var())
    ex, ey = ex / r * RMS, ey / r * RMS
    xi_x = xp.asarray(ex)
    xi_y = xp.asarray(ey)

    dp = probe_derivatives(probe)
    probe_shifted = taylor_shift_probe(dp, xi_x, xi_y)["O"]
    mapid = map_frames(translations_x, translations_y, NX, NY, Nx, Ny)
    frames = Splitc(truth, mapid) * probe_shifted
    plan = position_plan(translations_x, translations_y, nframes, NX, NY, Nx, Ny)
    return dict(frames=frames, dp=dp, truth=truth, mapid=mapid, Nx=Nx, Ny=Ny,
                nframes=nframes, xi_x=xi_x, xi_y=xi_y, plan=plan)


def _sync():
    if config.GPU:
        cp.cuda.Stream.null.synchronize()


if __name__ == "__main__":
    p = build(seed=0)
    hx = xp.zeros(p["nframes"])
    hy = xp.zeros(p["nframes"])
    r0 = float(shift_rmse(p["xi_x"], p["xi_y"], hx, hy))
    print(f"config.GPU={config.GPU}, {p['nframes']} frames, COUPLED solver, "
          f"{NITER} iters", flush=True)
    print(f"  iter  0: Delta_r = {r0:.3e}", flush=True)

    last = r0
    for k in range(NITER):
        t = time.time()
        hx, hy = position_solve_coupled(
            p["frames"], p["dp"], p["truth"], p["mapid"],
            p["Nx"], p["Ny"], hx, hy, p["plan"], max_step=0.5,
        )
        _sync()
        last = float(shift_rmse(p["xi_x"], p["xi_y"], hx, hy))
        print(f"  iter {k + 1:>2}: Delta_r = {last:.3e}  ({time.time() - t:.1f}s)",
              flush=True)

    ok = np.isfinite(last) and last < 1e-3 * r0
    print(f"-> {'CONVERGED' if ok else 'DID NOT CONVERGE'} "
          f"({r0:.3e} -> {last:.3e})", flush=True)
    sys.exit(0 if ok else 1)
