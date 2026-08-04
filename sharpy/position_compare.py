"""
Compare position-retrieval methods on the same problem:

  1. diagonal  (Section IV, per-frame 2x2 Gauss-Newton)        position_solve_diag
  2. coupled   (Section IV, full Eq. 27 with off-diagonal)     position_solve_coupled
  3. gradient  (Guizar-Sicairos & Fienup 2008, Eq. 21,         position_solve_gradient
                steepest descent on the data discrepancy)

Setup isolates the position solver: consistent (noise-free) data generated with
the Taylor probe-shift model, true image held fixed, recover shifts from zero.
Reports the global-shift-invariant shift RMSE (Fienup Eq. 15) vs iteration and
wall-clock time.

Run:  python position_compare.py
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
    position_solve_diag,
    position_solve_coupled,
    position_solve_gradient,
    position_plan,
    shift_rmse,
)


def build(seed=1, sigma_pix=0.8, probe_radii=None):
    rng = np.random.default_rng(seed)
    nx = ny = 32
    nnx = nny = 12
    step = 4
    Nx = Ny = nnx * step

    if probe_radii is None:
        probe = make_probe(nx, ny)              # smooth broad probe
    else:
        probe = make_probe(nx, ny, r1=probe_radii[0], r2=probe_radii[1])  # zone plate
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

    xi_x = xp.asarray(rng.standard_normal(nframes) * sigma_pix)
    xi_y = xp.asarray(rng.standard_normal(nframes) * sigma_pix)

    dp = probe_derivatives(probe)
    probe_shifted = taylor_shift_probe(dp, xi_x, xi_y)["O"]
    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)
    frames_clean = Splitc(truth, mapid) * probe_shifted
    frames_data = xp.abs(xp.fft.fft2(frames_clean)) ** 2  # intensity

    plan = position_plan(translations_x, translations_y, nframes, nx, ny, Nx, Ny)
    return dict(
        frames_clean=frames_clean, frames_data=frames_data, dp=dp, truth=truth,
        mapid=mapid, Nx=Nx, Ny=Ny, nframes=nframes, xi_x=xi_x, xi_y=xi_y, plan=plan,
    )


def run_method(name, p, niter):
    hx = xp.zeros(p["nframes"])
    hy = xp.zeros(p["nframes"])
    dr0 = shift_rmse(p["xi_x"], p["xi_y"], hx, hy)
    t0 = time.time()
    for _ in range(niter):
        if name == "diag":
            hx, hy = position_solve_diag(
                p["frames_clean"], p["dp"], p["truth"], p["mapid"],
                p["Nx"], p["Ny"], hx, hy, max_step=0.5)
        elif name == "coupled":
            hx, hy = position_solve_coupled(
                p["frames_clean"], p["dp"], p["truth"], p["mapid"],
                p["Nx"], p["Ny"], hx, hy, p["plan"], max_step=0.5)
        elif name == "gradient":
            hx, hy = position_solve_gradient(
                p["frames_data"], p["truth"], p["dp"], p["mapid"],
                hx, hy, max_step=0.5)
    dt = time.time() - t0
    dr = shift_rmse(p["xi_x"], p["xi_y"], hx, hy)
    return dr0, dr, dt


if __name__ == "__main__":
    # within-regime perturbation (< trust region / Taylor validity)
    for plabel, radii in (("smooth probe", None),
                          ("zone-plate probe", (0.075, 0.255))):
        p = build(seed=2, sigma_pix=0.35, probe_radii=radii)
        dr_init = shift_rmse(p["xi_x"], p["xi_y"], 0 * p["xi_x"], 0 * p["xi_y"])
        print(f"\n### {plabel}: frames={p['nframes']}, true shift RMS ~ {dr_init:.3f} px")
        print(f"{'method':>9} | {'Delta_r start':>13} -> {'Delta_r end':>11} | {'time(s)':>8} | {'iters':>5}")
        print("-" * 62)
        # Gauss-Newton methods need few iters; steepest descent needs many.
        for name, niter in (("diag", 15), ("coupled", 15), ("gradient", 200)):
            dr0, dr, dt = run_method(name, p, niter)
            print(f"{name:>9} | {dr0:>13.3e} -> {dr:>11.3e} | {dt:>8.2f} | {niter:>5}")
