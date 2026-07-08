"""
Long-range drift benchmark (v2): 1D scan, 50% overlap, linearly accumulating drift.

Geometry: single line of N frames, step=nx/2 (50% overlap).
Each frame overlaps only with ~2 neighbors on each side.

Drift: xi_x[i] = i * nbr_step (linear accumulation along scan).
  - Local pairwise diff between adjacent frames = nbr_step (small, fixed)
  - Global absolute drift = N * nbr_step (grows with N)

Hypothesis: coupled solver should outperform diagonal when global drift is large
but local pairwise diffs are small, because it uses the pairwise overlap structure
rather than comparing each frame against the (corrupted) global object estimate.

Usage:
  python -u test_longrange_drift.py           # full run
  python -u test_longrange_drift.py --dry     # dry run: N=10,20 only
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--dry', action='store_true')
args = parser.parse_args()

import config
import numpy as np
if config.GPU:
    import cupy as cp
    xp = cp
else:
    xp = np

from PIL import Image
from Operators import make_probe, map_frames, Splitc
from position_retrieval import (
    probe_derivatives, shift_probe_fourier, apodize_probe, apodization_mask,
    position_solve_diag, position_solve_coupled,
    position_plan, shift_rmse,
)

# ---- Fixed geometry ----
nx       = 32
step     = 16        # 50% overlap: neighbors overlap by nx/2
r1       = 0.075
r2       = 0.195     # r2*step=3.12 ... wait, step here is pixels not fraction
                     # resonance: r2*step_fraction where step_fraction = step/nx = 0.5
                     # r2*step_fraction = 0.195*0.5 = 0.098 << 1, well below resonance
NBR_STEP = 0.10      # per-frame drift increment (px) — fixed
NITER    = 60
CAPTURE_THR = 0.10
N_values = [10, 20] if args.dry else [10, 20, 50, 100, 200]

# ---- Probe (CPU, then move to xp) ----
probe_np, _ = make_probe(nx, nx, r1=r1, r2=r2)
if config.GPU:
    probe_np = probe_np.get()
probe_np = np.asarray(probe_np, dtype=np.complex128)
probe_np = (probe_np / np.abs(probe_np).max()).astype(np.complex64)
apo = np.asarray(apodization_mask(nx, nx).get() if config.GPU else apodization_mask(nx, nx))
probe_np = (probe_np * apo).astype(np.complex64)
probe = xp.asarray(probe_np)

# ---- Gold-balls object loader ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_density_full = np.array(
    Image.open(os.path.join(_HERE, '..', 'data', 'gold_balls.png')), np.float32) / 63.0
_obj_full = np.exp(0.69 * (-1.0 + 0.5j) * _density_full).astype(np.complex64)

def make_truth(Nx, Ny):
    h, w = _obj_full.shape
    r0 = max(0, (h - Nx) // 2); c0 = max(0, (w - Ny) // 2)
    patch = _obj_full[r0:r0+Nx, c0:c0+Ny]
    # tile if object is smaller than needed
    reps_x = int(np.ceil(Nx / patch.shape[0])) + 1
    reps_y = int(np.ceil(Ny / patch.shape[1])) + 1
    tiled = np.tile(patch, (reps_x, reps_y))[:Nx, :Ny]
    return xp.asarray(tiled)

mode = 'DRY RUN' if args.dry else 'FULL RUN'
print(f"\n{'='*72}")
print(f"Long-range drift benchmark v2 [{mode}]  GPU={config.GPU}")
print(f"nx={nx}, step={step} ({step/nx*100:.0f}% overlap), r2={r2}, NBR_STEP={NBR_STEP} px")
print(f"r2*(step/nx)={r2*step/nx:.3f}  (resonance threshold ~1.0, well below)")
print(f"{'='*72}\n")
sys.stdout.flush()

print(f"{'N':>5} | {'nframes':>7} | {'total_drift':>11} | {'dr0':>7} | "
      f"{'diag':>10} ok? time | {'coupled':>10} ok? time")
print('-' * 78)

for N in N_values:
    nframes = N
    Nx = Ny = N * step + nx   # square object, scan uses 1D strip along x

    truth = make_truth(Nx, Ny)

    # Scan along x-axis (tx varies), all frames at ty=0
    tx_np = np.arange(N, dtype=np.float64) * step
    ty_np = np.zeros(N, dtype=np.float64)
    tx = xp.asarray(tx_np); ty = xp.asarray(ty_np)

    # Linear accumulating drift along scan (x) direction, mean-removed
    xi_x_np = (np.arange(N) - (N-1)/2.0) * NBR_STEP
    xi_y_np = np.zeros(N)
    total_drift = xi_x_np.max() - xi_x_np.min()
    xi_x = xp.asarray(xi_x_np); xi_y = xp.asarray(xi_y_np)

    dp = probe_derivatives(probe)
    probe_shifted = shift_probe_fourier(probe, xi_x, xi_y)
    mapid = map_frames(tx, ty, nx, nx, Nx, Ny)
    frames = Splitc(truth, mapid) * probe_shifted
    plan = position_plan(tx, ty, nframes, nx, nx, Nx, Ny)
    dr0 = float(shift_rmse(xi_x, xi_y, xp.zeros(nframes, dtype=xp.float64), xp.zeros(nframes, dtype=xp.float64)))

    results = {}
    for name in ['diag', 'coupled']:
        hx = xp.zeros(nframes, dtype=xp.float64)
        hy = xp.zeros(nframes, dtype=xp.float64)
        t0 = time.time()
        for _ in range(NITER):
            if name == 'diag':
                hx, hy = position_solve_diag(
                    frames, dp, truth, mapid, Nx, Ny, hx, hy, max_step=0.5)
            else:
                hx, hy = position_solve_coupled(
                    frames, dp, truth, mapid, Nx, Ny, hx, hy, plan,
                    max_step=0.5, lam=0.0)
        elapsed = time.time() - t0
        results[name] = (float(shift_rmse(xi_x, xi_y, hx, hy)), elapsed)

    dr_d, t_d = results['diag']
    dr_c, t_c = results['coupled']
    ok_d = 'YES' if dr_d < CAPTURE_THR * dr0 else 'NO '
    ok_c = 'YES' if dr_c < CAPTURE_THR * dr0 else 'NO '
    print(f"{N:>5} | {nframes:>7} | {total_drift:>11.2f} | {dr0:>7.3f} | "
          f"{dr_d:>10.3e} {ok_d} {t_d:>4.1f}s | {dr_c:>10.3e} {ok_c} {t_c:>4.1f}s")
    sys.stdout.flush()
