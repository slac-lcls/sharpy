"""
End-to-end test: coupled vs diagonal in the full AP+position loop.

Geometry: 1D scan, step=16 (50% overlap), N frames, linear drift.
Starts from a blurry initial image (wrong positions) so diagonal suffers
from corrupted object estimate while coupled uses frame-pair correlations.

Usage:
  python -u test_coupled_e2e.py           # N=50
  python -u test_coupled_e2e.py --dry     # N=20, fewer iters
"""
import os, sys, argparse
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
from Operators import make_probe, map_frames, Splitc, Overlapc
from position_retrieval import (
    probe_derivatives, shift_probe_fourier, apodize_probe, apodization_mask,
    shift_rmse,
)
from Solvers import Alternating_projections_position

# ---- Parameters ----
nx       = 32
step     = 16          # 50% overlap
r1       = 0.075
r2       = 0.195
NBR_STEP = 0.10        # per-frame drift increment (px)
N        = 20 if args.dry else 50
AP_WARMUP   = 20       # AP iters before position retrieval starts
AP_TOTAL    = 60 if args.dry else 200
POS_EVERY   = 1        # position update every iteration
MAX_STEP    = 0.5

Nx = Ny = N * step + nx   # square object, scan along x

# ---- Probe ----
probe_np, _ = make_probe(nx, nx, r1=r1, r2=r2)
if config.GPU: probe_np = probe_np.get()
probe_np = np.asarray(probe_np, dtype=np.complex128)
probe_np = (probe_np / np.abs(probe_np).max()).astype(np.complex64)
apo = np.asarray(apodization_mask(nx, nx).get() if config.GPU else apodization_mask(nx, nx))
probe_np = (probe_np * apo).astype(np.complex64)
probe = xp.asarray(probe_np)

# ---- Object (gold balls, tiled) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_density = np.array(Image.open(os.path.join(_HERE, '..', 'data', 'gold_balls.png')), np.float32) / 63.0
_obj = np.exp(0.69 * (-1.0 + 0.5j) * _density).astype(np.complex64)
h, w = _obj.shape
reps_x = int(np.ceil(Nx / h)) + 1
reps_y = int(np.ceil(Ny / w)) + 1
truth_np = np.tile(_obj, (reps_x, reps_y))[:Nx, :Ny]
truth = xp.asarray(truth_np)

# ---- Scan grid (1D along x) ----
tx_np = np.arange(N, dtype=np.float64) * step
ty_np = np.zeros(N, dtype=np.float64)
tx = xp.asarray(tx_np); ty = xp.asarray(ty_np)
nframes = N

# ---- True shifts: linear accumulating drift ----
xi_x_truth_np = (np.arange(N) - (N-1)/2.0) * NBR_STEP
xi_y_truth_np = np.zeros(N)
total_drift = xi_x_truth_np.max() - xi_x_truth_np.min()
xi_x_truth = xp.asarray(xi_x_truth_np)
xi_y_truth = xp.asarray(xi_y_truth_np)

print(f"\n{'='*68}")
print(f"E2E test: N={N}, step={step}, nx={nx}, Nx={Nx}")
print(f"Drift: NBR_STEP={NBR_STEP} px, total_drift={total_drift:.2f} px")
print(f"AP: warmup={AP_WARMUP}, total={AP_TOTAL}, pos_every={POS_EVERY}")
print(f"GPU={config.GPU}")
print(f"{'='*68}\n")
sys.stdout.flush()

# ---- Simulate data with true shifts ----
mapid = map_frames(tx, ty, nx, nx, Nx, Ny)
probe_shifted = shift_probe_fourier(probe, xi_x_truth, xi_y_truth)
frames_clean = Splitc(truth, mapid) * probe_shifted
frames_data = xp.abs(xp.fft.fft2(frames_clean)) ** 2

# ---- Blurry initial image: overlap with WRONG positions (xi=0) ----
probe_flat = xp.broadcast_to(probe[xp.newaxis], (nframes, nx, nx))
normalization = Overlapc(xp.abs(probe_flat)**2, Nx, Ny, mapid)
nrm = xp.where(xp.abs(normalization) < 1e-6 * float(xp.abs(normalization).max()),
               1.0, normalization)
# use measured amplitudes with flat probe → blurry image
frames_init = xp.sqrt(frames_data) * xp.exp(1j * xp.angle(frames_clean))
img0 = Overlapc(frames_init * xp.conj(probe_flat), Nx, Ny, mapid) / nrm

dr0 = float(shift_rmse(xi_x_truth, xi_y_truth, xp.zeros(nframes), xp.zeros(nframes)))
print(f"Initial position RMSE (no correction): {dr0:.4f} px")
print(f"Initial image quality: ||img0-truth||/||truth|| = "
      f"{float(xp.linalg.norm(img0-truth)/xp.linalg.norm(truth)):.4f}\n")
sys.stdout.flush()

# ---- Run diag and coupled, with several initial conditions ----
# Case A: xi_init=0 (blind cold start — tests whether AP+position can self-bootstrap)
# Case B: xi_init=truth (oracle warm start — upper bound on performance)
# Case C: xi_init=truth+noise (noisy warm start — realistic near-truth scenario)
rng = np.random.default_rng(42)
NOISE_SIGMA = 0.3   # px, added on top of truth for case C

test_cases = [
    ("blind",  None),
    ("oracle", xi_x_truth_np),
    ("noisy",  xi_x_truth_np + rng.standard_normal(N).astype(np.float64) * NOISE_SIGMA),
]

for method in ['diag', 'coupled']:
    print(f"--- {method.upper()} ---")
    for case_name, xi_init_np in test_cases:
        xi_init = xp.asarray(xi_init_np) if xi_init_np is not None else None
        xi_init_y = xp.zeros(nframes) if xi_init is not None else None

        dr_init = float(shift_rmse(xi_x_truth, xi_y_truth,
                                   xi_init if xi_init is not None else xp.zeros(nframes),
                                   xi_init_y if xi_init_y is not None else xp.zeros(nframes)))

        img_out, frames_out, xi_x_out, xi_y_out, residuals = \
            Alternating_projections_position(
                img0 + 0.0,
                probe,
                frames_data,
                tx + 0.0, ty + 0.0,
                nx, nx, Nx, Ny,
                maxiter=AP_TOTAL,
                position_start=AP_WARMUP,
                position_every=POS_EVERY,
                max_step=MAX_STEP,
                method=method,
                xi_x_init=xi_init,
                xi_y_init=xi_init_y,
                img_truth=truth,
                xi_x_truth=xi_x_truth,
                xi_y_truth=xi_y_truth,
                residuals_interval=20,
            )

        dr_final = float(shift_rmse(xi_x_truth, xi_y_truth, xi_x_out, xi_y_out))
        img_err  = float(xp.linalg.norm(img_out - truth) / xp.linalg.norm(truth))
        res_np = np.array(residuals.get() if config.GPU else residuals)
        print(f"  [{case_name:6s}] dr_init={dr_init:.3f} → dr_final={dr_final:.4f}  img_err={img_err:.4f}")
        xi_traj = "  ".join(f"{row[3]:.4f}" for row in res_np[::2])
        print(f"           xi_err @ 0,40,80,...: {xi_traj}")
        sys.stdout.flush()
    print()
