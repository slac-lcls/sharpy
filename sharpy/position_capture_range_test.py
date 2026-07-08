"""
Capture range of coupled vs diagonal vs gradient solvers.

Uses realistic simulated data from position_simulate.py:
  - Transmission object: gold-balls Beer-Lambert (|obj| ~ 0.5-1)
  - Probe: apodized zone-plate (r1=0.075, r2=0.195, below resonance)
  - Scan: hexagonal grid, step=3.5

Part 1: i.i.d. shift pattern — sweeps max|xi|.
Part 2: ramp (smooth drift) shift pattern.
Part 3: convergence speed on ramp at max|xi|=0.3 px (diag vs coupled).

Run:  python position_capture_range_test.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import numpy as np
if config.GPU:
    import cupy as cp
    xp = cp
else:
    xp = np

from Operators import make_probe, map_frames, Splitc
from position_retrieval import (
    probe_derivatives, shift_probe_fourier, apodize_probe,
    position_solve_diag, position_solve_coupled,
    position_solve_gradient, position_plan, shift_rmse,
)


# -------------------------------------------------------------------------
# Experiment parameters  (matching position_simulate.py defaults)
# -------------------------------------------------------------------------
nx   = 32
nnx  = 36
step = 3.5
r1   = 0.075
r2   = 0.195   # r2*step=0.68, below resonance threshold 0.70
Nx   = Ny = int(round(nnx * step))

CAPTURE_THRESHOLD = 0.10
sigma_vals = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
solvers    = [('diag', 40), ('coupled', 40), ('gradient', 400)]

# -------------------------------------------------------------------------
# Build probe and object (inline, no h5)
# -------------------------------------------------------------------------
probe_np, _ = make_probe(nx, nx, r1=r1, r2=r2)
if config.GPU:
    probe_np = probe_np.get()
probe_np = np.asarray(probe_np, dtype=np.complex128)
probe_np = (probe_np / np.abs(probe_np).max()).astype(np.complex64)
# apodize on CPU (apodize_probe uses xp internally; force numpy path)
from position_retrieval import apodization_mask as _apo_mask
apo = np.asarray(_apo_mask(nx, nx).get() if config.GPU else _apo_mask(nx, nx))
probe_np = (probe_np * apo).astype(np.complex64)
probe = xp.asarray(probe_np)

from PIL import Image
_HERE = os.path.dirname(os.path.abspath(__file__))
_density = np.array(Image.open(os.path.join(_HERE, '..', 'data', 'gold_balls.png')), np.float32) / 63.0
_obj = np.exp(0.69 * (-1.0 + 0.5j) * _density).astype(np.complex64)
# crop on CPU
_h, _w = _obj.shape
_r0 = (_h - Nx) // 2; _c0 = (_w - Ny) // 2
truth_np = _obj[_r0:_r0+Nx, _c0:_c0+Ny]

# Optional wide Gaussian phase blob multiplied onto the object.
# Controls phase_amp (radians) and blob width relative to object size.
PHASE_AMP   = 2.0   # peak phase in radians (0 = off)
PHASE_WIDTH = 0.3   # sigma as fraction of object width
if PHASE_AMP > 0:
    yy, xx = np.meshgrid(np.arange(Ny) - Ny/2, np.arange(Nx) - Nx/2, indexing='ij')
    sigma_px = PHASE_WIDTH * Nx
    gauss = np.exp(-(xx**2 + yy**2) / (2 * sigma_px**2)).astype(np.float32)
    truth_np = (truth_np * np.exp(1j * PHASE_AMP * gauss)).astype(np.complex64)

truth = xp.asarray(truth_np)

# Hexagonal scan grid (same as position_simulate.py)
ix1 = np.arange(nnx) * step
iy1 = np.arange(nnx) * step
ix, iy = np.meshgrid(ix1, iy1, indexing='ij')
ix = ix + np.floor(step / 2) * (np.arange(1, nnx + 1) % 2)[:, None]
tx_np = np.round(ix).ravel().astype(np.float64)
ty_np = np.round(iy).ravel().astype(np.float64)
tx = xp.asarray(tx_np)
ty = xp.asarray(ty_np)
nframes = len(tx_np)
gx = ix.ravel()   # scan x float positions for ramp (CPU)

print(f"Probe: r1={r1}, r2={r2}, r2*step={r2*step:.2f}  "
      f"nx={nx}, nnx={nnx}, step={step}, nframes={nframes}, Nx={Nx}")
print(f"Object: gold-balls transmission, |obj| range "
      f"{float(np.abs(truth_np).min()):.3f}..{float(np.abs(truth_np).max()):.3f}  "
      f"phase_amp={PHASE_AMP} rad  phase_width={PHASE_WIDTH}")
sys.stdout.flush()


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def build_scene(xi_x, xi_y):
    dp = probe_derivatives(probe)
    probe_shifted = shift_probe_fourier(probe, xi_x, xi_y)
    mapid = map_frames(tx, ty, nx, nx, Nx, Ny)
    frames = Splitc(truth, mapid) * probe_shifted
    frames_data = np.abs(np.fft.fft2(frames)) ** 2
    plan = position_plan(tx, ty, nframes, nx, nx, Nx, Ny)
    return dict(dp=dp, truth=truth, mapid=mapid, frames=frames,
                frames_data=frames_data, plan=plan,
                xi_x=xi_x, xi_y=xi_y, nframes=nframes, Nx=Nx, Ny=Ny)


def run_solver(name, scene, niter):
    s = scene
    hx = xp.zeros(s['nframes'])
    hy = xp.zeros(s['nframes'])
    for _ in range(niter):
        if name == 'diag':
            hx, hy = position_solve_diag(
                s['frames'], s['dp'], s['truth'], s['mapid'],
                s['Nx'], s['Ny'], hx, hy, max_step=0.5)
        elif name == 'coupled':
            hx, hy = position_solve_coupled(
                s['frames'], s['dp'], s['truth'], s['mapid'],
                s['Nx'], s['Ny'], hx, hy, s['plan'], max_step=0.5, lam=0.0)
        elif name == 'gradient':
            hx, hy = position_solve_gradient(
                s['frames_data'], s['truth'], s['dp'], s['mapid'],
                hx, hy, max_step=0.5)
    return shift_rmse(s['xi_x'], s['xi_y'], hx, hy)


def iters_to_threshold(name, scene, threshold, max_iters):
    s = scene
    hx = xp.zeros(s['nframes'])
    hy = xp.zeros(s['nframes'])
    for i in range(1, max_iters + 1):
        if name == 'diag':
            hx, hy = position_solve_diag(
                s['frames'], s['dp'], s['truth'], s['mapid'],
                s['Nx'], s['Ny'], hx, hy, max_step=0.5)
        elif name == 'coupled':
            hx, hy = position_solve_coupled(
                s['frames'], s['dp'], s['truth'], s['mapid'],
                s['Nx'], s['Ny'], hx, hy, s['plan'], max_step=0.5, lam=0.0)
        if shift_rmse(s['xi_x'], s['xi_y'], hx, hy) < threshold:
            return i
    return max_iters


# -------------------------------------------------------------------------
# Part 1: i.i.d. shifts
# -------------------------------------------------------------------------
print("\n" + "#"*72)
print("PART 1: i.i.d. shifts (absolute per-frame, no correlation)")
print("#"*72)

header = f"{'max|xi|':>8} | {'dr0':>7}"
for name, _ in solvers:
    header += f" | {name:>9} ok?"
print(header); print('-' * (len(header)+4))

rng = np.random.default_rng(3)
for sigma in sigma_vals:
    xi_x = xp.asarray(rng.standard_normal(nframes) * sigma)
    xi_y = xp.asarray(rng.standard_normal(nframes) * sigma)
    scene = build_scene(xi_x, xi_y)
    dr0 = shift_rmse(xi_x, xi_y, xp.zeros(nframes), xp.zeros(nframes))
    row = f"{sigma:>8.1f} | {dr0:>7.3f}"
    for name, niter in solvers:
        dr = run_solver(name, scene, niter)
        row += f" | {dr:>7.3e} {'YES' if dr < CAPTURE_THRESHOLD*dr0 else 'NO ':>3}"
    print(row)
    sys.stdout.flush()

# -------------------------------------------------------------------------
# Part 2: ramp shifts
# -------------------------------------------------------------------------
print("\n" + "#"*72)
print("PART 2: ramp shifts (smooth drift, small pairwise, large absolute)")
print("        max|xi| = half-range of ramp; pairwise diff = max|xi|*2/(nnx-1)")
print("#"*72)

header = f"{'max|xi|':>8} | {'dr0':>7} | {'nbr diff':>8}"
for name, _ in solvers:
    header += f" | {name:>9} ok?"
print(header); print('-' * (len(header)+4))

g = gx - gx.mean()
g_scale = np.abs(g).max()
for max_abs in sigma_vals:
    xi_x = xp.asarray(g / g_scale * max_abs)
    xi_y = xp.zeros(nframes)
    nbr_diff = max_abs * 2.0 / (nnx - 1)
    scene = build_scene(xi_x, xi_y)
    dr0 = shift_rmse(xi_x, xi_y, xp.zeros(nframes), xp.zeros(nframes))
    row = f"{max_abs:>8.1f} | {dr0:>7.3f} | {nbr_diff:>8.3f}"
    for name, niter in solvers:
        dr = run_solver(name, scene, niter)
        row += f" | {dr:>7.3e} {'YES' if dr < CAPTURE_THRESHOLD*dr0 else 'NO ':>3}"
    print(row)
    sys.stdout.flush()

# -------------------------------------------------------------------------
# Part 3: convergence speed on ramp at max|xi|=0.3 px
# -------------------------------------------------------------------------
print("\n" + "#"*72)
print("PART 3: convergence speed on ramp at max|xi|=0.3 px (diag vs coupled)")
print("        iters to reach Delta_r < 1e-6")
print("#"*72)
print(f"\n{'solver':>10} | {'iters':>6}")
print('-' * 22)
xi_x = xp.asarray(g / g_scale * 0.3)
xi_y = xp.zeros(nframes)
scene = build_scene(xi_x, xi_y)
thr = 1e-6
for name in ['diag', 'coupled']:
    n_it = iters_to_threshold(name, scene, thr, max_iters=200)
    print(f"{name:>10} | {n_it:>6}")
