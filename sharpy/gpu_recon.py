"""Real AP reconstruction on GPU through the bincount/sparse operator path.

Exercises Solvers.Alternating_projections + Split_Overlap_plan (the path the
numpy_groupies->xp.bincount fix touches) on simulated ptychography data.
Runs on whatever config.GPU selects (cupy on Perlmutter). sync=False so it
needs no Gramian-CUDA. Reports object NMSE, data residual, and timing for a
plain AP run vs the Newton line-search accelerated run.

Run on a GPU node:
  srun -A lcls_g -C gpu -q interactive -N1 -n1 --gpus 1 -t 00:10:00 python gpu_recon.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import numpy as np
from Operators import Split_Overlap_plan, xp
import Solvers


def phantom(Nx, Ny, contrast=1.5, seed=0):
    """Smooth textured complex object (amp+phase), built in numpy then moved
    to device -- avoids position_simulate.transmission_object, which routes a
    numpy array through Operators.cropmat (xp) and breaks on GPU."""
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.standard_normal((Nx, Ny)))
    k = np.fft.fftfreq(Nx)
    KX, KY = np.meshgrid(k, k)
    sm = np.real(np.fft.ifft2(F * np.exp(-(KX**2 + KY**2) / (2 * 0.03**2))))
    sm = (sm - sm.min()) / (sm.max() - sm.min())
    obj = (0.5 + 0.5 * sm) * np.exp(1j * contrast * (sm - 0.5))
    return obj.astype(np.complex64)

print("xp =", xp.__name__, "| config.GPU =", config.GPU)
if config.GPU:
    import cupy as cp
    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print("device:", name)

# ---- geometry: object > scan-covered region, full raster coverage ----
Nx = Ny = 256
nx = ny = 64
K, step = 16, 16                      # dense overlap (frame 64) -> well-conditioned AP
g = xp.arange(K) * step
tx, ty = xp.meshgrid(g, g, indexing="ij")
tx = tx.ravel().astype(float); ty = ty.ravel().astype(float)
nframes = tx.size

# ---- truth object + smooth Gaussian probe (well-conditioned, no Fourier zeros) ----
truth = xp.asarray(phantom(Nx, Ny, contrast=1.5)).astype(xp.complex64)
c = nx // 2
gx = xp.arange(nx) - c
X, Y = xp.meshgrid(gx, gx)
probe = xp.exp(-(X**2 + Y**2) / (2.0 * (0.18 * nx) ** 2)).astype(xp.complex64)
probe = (probe / xp.abs(probe).max()).astype(xp.complex64)

# ---- simulate diffraction data (intensity) through the GPU operators ----
Split, Overlap = Split_Overlap_plan(tx, ty, nx, ny, Nx, Ny)
exit_waves = Split(truth) * probe[None]
data = (xp.abs(xp.fft.fft2(exit_waves)) ** 2).astype(xp.float32)
print("geometry: img %s, frames %s x %d, data %.1f MB"
      % ((Nx, Ny), (nx, ny), nframes, data.nbytes / 1e6))


def run(line_search, maxiter=int(os.environ.get("SHARPY_MAXITER", 250))):
    img0 = xp.ones((Nx, Ny), dtype=xp.complex64)
    if config.GPU:
        cp.cuda.Stream.null.synchronize()
    t0 = time.time()
    img, frames, illum, res = Solvers.Alternating_projections(
        False,            # sync
        img0, None,       # img, Gramiam
        probe + 0,
        Overlap, Split,
        data,
        False,            # refine_illumination (known probe)
        maxiter,
        None,             # normalization
        truth,            # img_truth
        1,                # residuals_interval
        line_search=line_search,
    )
    if config.GPU:
        cp.cuda.Stream.null.synchronize()
    dt = time.time() - t0
    r = res.get() if config.GPU else res
    return r, dt


for ls in (False, True):
    r, dt = run(line_search=ls)
    tag = "line_search" if ls else "plain AP   "
    print(">>> %s : obj NMSE=%.3e  data-resid=%.3e  %d iters  %.2fs  %.1f ms/iter"
          % (tag, r[-1, 0], r[-1, 1], r.shape[0], dt, 1e3 * dt / r.shape[0]))
print("GPU RECONSTRUCTION OK")
