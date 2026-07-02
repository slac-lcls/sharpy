"""End-to-end A/B: full AP reconstruction with the fused ProxD kernel ON vs OFF.

Runs Solvers.Alternating_projections (the gpu_recon.py path) for MAXITER iters
twice -- SHARPY_FUSED_PROXD effectively off then on (toggled at runtime via the
module global Operators._FUSED_PROXD, which Project_data reads per-call) -- and
reports: final object NMSE (must match -> recon unchanged), ms/iter, the ProxD
phase time from Solvers.timers, and its fraction of the loop. This tells the
END-TO-END gain (the 3x on the data-prox op is diluted by the FFTs + overlap).

  srun -A lcls_g -C gpu -q interactive -N1 -n1 --gpus 1 -t 15 \
    bash -c 'source ~/sharpy-venv/bin/activate; cd $SCRATCH/sharpy/sharpy; \
             python -u proxd_e2e_test.py'
  env: NX (frame size, default 128), KG (scan grid KxK, default 24), MAXITER (150)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import numpy as np
from Operators import Split_Overlap_plan, xp
import Operators
import Solvers

GPU = config.GPU
if GPU:
    import cupy as cp
    print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())


def phantom(Nx, Ny, contrast=1.5, seed=0):
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.standard_normal((Nx, Ny)))
    k = np.fft.fftfreq(Nx)
    KX, KY = np.meshgrid(k, k)
    sm = np.real(np.fft.ifft2(F * np.exp(-(KX ** 2 + KY ** 2) / (2 * 0.03 ** 2))))
    sm = (sm - sm.min()) / (sm.max() - sm.min())
    return ((0.5 + 0.5 * sm) * np.exp(1j * contrast * (sm - 0.5))).astype(np.complex64)


nx = ny = int(os.environ.get("NX", 128))
K = int(os.environ.get("KG", 24))
step = nx // 4                              # 75% overlap
Nx = Ny = (K - 1) * step + nx              # object covers the scan
g = xp.arange(K) * step
tx, ty = xp.meshgrid(g, g, indexing="ij")
tx = tx.ravel().astype(float); ty = ty.ravel().astype(float)
nframes = tx.size

truth = xp.asarray(phantom(Nx, Ny, 1.5)).astype(xp.complex64)
c = nx // 2
gx = xp.arange(nx) - c
X, Y = xp.meshgrid(gx, gx)
probe = xp.exp(-(X ** 2 + Y ** 2) / (2.0 * (0.18 * nx) ** 2)).astype(xp.complex64)
probe = (probe / xp.abs(probe).max()).astype(xp.complex64)

Split, Overlap = Split_Overlap_plan(tx, ty, nx, ny, Nx, Ny)
data = (xp.abs(xp.fft.fft2(Split(truth) * probe[None])) ** 2).astype(xp.float32)
print(f"img {Nx}x{Ny}, frames {nframes} x {nx}, data {data.nbytes/1e6:.1f} MB, "
      f"fused-kernel size = {nframes*nx*ny/1e6:.1f}M complex")


def sync():
    if GPU:
        cp.cuda.Stream.null.synchronize()


def reset_timers():
    for k in list(Solvers.timers):
        Solvers.timers[k] = 0


def run(fused, maxiter):
    # bool(fused) (not "and GPU") so the fused code path is exercised on CPU too
    # (the kernels fall back, but the reference-trick / _diffnorm branches run).
    Operators._FUSED_PROXD = bool(fused)
    reset_timers()
    img0 = xp.ones((Nx, Ny), dtype=xp.complex64)
    sync(); t0 = time.time()
    img, frames, illum, res = Solvers.Alternating_projections(
        False, img0, None, probe + 0, Overlap, Split, data,
        False, maxiter, None, truth, 1)
    sync(); dt = time.time() - t0
    r = res.get() if GPU else res
    return r[-1, 0], dt, float(Solvers.timers.get("ProxD", 0.0)), dict(Solvers.timers)


MAXITER = int(os.environ.get("MAXITER", 100))
REPS = int(os.environ.get("REPS", 4))
run(False, 5); run(True, 5)                # warmup: cuFFT plans + fused kernel compile

from statistics import median
# ALTERNATE plain/fused across reps so GPU clock-ramp / pool state biases both equally
dts = {False: [], True: []}
proxds = {False: [], True: []}
nmses = {}
for rep in range(REPS):
    for fused in (False, True):
        nmse, dt, proxd, T = run(fused, MAXITER)
        dts[fused].append(dt); proxds[fused].append(proxd); nmses[fused] = nmse

print(f"\nmedian over {REPS} alternating reps, {MAXITER} iters each")
print(f"{'':10} {'NMSE':>11} {'ms/iter':>9} {'ProxD/iter':>11} {'ProxD%loop':>11}")
for fused in (False, True):
    d = median(dts[fused]); p = median(proxds[fused])
    print(f"{'fused' if fused else 'plain':10} {nmses[fused]:>11.4e} "
          f"{1e3*d/MAXITER:>9.3f} {1e3*p/MAXITER:>11.3f} {100*p/d:>10.1f}%")

db, df = median(dts[False]), median(dts[True])
pb, pf = median(proxds[False]), median(proxds[True])
print(f"\nNMSE match: {abs(nmses[False]-nmses[True])/max(abs(nmses[False]),1e-30) < 1e-3}"
      f"  (plain {nmses[False]:.4e} / fused {nmses[True]:.4e})")
print(f"END-TO-END loop speedup: {db/df:.3f}x   ({1e3*db/MAXITER:.3f} -> {1e3*df/MAXITER:.3f} ms/iter)")
print(f"ProxD phase (incl 2 FFTs) speedup: {pb/pf:.3f}x   ({1e3*pb/MAXITER:.3f} -> {1e3*pf/MAXITER:.3f} ms/iter)")
print("note: Solvers.timers['ProxD'] wraps Propagate+middle+IPropagate, so it is FFT-dominated.")
