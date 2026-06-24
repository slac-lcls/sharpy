#!/usr/bin/env python3
"""Noise-feasibility envelope for synchronization (Phase A: threshold at fixed N).

How few photons/frame before sync stops recovering the LONG-RANGE (low-frequency)
phase? Physical Poisson photon noise on the diffraction; AP reconstruction from a flat
start; metric = LOW-FREQUENCY-BAND NMSE of the recovered image vs truth (the long-range
phase that synchronization is responsible for; local overlap alone cannot build it).
Compares no-sync / power-sync / eigsh-sync across a photons/frame sweep.

  CYCLES=1.0 R2=0.40 PHOTONS="3000 1000 300 100 30 10 3" SHARPY_MAXITER=300 \
    srun -A lcls_g -C gpu -q interactive -N1 -n1 --gpus 1 -t 00:20:00 python sync_noise_envelope.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config; assert config.GPU
import cupy as cp, numpy as np
from cupyx.scipy.sparse.linalg import eigsh
import Operators
from Operators import make_probe, make_translations, Gramiam_plan, eig_reset, xp
from wrap_ops import overlap_cuda, split_cuda
import Solvers

nx = ny = 16; Dx = 5; nnx = int(os.environ.get("NNX", 64))
r2 = float(os.environ.get("R2", 0.40)); maxiter = int(os.environ.get("SHARPY_MAXITER", 300))
CYC = float(os.environ.get("CYCLES", 1.0))
photons = [float(p) for p in os.environ.get("PHOTONS", "3000 1000 300 100 30 10 3").split()]
LOWPASS = float(os.environ.get("LOWPASS", 0.10))   # low-freq band radius (fraction of Nyquist)
SEED = int(os.environ.get("SEED", 0)); cp.random.seed(SEED)

probe = make_probe(nx, ny, r1=0.025 * 3, r2=r2, fx=0.0, fy=0.0)
probe = xp.asarray(probe[0] if isinstance(probe, tuple) else probe).astype(xp.complex64)
tx, ty = make_translations(Dx, Dx, nnx, nnx, nnx * Dx, nnx * Dx)
tx = xp.asarray(tx).astype(xp.int64); ty = xp.asarray(ty).astype(xp.int64)
Nx = int(xp.ceil(xp.max(tx) - xp.min(tx))); nf = tx.size
tr = (tx + 1j * ty).astype(xp.complex64)
G = Gramiam_plan(tx, ty, nf, nx, ny, Nx, Nx, bw=0)
xx = (xp.arange(Nx) - Nx / 2).astype(xp.float32); R2 = xx[:, None] ** 2 + xx[None, :] ** 2
alpha = CYC * 2 * np.pi / float(R2.max())
truth = xp.exp(1j * alpha * R2).astype(xp.complex64)         # unit-amplitude: phase only
frd = xp.zeros((nf, nx, ny), xp.complex64); split_cuda(truth, frd, tr, probe)
data_clean = (xp.abs(xp.fft.fft2(frd)) ** 2).astype(xp.float32)

kx = (xp.arange(Nx) - Nx // 2)
KK = xp.sqrt(kx[:, None].astype(xp.float32) ** 2 + kx[None, :].astype(xp.float32) ** 2)
lpmask = xp.fft.ifftshift((KK <= LOWPASS * Nx / 2).astype(xp.float32))   # centered low-pass disk


def band_nmse(img, mask=None):
    t = truth if mask is None else xp.fft.ifft2(mask * xp.fft.fft2(truth))
    i = img if mask is None else xp.fft.ifft2(mask * xp.fft.fft2(img))
    s = xp.vdot(i, t) / (xp.vdot(i, i) + 1e-30)               # remove global phase/scale
    return float(xp.linalg.norm(s * i - t) / xp.linalg.norm(t))


def gauge(w, n):
    w = w.ravel() / (cp.abs(w.ravel()) + 1e-30); s = cp.sum(w)
    return cp.reshape(w * cp.conj(s) / (cp.abs(s) + 1e-30), (n, 1, 1))

def eigsh_cupy(H, num_iter, v0=None, tol=1e-6):
    lam, V = eigsh(H, k=1, ncv=3, maxiter=20, which="LM"); return gauge(V[:, int(cp.argmax(lam))], H.shape[0])

_power = Operators.Eigensolver


def run(sync, solver, data):
    Operators.Eigensolver = solver if solver else _power; eig_reset()
    img, frames, illum, res = Solvers.Alternating_projections_c(
        sync, xp.ones((Nx, Nx), xp.complex64), G, probe + 0, tx, ty, overlap_cuda,
        split_cuda, data, False, maxiter, None, truth, 1, sync_interval=1, num_iter=5)
    return img


print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
      "| frames", nf, "img", Nx, "cyc", CYC, "lowpass", LOWPASS, "maxiter", maxiter)
print("  (lf = low-freq-band NMSE = long-range phase; full = whole-image NMSE)")
print(f"  {'phot/fr':>8}{'SNR':>7} | {'nosync lf':>10}{'power lf':>10}{'eigsh lf':>10} | "
      f"{'nosync full':>12}{'eigsh full':>11}")
for P in photons:
    scale = P * nf / float(data_clean.sum())                  # avg photons/frame = P
    data_n = (cp.random.poisson(cp.asarray(data_clean) * scale).astype(cp.float32)) / scale
    snr = float(xp.linalg.norm(data_clean.ravel()) / (xp.linalg.norm((data_n - data_clean).ravel()) + 1e-30))
    out = {}
    for label, sync, solver in [("nosync", False, _power), ("power", True, _power), ("eigsh", True, eigsh_cupy)]:
        img = run(sync, solver, data_n); out[label] = (band_nmse(img, lpmask), band_nmse(img, None))
    print(f"  {P:>8.0f}{snr:>7.1f} | {out['nosync'][0]:>10.3f}{out['power'][0]:>10.3f}{out['eigsh'][0]:>10.3f} | "
          f"{out['nosync'][1]:>12.3f}{out['eigsh'][1]:>11.3f}")

print("\nThreshold = photons/frame where eigsh-sync lf-NMSE stops beating no-sync")
print("(below it, sync no longer recovers the long-range phase -> noise-limited, not compute-limited).")
