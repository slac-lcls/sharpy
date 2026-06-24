#!/usr/bin/env python3
"""Phase C (mitigation #1): does cropping the aliased frame BORDERS from the Gramian
inner products (the bw / border-width flag in Gramiam_plan -> zQQz.cu) improve sync's
NOISE robustness? The exit-wave autocorrelation overruns the frame, so the outer pixels
are aliased + low-SNR (probe tail); excluding them should give a cleaner relative-phase
measurement under photon noise. Sweep photons/frame x bw; metric = low-freq-band NMSE.

  srun ... python sync_noise_bw.py
"""
import os, sys
os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config; assert config.GPU
import cupy as cp, numpy as np
from cupyx.scipy.sparse.linalg import eigsh
import Operators
from Operators import make_probe, make_translations, Gramiam_plan, eig_reset, xp
from wrap_ops import overlap_cuda, split_cuda
import Solvers

nx = ny = 16; Dx = 5; nnx = 64; r2 = 0.40; CYC = 1.0; maxiter = 300; LOWPASS = 0.10
photons = [1000., 300., 100., 30.]
bws = [0, 1, 2, 3, 4]
seeds = [0, 1]
_power = Operators.Eigensolver

probe = make_probe(nx, ny, r1=0.025 * 3, r2=r2, fx=0., fy=0.)
probe = xp.asarray(probe[0] if isinstance(probe, tuple) else probe).astype(xp.complex64)
tx, ty = make_translations(Dx, Dx, nnx, nnx, nnx * Dx, nnx * Dx)
tx = xp.asarray(tx).astype(xp.int64); ty = xp.asarray(ty).astype(xp.int64)
Nx = int(xp.ceil(xp.max(tx) - xp.min(tx))); nf = tx.size
tr = (tx + 1j * ty).astype(xp.complex64)
xx = (xp.arange(Nx) - Nx / 2).astype(xp.float32); R2 = xx[:, None] ** 2 + xx[None, :] ** 2
truth = xp.exp(1j * (CYC * 2 * np.pi / float(R2.max())) * R2).astype(xp.complex64)
frd = xp.zeros((nf, nx, ny), xp.complex64); split_cuda(truth, frd, tr, probe)
data_clean = (xp.abs(xp.fft.fft2(frd)) ** 2).astype(xp.float32)
kx = (xp.arange(Nx) - Nx // 2)
KK = xp.sqrt(kx[:, None].astype(xp.float32) ** 2 + kx[None, :].astype(xp.float32) ** 2)
lp = xp.fft.ifftshift((KK <= LOWPASS * Nx / 2).astype(xp.float32))

# one plan per bw (border-width crop on the Gramian inner products)
plans = {bw: Gramiam_plan(tx, ty, nf, nx, ny, Nx, Nx, bw=bw) for bw in bws}


def lf_nmse(img):
    t = xp.fft.ifft2(lp * xp.fft.fft2(truth)); i = xp.fft.ifft2(lp * xp.fft.fft2(img))
    s = xp.vdot(i, t) / (xp.vdot(i, i) + 1e-30)
    return float(xp.linalg.norm(s * i - t) / xp.linalg.norm(t))

def gauge(w, n):
    w = w.ravel() / (cp.abs(w.ravel()) + 1e-30); s = cp.sum(w)
    return cp.reshape(w * cp.conj(s) / (cp.abs(s) + 1e-30), (n, 1, 1))

def eigsh_cupy(H, num_iter, v0=None, tol=1e-6):
    lam, V = eigsh(H, k=1, ncv=3, maxiter=20, which="LM"); return gauge(V[:, int(cp.argmax(lam))], H.shape[0])

def recon(plan, data):
    Operators.Eigensolver = eigsh_cupy; eig_reset()
    img, *_ = Solvers.Alternating_projections_c(
        True, xp.ones((Nx, Nx), xp.complex64), plan, probe + 0, tx, ty, overlap_cuda, split_cuda,
        data, False, maxiter, None, truth, 1, sync_interval=1, num_iter=5)
    return img


print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
      "| frames", nf, "r2", r2, "| bw sweep", bws)
print(f"  {'phot/fr':>8} |" + "".join(f"  bw={bw:<6}" for bw in bws))
for P in photons:
    row = []
    for bw in bws:
        acc = []
        for sd in seeds:
            cp.random.seed(sd)
            scale = P * nf / float(data_clean.sum())
            dn = (cp.random.poisson(cp.asarray(data_clean) * scale).astype(cp.float32)) / scale
            acc.append(lf_nmse(recon(plans[bw], dn)))
        row.append(float(np.mean(acc)))
    print(f"  {P:>8.0f} |" + "".join(f"  {v:>8.3f}" for v in row))
print("\nLower = better. If bw>0 lowers lf-NMSE at low photons, cropping the aliased")
print("borders improves sync's noise robustness (spatial weighting via the Gramian).")
