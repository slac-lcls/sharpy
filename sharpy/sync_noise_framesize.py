#!/usr/bin/env python3
"""How does the photon threshold scale with FRAME SIZE (realistic detectors are 128-256 px,
not 16)? Sweep nx in {16,64,128} at FIXED overlap fraction (step Dx propto nx) and FIXED
frame count, physical Poisson, eigsh-sync vs no-sync, low-freq-band NMSE. Report the
threshold in photons/frame AND photons/pixel to see which is the transferable unit.

  srun -A lcls_g -C gpu -q interactive -N1 -n1 --gpus 1 -t 00:30:00 python sync_noise_framesize.py
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

nnx = 24; CYC = 1.0; r2 = 0.40; maxiter = 300; LOWPASS = 0.10
nx_list = [16, 64, 128]
photons = [10000., 3000., 1000., 300., 100., 30.]
seeds = [0]
_power = Operators.Eigensolver


def gauge(w, n):
    w = w.ravel() / (cp.abs(w.ravel()) + 1e-30); s = cp.sum(w)
    return cp.reshape(w * cp.conj(s) / (cp.abs(s) + 1e-30), (n, 1, 1))

def eigsh_cupy(H, num_iter, v0=None, tol=1e-6):
    lam, V = eigsh(H, k=1, ncv=3, maxiter=20, which="LM"); return gauge(V[:, int(cp.argmax(lam))], H.shape[0])


def build(nx):
    Dx = max(2, int(round(5 * nx / 16)))                  # step propto nx -> ~const overlap fraction
    probe = make_probe(nx, nx, r1=0.025 * 3, r2=r2, fx=0., fy=0.)
    probe = xp.asarray(probe[0] if isinstance(probe, tuple) else probe).astype(xp.complex64)
    tx, ty = make_translations(Dx, Dx, nnx, nnx, nnx * Dx, nnx * Dx)
    tx = xp.asarray(tx).astype(xp.int64); ty = xp.asarray(ty).astype(xp.int64)
    Nx = int(xp.ceil(xp.max(tx) - xp.min(tx))); nf = tx.size
    tr = (tx + 1j * ty).astype(xp.complex64)
    G = Gramiam_plan(tx, ty, nf, nx, nx, Nx, Nx, bw=0)
    deg = float(G['col'].size) * 2.0 / nf
    xx = (xp.arange(Nx) - Nx / 2).astype(xp.float32); R2 = xx[:, None] ** 2 + xx[None, :] ** 2
    truth = xp.exp(1j * (CYC * 2 * np.pi / float(R2.max())) * R2).astype(xp.complex64)
    frd = xp.zeros((nf, nx, nx), xp.complex64); split_cuda(truth, frd, tr, probe)
    data_clean = (xp.abs(xp.fft.fft2(frd)) ** 2).astype(xp.float32)
    kx = (xp.arange(Nx) - Nx // 2)
    KK = xp.sqrt(kx[:, None].astype(xp.float32) ** 2 + kx[None, :].astype(xp.float32) ** 2)
    lp = xp.fft.ifftshift((KK <= LOWPASS * Nx / 2).astype(xp.float32))
    return dict(probe=probe, tx=tx, ty=ty, Nx=Nx, nf=nf, nx=nx, Dx=Dx, G=G, truth=truth,
                data_clean=data_clean, lp=lp, deg=deg)


def lf_nmse(img, b):
    t = xp.fft.ifft2(b['lp'] * xp.fft.fft2(b['truth'])); i = xp.fft.ifft2(b['lp'] * xp.fft.fft2(img))
    s = xp.vdot(i, t) / (xp.vdot(i, i) + 1e-30)
    return float(xp.linalg.norm(s * i - t) / xp.linalg.norm(t))

def recon(b, sync, solver, data):
    Operators.Eigensolver = solver if solver else _power; eig_reset()
    img, *_ = Solvers.Alternating_projections_c(
        sync, xp.ones((b['Nx'], b['Nx']), xp.complex64), b['G'], b['probe'] + 0, b['tx'], b['ty'],
        overlap_cuda, split_cuda, data, False, maxiter, None, b['truth'], 1, sync_interval=1, num_iter=5)
    return img


print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
      "| frame-count side nnx", nnx, "| photons/frame", photons)
for nx in nx_list:
    b = build(nx)
    ov = (b['nx'] - b['Dx']) / b['nx']
    print(f"\n=== frame {nx}x{nx} ({b['nx']*b['nx']} px)  step={b['Dx']}  overlap~{ov:.0%}  "
          f"frames={b['nf']}  img={b['Nx']}  deg={b['deg']:.0f} ===")
    print(f"  {'phot/fr':>8}{'phot/px':>9} | {'nosync lf':>10}{'eigsh lf':>10}")
    for P in photons:
        acc_n, acc_e = [], []
        for sd in seeds:
            cp.random.seed(sd)
            scale = P * b['nf'] / float(b['data_clean'].sum())
            dn = (cp.random.poisson(cp.asarray(b['data_clean']) * scale).astype(cp.float32)) / scale
            acc_n.append(lf_nmse(recon(b, False, _power, dn), b))
            acc_e.append(lf_nmse(recon(b, True, eigsh_cupy, dn), b))
        print(f"  {P:>8.0f}{P/(nx*nx):>9.3f} | {np.mean(acc_n):>10.3f}{np.mean(acc_e):>10.3f}")

print("\nIf the photons/FRAME threshold rises ~propto pixels, the transferable unit is photons/PIXEL;")
print("if it's ~constant in photons/frame, big detectors tolerate far fewer photons/pixel.")
