#!/usr/bin/env python3
"""Phase B: noise-feasibility ENVELOPE for synchronization. Low-freq-band NMSE of the
recovered long-range phase vs photons/frame, swept over OVERLAP (probe size r2) and
SCALE (frame count nnx). Real AP+sync recon, physical Poisson noise, eigsh-sync vs
no-sync, seed-averaged. Saves sync_noise_envelope.npz for plotting (the report figure).

  srun -A lcls_g -C gpu -q interactive -N1 -n1 --gpus 1 -t 00:30:00 python sync_noise_figure.py
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

nx = ny = 16; Dx = 5; CYC = 1.0; maxiter = 300; LOWPASS = 0.10
photons = [3000., 1000., 300., 100., 30., 10.]
seeds = [0, 1]
_power = Operators.Eigensolver


def gauge(w, n):
    w = w.ravel() / (cp.abs(w.ravel()) + 1e-30); s = cp.sum(w)
    return cp.reshape(w * cp.conj(s) / (cp.abs(s) + 1e-30), (n, 1, 1))

def eigsh_cupy(H, num_iter, v0=None, tol=1e-6):
    lam, V = eigsh(H, k=1, ncv=3, maxiter=20, which="LM"); return gauge(V[:, int(cp.argmax(lam))], H.shape[0])


def build(nnx, r2):
    probe = make_probe(nx, ny, r1=0.025 * 3, r2=r2, fx=0., fy=0.)
    probe = xp.asarray(probe[0] if isinstance(probe, tuple) else probe).astype(xp.complex64)
    tx, ty = make_translations(Dx, Dx, nnx, nnx, nnx * Dx, nnx * Dx)
    tx = xp.asarray(tx).astype(xp.int64); ty = xp.asarray(ty).astype(xp.int64)
    Nx = int(xp.ceil(xp.max(tx) - xp.min(tx))); nf = tx.size
    tr = (tx + 1j * ty).astype(xp.complex64)
    G = Gramiam_plan(tx, ty, nf, nx, ny, Nx, Nx, bw=0)
    deg = float(G['col'].size) * 2.0 / nf                  # mean overlap-graph degree
    xx = (xp.arange(Nx) - Nx / 2).astype(xp.float32); R2 = xx[:, None] ** 2 + xx[None, :] ** 2
    truth = xp.exp(1j * (CYC * 2 * np.pi / float(R2.max())) * R2).astype(xp.complex64)
    frd = xp.zeros((nf, nx, ny), xp.complex64); split_cuda(truth, frd, tr, probe)
    data_clean = (xp.abs(xp.fft.fft2(frd)) ** 2).astype(xp.float32)
    kx = (xp.arange(Nx) - Nx // 2)
    KK = xp.sqrt(kx[:, None].astype(xp.float32) ** 2 + kx[None, :].astype(xp.float32) ** 2)
    lp = xp.fft.ifftshift((KK <= LOWPASS * Nx / 2).astype(xp.float32))
    return dict(probe=probe, tx=tx, ty=ty, Nx=Nx, nf=nf, G=G, truth=truth,
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


def sweep(nnx, r2):
    b = build(nnx, r2)
    res = {m: np.zeros(len(photons)) for m in ("nosync", "eigsh")}
    for pi, P in enumerate(photons):
        acc = {"nosync": [], "eigsh": []}
        for sd in seeds:
            cp.random.seed(sd)
            scale = P * b['nf'] / float(b['data_clean'].sum())
            dn = (cp.random.poisson(cp.asarray(b['data_clean']) * scale).astype(cp.float32)) / scale
            acc["nosync"].append(lf_nmse(recon(b, False, _power, dn), b))
            acc["eigsh"].append(lf_nmse(recon(b, True, eigsh_cupy, dn), b))
        for m in acc:
            res[m][pi] = float(np.mean(acc[m]))
    return res, b['deg'], b['nf']


print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), "| photons", photons, "seeds", seeds)
out = {"photons": np.array(photons)}
print("=== Sweep A: OVERLAP (r2 at nnx=64, 4096 frames) ===")
for r2 in (0.40, 0.30, 0.22):
    res, deg, nf = sweep(64, r2)
    out[f"A_r2_{r2}_nosync"] = res["nosync"]; out[f"A_r2_{r2}_eigsh"] = res["eigsh"]; out[f"A_r2_{r2}_deg"] = deg
    print(f"  r2={r2} deg={deg:.0f} nf={nf}  eigsh-lf={np.round(res['eigsh'],3)}  nosync-lf={np.round(res['nosync'],3)}")
print("=== Sweep B: SCALE (nnx at r2=0.30, fixed photons/frame) ===")
for nnx in (32, 64, 96):
    res, deg, nf = sweep(nnx, 0.30)
    out[f"B_n_{nnx}_nosync"] = res["nosync"]; out[f"B_n_{nnx}_eigsh"] = res["eigsh"]
    out[f"B_n_{nnx}_nf"] = nf; out[f"B_n_{nnx}_deg"] = deg
    print(f"  nnx={nnx} nf={nf} deg={deg:.0f}  eigsh-lf={np.round(res['eigsh'],3)}")

np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sync_noise_envelope.npz"), **out)
print("saved sync_noise_envelope.npz")
