"""Redundancy vs (photon noise) vs (uncorrected jitter).

"Compensate jitter with redundancy: many shots, noisy, lots of overlap." Splits
into two effects with opposite responses to redundancy:
  - photon noise = VARIANCE  -> redundancy (more shots/overlap = more total dose)
    beats it down: floor ~ 1/sqrt(total photons).
  - uncorrected position jitter = BIAS (mis-registration) -> averaging blurs by the
    jitter PSF; redundancy does NOT remove it. Needs position refinement (which
    redundancy makes well-posed -- separate lever).

Sweep overlap/redundancy (denser scan = smaller step = more frames/pixel) at fixed
low dose/shot; reconstruct with NOMINAL positions (jitter uncorrected). Compare
jitter=0 (noise only) vs jitter>0.

  /opt/anaconda3/bin/python3 od2p_redundancy_test.py
  env: DOSE(5 ph/px/shot) JIT(1.5 px) ITERS(150)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config
from Operators import map_frames, Splitc, Overlapc, Illuminate_frames, Project_data, xp

Nx = Ny = 192
nx = ny = 48
c = nx // 2
Xp, Yp = xp.meshgrid(xp.arange(nx) - c, xp.arange(nx) - c)
probe = xp.exp(-(Xp ** 2 + Yp ** 2) / (2.0 * (0.18 * nx) ** 2)).astype(xp.complex64)
probe = (probe / xp.abs(probe).max()).astype(xp.complex64)
cprobe = xp.conj(probe)


def phantom(seed=0):
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.standard_normal((Nx, Ny)))
    k = np.fft.fftfreq(Nx); KX, KY = np.meshgrid(k, k)
    sm = np.real(np.fft.ifft2(F * np.exp(-(KX ** 2 + KY ** 2) / (2 * 0.03 ** 2))))
    sm = (sm - sm.min()) / (sm.max() - sm.min())
    return ((0.5 + 0.5 * sm) * np.exp(1j * 1.5 * (sm - 0.5))).astype(np.complex64)


truth = xp.asarray(phantom()).astype(xp.complex64)


def build(step, jit, seed):
    rng = np.random.default_rng(seed)
    g = np.arange(0, Nx - nx + 1, step, dtype=float)
    tx, ty = np.meshgrid(g, g); tx = tx.ravel(); ty = ty.ravel()
    nf = tx.size
    jx = np.clip(np.round(tx + rng.normal(0, jit, nf)), 0, Nx - nx)
    jy = np.clip(np.round(ty + rng.normal(0, jit, nf)), 0, Nx - nx)
    mnom = map_frames(xp.asarray(tx), xp.asarray(ty), nx, ny, Nx, Ny)
    mjit = map_frames(xp.asarray(jx), xp.asarray(jy), nx, ny, Nx, Ny)
    return mnom, mjit, nf, (nx / step) ** 2                     # ~frames per pixel


def make_data(mjit, nf, dose, seed):
    lam = xp.abs(xp.fft.fft2(Splitc(truth, mjit) * probe[None])) ** 2
    lam = lam / float(lam.mean()) * dose
    xp.random.seed(seed)
    return xp.random.poisson(lam).astype(xp.float32)


def recon(mnom, nf, data, iters):
    absP2 = xp.broadcast_to(xp.abs(probe) ** 2, (nf, nx, ny)).astype(xp.complex64)
    norm = Overlapc(absP2, Nx, Ny, mnom)
    norm = xp.where(xp.abs(norm) < 1e-6 * float(xp.max(xp.abs(norm))), xp.complex64(1), norm)
    u = xp.ones((Ny, Nx), dtype=xp.complex64)
    for _ in range(iters):
        z = Illuminate_frames(Splitc(u, mnom), probe)
        z, _ = Project_data(z, data)
        u = Overlapc(Illuminate_frames(z, cprobe), Nx, Ny, mnom) / norm
    return u


def nmse(a, b):
    s = xp.vdot(a, b) / (xp.vdot(a, a) + 1e-30)
    return float(xp.linalg.norm(s * a - b) / xp.linalg.norm(b))


if __name__ == "__main__":
    DOSE = float(os.environ.get("DOSE", 5))
    JIT = float(os.environ.get("JIT", 4.0))
    ITERS = int(os.environ.get("ITERS", 250))
    NSEED = int(os.environ.get("NSEED", 5))
    print(f"FOV {Nx}x{Ny}, frame {nx}, dose {DOSE:g} ph/px/shot, jitter {JIT:g} px (UNCORRECTED), "
          f"median of {NSEED} seeds")
    print(f"{'step':>5} {'frames':>7} {'redund':>7} {'tot ph/px':>10} "
          f"{'NMSE noise-only':>16} {'NMSE +jitter':>13}")
    for step in [nx // 2, nx // 3, nx // 4, nx // 6, nx // 8]:
        n0s, njs = [], []
        for sd in range(NSEED):
            mnom0, mjit0, nf0, redun = build(step, 0.0, seed=sd)          # jitter=0
            mnom, mjit, nf, _ = build(step, JIT, seed=sd)                 # jittered positions
            n0s.append(nmse(recon(mnom0, nf0, make_data(mjit0, nf0, DOSE, 100 + sd), ITERS), truth))
            njs.append(nmse(recon(mnom, nf, make_data(mjit, nf, DOSE, 100 + sd), ITERS), truth))
        print(f"{step:>5} {nf:>7} {redun:>7.1f} {DOSE*redun:>10.0f} "
              f"{np.median(n0s):>16.4f} {np.median(njs):>13.4f}")
    print("\nnoise-only: NMSE drops with redundancy (variance averaged out).")
    print("+jitter: plateaus above -- uncorrected jitter is BIAS/blur, redundancy can't")
    print("remove it; it needs position refinement (which the redundancy makes well-posed).")
