"""
Where does the Gramian phase-synchronization help, and how should it be combined
with data fitting? A CPU demonstration (the existing sync tests are GPU-only).

Setup: a strong LOW-SPATIAL-FREQUENCY PHASE object (quadratic phase + high-freq
amplitude texture), grid scan, intensity data. The AP loop alternates a data-fit
step (Fourier-magnitude P_F, optionally over-relaxed Fienup-style) with the
overlap consensus, and -- on a cadence -- a Gramian re-sync (top eigenvector of
the frame Gramian = per-frame relative phase, Eq.18-19 of arXiv:1209.4924).
Metric: low-frequency-band phase NMSE (the mode AP cannot propagate locally).

Findings (see also [[gramian-sync-eigsolver]], [[line-search-ap-acceleration]]):
  1. Sync is ESSENTIAL for the long-range/low-freq phase: no-sync plateaus at
     ~0.7-0.9, with sync -> 1e-7.
  2. It is a SLOW-MODE fixer -> MULTI-RATE works: syncing every ~5-20 iters is
     ~as good as every iter (the expensive global eigen-solve is paid rarely
     while cheap data+overlap run every iteration).
  3. Data fitting ALONE never recovers the low-freq phase (hard or over-relaxed);
     the Gramian supplies the global phase the gradient/projection cannot find.
  4. Acceleration (Fienup over-relaxation) is COMPLEMENTARY, not a substitute:
     with sparse sync it converges much further at a tight iteration budget, but
     fails without sync.

Note: uses a direct scipy eigsh for the Gramian top eigenvector because the
production Operators.Eigensolver currently has no working CPU branch (its power
iteration is nested under `if GPU:`); the Gramiam_plan/Gramiam_calc CPU path is fine.

Run:  python sync_hybrid_cadence_test.py
"""

import os
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: F401  (CPU/GPU switch; this demo is CPU)

from Operators import (make_probe, map_frames, Splitc, Overlapc, Illuminate_frames,
                       Gramiam_plan, Gramiam_calc, Precondition_calc)
from position_retrieval import apodize_probe
from Solvers import Project_data

# ---- scene with strong LOW-FREQUENCY PHASE (where sync is essential) ----
nx = 32; nnx = 14; step = 4; Nx = Ny = nnx * step          # 56, periodic tiling
xx = (np.arange(Nx) - Nx / 2.0); R2 = xx[:, None] ** 2 + xx[None, :] ** 2
CYC = 2.0
phi = (CYC * 2 * np.pi / R2.max()) * R2                     # quadratic low-freq phase
rng = np.random.default_rng(0)
amp = np.clip(1.0 + 0.4 * rng.standard_normal((Nx, Ny)), 0.2, None)  # high-freq texture
truth = (amp * np.exp(1j * phi)).astype(np.complex64)

probe = make_probe(nx, nx, r1=0.075, r2=0.255)
probe = probe[0] if isinstance(probe, tuple) else probe
probe = apodize_probe(np.asarray(probe / np.abs(probe).max(), np.complex64))

gx, gy = np.meshgrid(np.arange(nnx), np.arange(nnx), indexing="ij"); gx = gx.ravel(); gy = gy.ravel()
tx = (gx * step + np.floor(step / 2) * (gx % 2)).astype(float)
ty = (gy * step).astype(float)
nf = gx.size
mapid = map_frames(tx, ty, nx, nx, Nx, Ny)
data = (np.abs(np.fft.fft2(Splitc(truth, mapid) * probe)) ** 2).astype(np.float64)

# ---- sync setup (CPU) ----
Gramiam = Gramiam_plan(tx, ty, nf, nx, nx, Nx, Ny, bw=0)
QQ = Overlapc(np.tile(np.abs(probe) ** 2, (nf, 1, 1)), Nx, Ny, mapid)
inorm_split = Splitc((1.0 / QQ).astype(np.complex64), mapid)
frames_norm = Precondition_calc(data.astype(np.complex64), bw=0)

# ---- low-frequency band mask + phase-aligned NMSE ----
kx = (np.arange(Nx) - Nx // 2)
KK = np.sqrt(kx[:, None].astype(float) ** 2 + kx[None, :].astype(float) ** 2)
lp = np.fft.ifftshift((KK <= 0.10 * Nx / 2).astype(float))


def lf_nmse(img):
    t = np.fft.ifft2(lp * np.fft.fft2(truth)); i = np.fft.ifft2(lp * np.fft.fft2(img))
    s = np.vdot(i, t) / (np.vdot(i, i) + 1e-30)
    return float(np.linalg.norm(s * i - t) / np.linalg.norm(t))


def sync_omega(frames):
    """Top eigenvector of the Gramian = per-frame relative-phase sync (Eq.18-19)."""
    framesl = Illuminate_frames(frames, np.conj(probe))     # conj(P).z
    framesr = framesl * inorm_split                          # weighted by 1/(Q*Q)
    H = Gramiam_calc(framesl, framesr, Gramiam, frames_norm)
    H = (H + H.getH()) * 0.5
    _, vecs = eigsh(H, k=1, which="LM")
    w = vecs[:, 0]
    return (w / (np.abs(w) + 1e-30)).astype(np.complex64)


def recon(sync_k, alpha=1.0, maxiter=200):
    """alpha=1 -> hard P_F (AP); alpha>1 -> over-relaxed Fienup-style data step."""
    img = np.ones((Nx, Ny), np.complex64); frames = Splitc(img, mapid) * probe
    for ii in range(maxiter):
        pf, _ = Project_data(frames, data, compute_residuals=False)       # data fit (P_F)
        frames = frames + alpha * (pf - frames)
        if sync_k and (ii % sync_k == 0):
            frames = frames * sync_omega(frames)[:, None, None]           # Gramian re-sync
        img = Overlapc(frames * np.conj(probe), Nx, Ny, mapid) / QQ        # overlap consensus
        frames = Splitc(img, mapid) * probe
    return lf_nmse(img)


if __name__ == "__main__":
    print(f"low-freq-phase scene: {nf} frames, image {Nx}, CYC={CYC} (quadratic phase)\n")
    print("Part 1 -- where sync helps + cadence (AP hard P_F + overlap, maxiter=200), lf-phase NMSE:")
    for k in (0, 20, 5, 1):
        tag = "no sync" if k == 0 else f"sync every {k}"
        print(f"  {tag:>14}: {recon(k, 1.0):.3e}")

    print("\nPart 2 -- data step x sync at a TIGHT budget (maxiter=60), lf-phase NMSE:")
    print(f"  {'data step':>27} | {'no sync':>10} | {'sync every 5':>12}")
    for name, a in (("AP hard P_F (a=1.0)", 1.0), ("Fienup over-relaxed (a=1.8)", 1.8)):
        print(f"  {name:>27} | {recon(0, a, 60):>10.3e} | {recon(5, a, 60):>12.3e}")
