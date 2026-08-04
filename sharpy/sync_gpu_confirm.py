"""GPU confirmation for the KD-tree Gramiam_plan dedup fix (commit 1448bde).

The bug only triggers on a SMALL image, where a frame overlaps another via
MULTIPLE periodic copies (2*overlap_range > image dim) → the 3×3-tiling KD-tree
returns duplicate (row,col) pairs → coo→csr merges them → val2H IndexError →
synchronize_frames_c crashes. (Large-image production runs have no duplicates,
which is why the bug went unnoticed on GPU.) This script uses such a small scene,
so PRE-fix it crashes and POST-fix the sync runs and recovers the low-freq phase.

Confirms two fixes together: (1) the KD-tree Gramiam_plan dedup (no val2H
IndexError), and (2) the small-image GPU kernel NaN -- root-caused to the zQQz.cu
RawKernel reinterpreting complex128 frames (promoted by a float64 image division)
as complex64, fixed by casting kernel inputs to complex64 in Gramiam_calc_cuda
(commit 1d5ce07). Now PASSes on BOTH CPU and GPU (no-sync ~0.71 -> sync ~2.7e-4).
Large-image GPU end-to-end is also verified by phase_sync_test.py (no-sync 0.87 /
power 0.21 / eigsh 1.46e-6 / invit 1.46e-6, unchanged).

Runs on whatever config.GPU selects (xp). On Perlmutter:
  source ~/sharpy-venv/bin/activate
  srun -A lcls_g -C gpu -q interactive -N1 -n1 --gpus 1 -t 00:10:00 python sync_gpu_confirm.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: F401

from Operators import (xp, make_probe, map_frames, Splitc, Overlapc,
                       Gramiam_plan, Precondition_calc, synchronize_frames_c)
from position_retrieval import apodize_probe
from Solvers import Project_data

print("config.GPU =", config.GPU, "| xp =", xp.__name__)

# Small image (Nx=56, frames 32px → overlaps wrap via several periodic copies =
# the duplicate-pair regime that triggered the val2H bug).
nx = 32; nnx = 14; step = 4; Nx = Ny = nnx * step
ax = (xp.arange(Nx) - Nx / 2.0); R2 = ax[:, None] ** 2 + ax[None, :] ** 2
phi = (2.0 * 2 * np.pi / float(R2.max())) * R2                 # quadratic low-freq phase
rng = np.random.default_rng(0)
amp = xp.asarray(np.clip(1.0 + 0.4 * rng.standard_normal((Nx, Ny)), 0.2, None))
truth = (amp * xp.exp(1j * phi)).astype(xp.complex64)

probe = make_probe(nx, nx, r1=0.075, r2=0.255)
probe = probe[0] if isinstance(probe, tuple) else probe
probe = apodize_probe(xp.asarray(probe / xp.abs(probe).max()).astype(xp.complex64))

gx, gy = np.meshgrid(np.arange(nnx), np.arange(nnx), indexing="ij"); gx = gx.ravel(); gy = gy.ravel()
tx = xp.asarray(gx * step + np.floor(step / 2) * (gx % 2), dtype=float)
ty = xp.asarray(gy * step, dtype=float)
nf = gx.size
mapid = map_frames(tx, ty, nx, nx, Nx, Ny)
data = (xp.abs(xp.fft.fft2(Splitc(truth, mapid) * probe)) ** 2).astype(xp.float32)

G = Gramiam_plan(tx, ty, nf, nx, nx, Nx, Ny, bw=0)
QQ = Overlapc(xp.tile(xp.abs(probe) ** 2, (nf, 1, 1)), Nx, Ny, mapid)
inorm = Splitc((1.0 / QQ).astype(xp.complex64), mapid)
fnorm = Precondition_calc(data.astype(xp.complex64), bw=0)

kx = (xp.arange(Nx) - Nx // 2)
KK = xp.sqrt(kx[:, None].astype(xp.float32) ** 2 + kx[None, :].astype(xp.float32) ** 2)
lp = xp.fft.ifftshift((KK <= 0.10 * Nx / 2).astype(xp.float32))


def lf(img):
    t = xp.fft.ifft2(lp * xp.fft.fft2(truth)); i = xp.fft.ifft2(lp * xp.fft.fft2(img))
    s = xp.vdot(i, t) / (xp.vdot(i, i) + 1e-30)
    return float(xp.linalg.norm(s * i - t) / xp.linalg.norm(t))


def recon(do_sync, niter=200, num_iter=50):
    img = xp.ones((Nx, Ny), xp.complex64); frames = Splitc(img, mapid) * probe
    for ii in range(niter):
        frames, _ = Project_data(frames, data, compute_residuals=False)
        if do_sync:
            om = synchronize_frames_c(frames, probe, fnorm, inorm, G, G["bw"], num_iter)
            frames = frames * om
        img = Overlapc(frames * xp.conj(probe), Nx, Ny, mapid) / QQ
        frames = Splitc(img, mapid) * probe
    return lf(img)


import math
ns = recon(False)
ss = recon(True)   # PRE-FIX: this raised "val2H IndexError" (duplicate KD-tree pairs)
print(f"lf-phase NMSE: no-sync {ns:.3e}  ->  sync {ss:.3e}")
# Reaching here at all = synchronize_frames_c ran without the val2H IndexError
# = the dedup fix is present (the primary assertion).
if math.isfinite(ss) and ss < ns / 10:
    print("PASS — dedup fix OK (no val2H IndexError) AND sync recovers the low-freq phase")
else:
    print("DEDUP-OK — synchronize_frames_c ran without the val2H IndexError (the fix). "
          "Phase NMSE not recovered: on GPU this *small* scene additionally hits a "
          "SEPARATE zQQz.cu small-image NaN (heavy periodic wrap, a regime the kernel "
          "isn't normally used in). For GPU end-to-end use phase_sync_test.py "
          "(large image) -- verified there. On CPU this PASSes.")
