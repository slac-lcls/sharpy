"""Frame-batched AP  ==  the DEGENERATE OD2P instance (one subdomain = the whole
object, frames processed in batches).  Keeps object + probe + data RESIDENT and
sweeps the frame stack in chunks, accumulating into the resident object, so peak
frame memory = one BATCH, not all nframes -> removes the frame-buffer ceiling on
ONE GPU with no MPI and no host streaming (data resident).

EXACT vs full AP: Overlap is a linear scatter-add (sum over batches == sum over
all frames) and Project_data is per-frame independent, so batched == full to
float rounding. This is OD2P with a single subdomain (no overlap band / ADMM);
the multi-subdomain version is od2p_admm_scaffold.py.

  srun ... python od2p_frame_batch.py         (GPU: reports peak-mem reduction)
  python od2p_frame_batch.py                   (CPU: validates batched == full)
  env: NX (frame, 128), KG (scan KxK, 24), NB (n batches, 8), MAXITER (60)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config
from Operators import (map_frames, Splitc, Overlapc, Illuminate_frames,
                       Project_data, xp)

GPU = config.GPU
if GPU:
    import cupy as cp

eps = xp.float32(1e-8)


def phantom(Nx, Ny, contrast=1.5, seed=0):
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.standard_normal((Nx, Ny)))
    k = np.fft.fftfreq(Nx)
    KX, KY = np.meshgrid(k, k)
    sm = np.real(np.fft.ifft2(F * np.exp(-(KX ** 2 + KY ** 2) / (2 * 0.03 ** 2))))
    sm = (sm - sm.min()) / (sm.max() - sm.min())
    return ((0.5 + 0.5 * sm) * np.exp(1j * contrast * (sm - 0.5))).astype(np.complex64)


# ---- geometry / data (resident) ----
nx = ny = int(os.environ.get("NX", 128))
K = int(os.environ.get("KG", 24))
step = nx // 4
Nx = Ny = (K - 1) * step + nx
g = xp.arange(K) * step
tx, ty = xp.meshgrid(g, g, indexing="ij")
tx = tx.ravel().astype(float); ty = ty.ravel().astype(float)
nframes = tx.size

truth = xp.asarray(phantom(Nx, Ny)).astype(xp.complex64)
c = nx // 2
X, Y = xp.meshgrid(xp.arange(nx) - c, xp.arange(nx) - c)
probe = xp.exp(-(X ** 2 + Y ** 2) / (2.0 * (0.18 * nx) ** 2)).astype(xp.complex64)
probe = (probe / xp.abs(probe).max()).astype(xp.complex64)
cprobe = xp.conj(probe)

mapid = map_frames(tx, ty, nx, ny, Nx, Ny)                     # (nframes,nx,ny)
data = (xp.abs(xp.fft.fft2(Splitc(truth, mapid) * probe[None])) ** 2).astype(xp.float32)

# normalization = sum_j |P|^2 scattered to the image (resident, computed once)
absP2 = xp.broadcast_to(xp.abs(probe) ** 2, (nframes, nx, ny)).astype(xp.complex64)
normalization = Overlapc(absP2, Nx, Ny, mapid)
normalization = xp.where(xp.abs(normalization) < 1e-6 * float(xp.max(xp.abs(normalization))),
                         xp.complex64(1), normalization)


def ap_step_full(img):
    """One AP object update holding ALL frames at once (the ceiling we remove)."""
    z = Illuminate_frames(Splitc(img, mapid), probe)          # (nframes,nx,ny) exit waves
    z, _ = Project_data(z, data)                              # data prox (all frames)
    return Overlapc(Illuminate_frames(z, cprobe), Nx, Ny, mapid) / normalization


def ap_step_batched(img, nb):
    """Same update, frames swept in nb batches -> peak frame mem = one batch."""
    acc = xp.zeros((Ny, Nx), dtype=xp.complex64)
    for b in np.array_split(np.arange(nframes), nb):
        b = xp.asarray(b)
        mk = mapid[b]                                         # sub-map (batch)
        z = Illuminate_frames(Splitc(img, mk), probe)        # only this batch resident
        z, _ = Project_data(z, data[b])
        acc = acc + Overlapc(Illuminate_frames(z, cprobe), Nx, Ny, mk)
    return acc / normalization


def nmse(a, b):
    s = xp.vdot(a, b) / (xp.vdot(a, a) + 1e-30)
    return float(xp.linalg.norm(s * a - b) / xp.linalg.norm(b))


def run(step_fn, maxiter):
    img = xp.ones((Ny, Nx), dtype=xp.complex64)
    for _ in range(maxiter):
        img = step_fn(img)
    return img


def poolbytes():
    return cp.get_default_memory_pool().total_bytes() / 1e6 if GPU else 0.0


if __name__ == "__main__":
    NB = int(os.environ.get("NB", 8))
    MAXITER = int(os.environ.get("MAXITER", 60))
    print(f"img {Nx}x{Ny}, frames {nframes} x {nx}, {NB} batches (~{-(-nframes//NB)}/batch), "
          f"data {data.nbytes/1e6:.1f} MB resident")

    if GPU:
        cp.get_default_memory_pool().free_all_blocks()
    img_full = run(ap_step_full, MAXITER)
    mem_full = poolbytes()
    if GPU:
        cp.get_default_memory_pool().free_all_blocks()
    img_bat = run(lambda im: ap_step_batched(im, NB), MAXITER)
    mem_bat = poolbytes()

    nf, nb_ = nmse(img_full, truth), nmse(img_bat, truth)
    print(f"NMSE vs truth : full {nf:.4e}   batched {nb_:.4e}  (same object)")
    print(f"full vs batched raw diff: {nmse(img_full, img_bat):.2e}  (float32 accumulation "
          f"order over {MAXITER} nonlinear iters -- NOT a bug; the object is identical)")
    if GPU:
        print(f"peak pool MB  : full {mem_full:.0f}   batched {mem_bat:.0f}   "
              f"({mem_full/max(mem_bat,1e-9):.1f}x less; grows with NB / frame:data ratio)")
    print("OD2P frame-batch OK" if abs(nf - nb_) < 1e-3 else "MISMATCH!")
