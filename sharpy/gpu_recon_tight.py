"""Step-2 prototype: a TIGHT (allocation-minimal, zero-host-sync, device-scalar)
GPU AP loop vs a NAIVE loop (same CUDA kernels, but per-iter allocations +
host-scalar alpha) vs the current Solvers.Alternating_projections.

Goal: measure what removing per-iteration allocations and host syncs buys --
the precursor to CUDA-graph capture. All three solve the same problem (sync off,
known probe) and must reach the same NMSE.

  srun -A lcls_g -C gpu -q interactive -N1 -n1 --gpus 1 -t 00:10:00 python gpu_recon_tight.py
"""
import os, sys, time, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
assert config.GPU, "run on Perlmutter with config.GPU=True"
import cupy as cp
import numpy as np
from Operators import (Project_data, map_frames, Splitc, Illuminate_frames,
                       Split_Overlap_plan, Propagate, xp)
from wrap_ops import split_cuda, overlap_cuda
from ap_accel import line_search as ls_host
import Solvers


def phantom(Nx, Ny, contrast=1.5, seed=0):
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.standard_normal((Nx, Ny)))
    k = np.fft.fftfreq(Nx); KX, KY = np.meshgrid(k, k)
    sm = np.real(np.fft.ifft2(F * np.exp(-(KX**2 + KY**2) / (2 * 0.03**2))))
    sm = (sm - sm.min()) / (sm.max() - sm.min())
    return ((0.5 + 0.5 * sm) * np.exp(1j * contrast * (sm - 0.5))).astype(np.complex64)

print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
MAXIT = int(os.environ.get("SHARPY_MAXITER", 250))
c64 = xp.complex64

# ---- geometry / data (dense raster, full coverage; env-configurable) ----
Nx = Ny = int(os.environ.get("BN", 256)); nx = ny = int(os.environ.get("BF", 64))
step = int(os.environ.get("BS", 16)); K = Nx // step
g = xp.arange(K) * step
tx, ty = xp.meshgrid(g, g, indexing="ij")
tx = tx.ravel().astype(xp.float32); ty = ty.ravel().astype(xp.float32)
nf = tx.size
translations = xp.ascontiguousarray((tx + 1j * ty).astype(c64))
mapid = map_frames(tx, ty, nx, ny, Nx, Ny)

truth = xp.asarray(phantom(Nx, Ny, 1.5))
cc = xp.arange(nx) - nx // 2
X, Y = xp.meshgrid(cc, cc)
probe = xp.asarray(xp.exp(-(X**2 + Y**2) / (2.0 * (0.18 * nx) ** 2))).astype(c64)
probe = (probe / xp.abs(probe).max()).astype(c64)

data = (xp.abs(xp.fft.fft2(Splitc(truth, mapid) * probe[None])) ** 2).astype(xp.float32)
frames_amp = xp.sqrt(data)
# image-domain normalization = sum_n |probe|^2 (via the overlap kernel, frames=0)
normalization = overlap_cuda(xp.zeros((Ny, Nx), c64), 0, translations, probe) + 1e-8
Split_pl, Overlap_pl = Split_Overlap_plan(tx, ty, nx, ny, Nx, Ny)


def nmse(img):
    s = xp.vdot(img, truth) / (xp.vdot(img, img) + 1e-30)   # global complex scale gauge
    return float(xp.linalg.norm(s * img - truth) / xp.linalg.norm(truth))


def ls_device(Z, D, a, n_newton=2):
    """Newton line search returning a 0-d DEVICE scalar (no float() -> no host sync)."""
    tiny = 1e-12
    Dabs2 = cp.abs(D) ** 2
    alpha = cp.asarray(1.0, dtype=cp.float64)
    for _ in range(n_newton):
        W = Z + alpha * D
        Wabs = cp.abs(W) + tiny
        r = Wabs - a
        u = cp.real(cp.conj(W) * D) / Wabs
        Jp = 2.0 * cp.sum(r * u, dtype=cp.float64)
        Jpp = 2.0 * cp.sum(u ** 2 + r * (Dabs2 - u ** 2) / Wabs, dtype=cp.float64)
        alpha = alpha - Jp / (Jpp + tiny)
    return alpha


def init_frames():
    f = xp.empty((nf, ny, nx), c64)
    img0 = xp.ones((Ny, Nx), c64)
    return split_cuda(img0, f, translations, probe)


def ap_naive(maxiter):
    frames = init_frames()
    for _ in range(maxiter):
        z0 = frames + 0.0                                   # alloc
        frames, _ = Project_data(frames, data)
        img = overlap_cuda(xp.zeros((Ny, Nx), c64), frames, translations, probe)  # alloc
        img = img / normalization                          # alloc
        fb = xp.empty((nf, ny, nx), c64)                   # alloc
        frames = split_cuda(img, fb, translations, probe)
        Dz = frames - z0                                   # alloc
        alpha = ls_host(Propagate(z0 + 0.0), Propagate(Dz + 0.0), frames_amp)  # float() -> HOST SYNC
        frames = z0 + alpha * Dz                           # alloc
    img = overlap_cuda(xp.zeros((Ny, Nx), c64), frames, translations, probe) / normalization
    return img


def ap_tight(maxiter):
    frames = init_frames()
    # preallocated scratch -- reused every iteration, no host syncs in the loop
    z0 = xp.empty_like(frames); Dz = xp.empty_like(frames)
    zf = xp.empty_like(frames); df = xp.empty_like(frames)
    apb = xp.empty_like(frames); pqf = xp.empty_like(frames)
    img = xp.empty((Ny, Nx), c64)
    for _ in range(maxiter):
        cp.copyto(z0, frames)
        frames, _ = Project_data(frames, data)             # ProxD still allocs 1 (pooled)
        img.fill(0)
        overlap_cuda(img, frames, translations, probe)     # img += conj(probe)*frames
        cp.divide(img, normalization, out=img)
        split_cuda(img, pqf, translations, probe)          # pqf = probe*crop(img)
        cp.subtract(pqf, z0, out=Dz)                       # Dz = P_Q - z0
        cp.copyto(zf, z0);  Propagate(zf)                  # in-place fft
        cp.copyto(df, Dz);  Propagate(df)
        alpha = ls_device(zf, df, frames_amp)              # 0-d DEVICE scalar, no sync
        cp.multiply(Dz, alpha, out=apb)
        cp.add(apb, z0, out=apb)                           # apb = z0 + alpha*Dz
        frames = apb                                       # next iter reuses pqf/apb in place
    img.fill(0); overlap_cuda(img, frames, translations, probe); cp.divide(img, normalization, out=img)
    return img


def bench(fn, maxiter, warm=10):
    fn(warm); cp.cuda.Stream.null.synchronize()
    t = time.time(); img = fn(maxiter); cp.cuda.Stream.null.synchronize()
    return img, 1e3 * (time.time() - t) / maxiter


# reference: current solver (stdout suppressed -- it print(ii)s every iter)
def ap_solver(maxiter):
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        img, *_ = Solvers.Alternating_projections(
            False, xp.ones((Nx, Ny), c64), None, probe + 0, Overlap_pl, Split_pl,
            data, False, maxiter, None, None, np.inf, line_search=True)
    return img

img_s = ap_solver(10); cp.cuda.Stream.null.synchronize()
t = time.time(); img_s = ap_solver(MAXIT); cp.cuda.Stream.null.synchronize()
t_s = 1e3 * (time.time() - t) / MAXIT
img_n, t_n = bench(ap_naive, MAXIT)
img_t, t_t = bench(ap_tight, MAXIT)

print("\n%d iters, %d frames %d^2, img %d^2" % (MAXIT, nf, nx, Nx))
print("solver (sparse path)   : NMSE %.3e  %6.2f ms/iter" % (nmse(img_s), t_s))
print("naive  kernels         : NMSE %.3e  %6.2f ms/iter" % (nmse(img_n), t_n))
print("tight  kernels (no sync): NMSE %.3e  %6.2f ms/iter  -> %.2fx vs naive, %.2fx vs solver"
      % (nmse(img_t), t_t, t_n / t_t, t_s / t_t))
print("DONE")
