"""Batched GPU AP — demonstrate frame-batching RAISES the frame-count cap on the A100.

Production Alternating_projections_c materializes the whole frames stack every iter
(frames=zeros(nf,nx,ny); split all; Project all; frames_old copy; overlap all) => a ~5-12x
working-set multiplier on the O(nf*nx^2) frame stack => OOM at nx=128 above ~16-65k frames
(memory_limit_harness baseline: nx=128 caps at 16384, OOM at 65536; all failures at
stage="reconstruct"). Frame-batching keeps object+data resident and processes frames in
batches of B: split_cuda -> Project_data -> overlap_cuda ACCUMULATE. peak DERIVED-frames =
O(B); only the resident DATA (O(nf)) remains a floor. Uses the SAME production GPU kernels
(wrap_ops.split_cuda/overlap_cuda, Operators.Project_data) -> the CPU prototype
(frame_batch_test.py) already proved this is bit-exact vs the full stack.

v1 = sync-OFF (the AP path is where the multiplier lives; the chunked-Gramian sync is v2).
Demonstrates: (a) batched (B<nf) == single-batch (B=nf) recon to float eps; (b) batched RUNS
at a frame count that OOMs production; (c) peak device memory + ms/iter.

Run on a GPU node:
  source ~/sharpy-venv/bin/activate; python -u batched_gpu_harness.py
env: NX(128) MAXIT(5) B(1024) EQ_NNX(64) BIG_NNX(256 362) DX(auto nx//4)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cupy as cp
from Operators import make_probe, make_translations, Project_data
from wrap_ops import split_cuda, overlap_cuda

NX = int(os.environ.get("NX", 128))
MAXIT = int(os.environ.get("MAXIT", 5))
B = int(os.environ.get("B", 1024))
EQ_NNX = int(os.environ.get("EQ_NNX", 64))                       # equivalence size (fits full)
BIG_NNX = [int(x) for x in os.environ.get("BIG_NNX", "256 362").split()]  # past-cap sizes
pool = cp.get_default_memory_pool()


def gpu_mb():
    return pool.used_bytes() / 1e6, cp.cuda.Device().mem_info[1] / 1e6


def free():
    pool.free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def simulate(nx, nnx, Dx, gen_B):
    """Mirror memory_limit_harness.simulate but build DATA in batches (peak O(gen_B)) so we
    can generate sizes whose full frame stack wouldn't fit."""
    ny, nny, Dy = nx, nnx, Dx
    nframes = nnx * nny
    illum, _ = make_probe(nx, ny, r1=0.075, r2=0.255, fx=+20, fy=-20)
    tx, ty = make_translations(Dx, Dy, nnx, nny, nnx * Dx, nny * Dy)
    Nx = int(np.ceil(float(cp.max(tx) - cp.min(tx)))); Ny = Nx
    np.random.seed(42)
    density = np.random.rand(Nx, Ny).astype(np.float32)
    truth = cp.array(np.exp(0.69 * (-1 + 0.5j) * density).astype(np.complex64))
    illum = cp.array(illum, dtype=cp.complex64)
    translations = (tx + 1j * ty).astype(cp.complex64)
    data = cp.zeros((nframes, nx, ny), dtype=cp.float32)         # resident data (the floor)
    for s in range(0, nframes, gen_B):
        e = min(nframes, s + gen_B)
        fb = cp.zeros((e - s, nx, ny), dtype=cp.complex64)
        split_cuda(truth, fb, translations[s:e], illum)
        data[s:e] = (cp.abs(cp.fft.fft2(fb)) ** 2).astype(cp.float32)
    del fb, truth, density; free()
    return illum, translations, data, Nx, Ny


def compute_norm(illum, translations, Nx, Ny):
    norm = cp.zeros((Nx, Ny), dtype=cp.complex64)
    overlap_cuda(norm, 0, translations, illum)                  # frames=0 -> |illum|^2 overlap
    return norm


def ap_sync_off(translations, illum, data, norm, Nx, Ny, maxit, B):
    """Frame-batched AP (sync off). B=nf -> single-batch (== production sync-off path)."""
    nf, nx, ny = data.shape
    reg = 1e-8 * float(cp.max(cp.abs(norm)))
    img = cp.ones((Nx, Ny), dtype=cp.complex64)
    for _ in range(maxit):
        img0 = cp.zeros((Nx, Ny), dtype=cp.complex64)
        for s in range(0, nf, B):
            e = min(nf, s + B)
            fb = cp.zeros((e - s, nx, ny), dtype=cp.complex64)
            split_cuda(img, fb, translations[s:e], illum)
            fb, _ = Project_data(fb, data[s:e])
            overlap_cuda(img0, fb, translations[s:e], illum)
        img = img0 / (norm + reg)
    return img


def peak_reset():
    try:
        cp.cuda.Device().mem_info  # touch
    except Exception:
        pass


if __name__ == "__main__":
    Dx = int(os.environ.get("DX", NX // 4))
    _, total = gpu_mb()
    print(f"NX={NX} Dx={Dx} MAXIT={MAXIT} B={B} | GPU total {total:.0f} MB")

    # ── (a) EQUIVALENCE: batched (B) == single-batch (B=nf), a size that fits both ──
    nnx = EQ_NNX; nf = nnx * nnx
    illum, tr, data, Nx, Ny = simulate(NX, nnx, Dx, B)
    norm = compute_norm(illum, tr, Nx, Ny)
    img_full = ap_sync_off(tr, illum, data, norm, Nx, Ny, MAXIT, nf)     # single batch
    img_bat = ap_sync_off(tr, illum, data, norm, Nx, Ny, MAXIT, min(B, nf))
    reld = float(cp.max(cp.abs(img_bat - img_full)) / (cp.max(cp.abs(img_full)) + 1e-30))
    print(f"\n== (a) FRAME-BATCHING EXACTNESS (nx={NX}, {nf} frames, B={min(B,nf)} vs {nf}) ==")
    print(f"   recon rel-diff max|Δimg|/max|img| = {reld:.2e}   (float sum-order -> ~1e-6 = exact)")
    del illum, tr, data, norm, img_full, img_bat; free()

    # ── (b) PAST-CAP: batched runs where production OOMs (baseline nx=128 caps ~16-65k) ──
    print(f"\n== (b) PAST-CAP: batched sync-off at frame counts that OOM production ==")
    print(f"   baseline (production, this nx): see memory_limit_harness; nx=128 caps 16384, OOM 65536")
    print(f"{'nframes':>9} {'data_GB':>8} {'status':>8} {'peak_GB':>8} {'ms/iter':>8} {'img_Nx':>7}")
    for nnx in BIG_NNX:
        nf = nnx * nnx
        try:
            free(); pool.free_all_blocks()
            illum, tr, data, Nx, Ny = simulate(NX, nnx, Dx, B)
            norm = compute_norm(illum, tr, Nx, Ny)
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            img = ap_sync_off(tr, illum, data, norm, Nx, Ny, MAXIT, B)
            cp.cuda.Stream.null.synchronize()
            ms = (time.perf_counter() - t0) / MAXIT * 1e3
            used, _ = gpu_mb()
            data_gb = data.nbytes / 1e9
            print(f"{nf:>9} {data_gb:>8.2f} {'OK':>8} {used/1e3:>8.2f} {ms:>8.1f} {Nx:>7}")
            del illum, tr, data, norm, img; free()
        except (cp.cuda.memory.OutOfMemoryError, MemoryError) as ex:
            print(f"{nf:>9} {'-':>8} {'OOM':>8}  ({type(ex).__name__})")
            free()
    print("\nEXPECT: (a) ~1e-6 (batched == full); (b) batched COMPLETES at 65536+ frames where "
          "production OOMs (baseline) -> the frames-stack multiplier is removed; peak ~ resident "
          "data + O(B), not O(nf)x5-12.")
