"""Benchmark: CUDA kernels (src/split.cu, src/overlap.cu via wrap_ops) vs the
portable path (Splitc gather + Illuminate / Overlapc bincount / Overlapd sparse)
on GPU. Times the SPLIT (image->frames, fused illumination) and OVERLAP
(frames->image, fused conj-illumination) operators at a few sizes, after a
correctness check that they agree.

GPU only:
  srun -A lcls_g -C gpu -q interactive -N1 -n1 --gpus 1 -t 00:10:00 python bench_ops.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
assert config.GPU, "run with config.GPU=True (Perlmutter)"
import cupy as cp
import numpy as np
from Operators import (map_frames, Splitc, Overlapc, Overlapd, Illuminate_frames,
                       Split_Overlap_plan, xp)
from wrap_ops import split_cuda, overlap_cuda

print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())


def bench(fn, reps=50, warmup=5):
    for _ in range(warmup):
        fn()
    cp.cuda.Stream.null.synchronize()
    import time
    t = time.time()
    for _ in range(reps):
        fn()
    cp.cuda.Stream.null.synchronize()
    return 1e3 * (time.time() - t) / reps   # ms/call


def run(Nx, nx, step):
    Ny, ny = Nx, nx
    K = Nx // step
    g = xp.arange(K) * step
    tx, ty = xp.meshgrid(g, g, indexing="ij")
    tx = tx.ravel().astype(xp.float32); ty = ty.ravel().astype(xp.float32)
    nf = tx.size
    translations = (tx + 1j * ty).astype(xp.complex64)           # float2 (.x=x,.y=y)
    mapid = map_frames(tx, ty, nx, ny, Nx, Ny)
    img = (xp.random.randn(Ny, Nx) + 1j * xp.random.randn(Ny, Nx)).astype(xp.complex64)
    illum = (xp.random.randn(ny, nx) + 1j * xp.random.randn(ny, nx)).astype(xp.complex64)
    frames = (xp.random.randn(nf, ny, nx) + 1j * xp.random.randn(nf, ny, nx)).astype(xp.complex64)
    Split, _ = Split_Overlap_plan(tx, ty, nx, ny, Nx, Ny)  # builds CSR SS (timed below)
    import time
    t0 = time.time(); Splp, _ = Split_Overlap_plan(tx, ty, nx, ny, Nx, Ny)
    cp.cuda.Stream.null.synchronize(); plan_ms = 1e3 * (time.time() - t0)
    # SS for Overlapd
    col = xp.arange(mapid.size); val = xp.ones(mapid.size, dtype=xp.float32)
    import cupyx.scipy.sparse as sp
    SS = sp.csr_matrix(sp.coo_matrix((val, (mapid.ravel(), col)), shape=(Nx * Ny, mapid.size)))

    fb = xp.empty_like(frames)
    data_mb = frames.nbytes / 1e6
    print("\n--- img %d^2, frame %d^2, %d frames (%.0f MB frames) ---"
          % (Nx, nx, nf, data_mb))

    # ---- SPLIT: image -> frames (fused illumination) ----
    s_cuda = split_cuda(img, fb, translations, illum).copy()
    s_port = Illuminate_frames(Splitc(img, mapid), illum)
    e = float(xp.linalg.norm(s_cuda - s_port) / xp.linalg.norm(s_port))
    t_cuda = bench(lambda: split_cuda(img, fb, translations, illum))
    t_port = bench(lambda: Illuminate_frames(Splitc(img, mapid), illum))
    print("SPLIT    cuda %.3f ms | gather+illum %.3f ms | speedup %.2fx | match %.1e"
          % (t_cuda, t_port, t_port / t_cuda, e))

    # ---- OVERLAP: frames -> image (fused conj-illumination) ----
    o_cuda = overlap_cuda(xp.zeros((Ny, Nx), xp.complex64), frames, translations, illum).copy()
    ill_fr = Illuminate_frames(frames, xp.conj(illum))
    o_binc = Overlapc(ill_fr, Nx, Ny, mapid)
    o_sprs = Overlapd(ill_fr, SS, (Ny, Nx))
    eb = float(xp.linalg.norm(o_cuda - o_binc) / xp.linalg.norm(o_binc))
    es = float(xp.linalg.norm(o_cuda - o_sprs) / xp.linalg.norm(o_sprs))
    tc = bench(lambda: overlap_cuda(xp.zeros((Ny, Nx), xp.complex64), frames, translations, illum))
    tb = bench(lambda: Overlapc(Illuminate_frames(frames, xp.conj(illum)), Nx, Ny, mapid))
    ts = bench(lambda: Overlapd(Illuminate_frames(frames, xp.conj(illum)), SS, (Ny, Nx)))
    print("OVERLAP  cuda %.3f ms | bincount %.3f ms (%.2fx, match %.1e) | sparse %.3f ms (%.2fx, match %.1e) [+plan %.1f ms]"
          % (tc, tb, tb / tc, eb, ts, ts / tc, es, plan_ms))


for cfg in [(256, 64, 16), (512, 128, 32), (512, 128, 16)]:
    run(*cfg)
print("\nDONE")
