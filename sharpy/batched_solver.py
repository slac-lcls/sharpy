"""Validation harness for Solvers.Alternating_projections_batched_c (the frame-batched GPU
drop-in, now folded INTO Solvers.py).

Checks, on a GPU node, that the batched solver reproduces the production
Solvers.Alternating_projections_c reconstruction and runs at frame counts that OOM the
full stack:
  (1) batched (single-batch B=nf, and batched B<nf) recon == production recon (~1e-5);
  (2) batched runs at 65536 frames (production OOMs) at ~production speed.

Run on a GPU node:  source ~/sharpy-venv/bin/activate; python -u batched_solver.py
env: NX(128) MAXIT(5) NUMITER(10) B(1024) EQ_NNX(64) BIG_NNX(256)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")                                            # headless: no plt.show() hang
import cupy as cp
import Solvers
from Solvers import Alternating_projections_batched_c
from Operators import make_probe, make_translations, Gramiam_plan
from wrap_ops import split_cuda, overlap_cuda

NX = int(os.environ.get("NX", 128)); Dx = NX // 4
MAXIT = int(os.environ.get("MAXIT", 5)); NUMITER = int(os.environ.get("NUMITER", 10))
B = int(os.environ.get("B", 1024))
EQ_NNX = int(os.environ.get("EQ_NNX", 64))
BIG_NNX = [int(x) for x in os.environ.get("BIG_NNX", "256").split()]
pool = cp.get_default_memory_pool()


def simulate(nx, nnx, Dx, gen_B):
    nframes = nnx * nnx
    illum, _ = make_probe(nx, nx, r1=0.075, r2=0.255, fx=+20, fy=-20)
    tx, ty = make_translations(Dx, Dx, nnx, nnx, nnx * Dx, nnx * Dx)
    Nx = int(np.ceil(float(cp.max(tx) - cp.min(tx))))
    np.random.seed(42)
    truth = cp.array(np.exp(0.69 * (-1 + 0.5j) * np.random.rand(Nx, Nx).astype(np.float32)).astype(np.complex64))
    illum = cp.array(illum, dtype=cp.complex64)
    data = cp.zeros((nframes, nx, nx), dtype=cp.float32)
    tr = (tx + 1j * ty).astype(cp.complex64)
    for s in range(0, nframes, gen_B):
        e = min(nframes, s + gen_B)
        fb = cp.zeros((e - s, nx, nx), dtype=cp.complex64)
        split_cuda(truth, fb, tr[s:e], illum)
        data[s:e] = (cp.abs(cp.fft.fft2(fb)) ** 2).astype(cp.float32)
    del fb; pool.free_all_blocks()
    return truth, illum, tx, ty, data, Nx


if __name__ == "__main__":
    # (1) faithfulness: batched (single + batched) vs production Alternating_projections_c
    nnx = EQ_NNX; nf = nnx * nnx
    truth, illum, tx, ty, data, Nx = simulate(NX, nnx, Dx, B)
    Gplan = Gramiam_plan(tx, ty, nf, NX, NX, Nx, Nx, bw=0)
    imgP, _, _, resP = Solvers.Alternating_projections_c(
        True, cp.ones((Nx, Nx), cp.complex64), Gplan, illum, tx, ty, overlap_cuda, split_cuda,
        data, False, MAXIT, normalization=None, img_truth=truth, residuals_interval=MAXIT,
        sync_interval=1, num_iter=NUMITER)
    imgS, _, _, _ = Alternating_projections_batched_c(
        True, cp.ones((Nx, Nx), cp.complex64), Gplan, illum, tx, ty, overlap_cuda, split_cuda,
        data, False, MAXIT, None, truth, MAXIT, 1, NUMITER, frame_batch=nf)
    imgB, _, _, resB = Alternating_projections_batched_c(
        True, cp.ones((Nx, Nx), cp.complex64), Gplan, illum, tx, ty, overlap_cuda, split_cuda,
        data, False, MAXIT, None, truth, MAXIT, 1, NUMITER, frame_batch=B)
    rS = float(cp.max(cp.abs(imgS - imgP)) / (cp.max(cp.abs(imgP)) + 1e-30))
    rB = float(cp.max(cp.abs(imgB - imgP)) / (cp.max(cp.abs(imgP)) + 1e-30))
    print(f"\n== BATCHED SOLVER vs PRODUCTION Alternating_projections_c (nx={NX}, {nf} frames) ==")
    print(f"   batched single-batch vs production = {rS:.2e} | batched B={B} vs production = {rB:.2e}")
    print(f"   truth-NMSE  production {float(resP[-1,0]):.4f}  batched {float(resB[-1,0]):.4f}")
    del truth, illum, data, Gplan, imgP, imgS, imgB; pool.free_all_blocks()

    # (2) past-cap: batched runs where production OOMs
    print(f"\n== PAST-CAP: Alternating_projections_batched_c (production nx=128 caps 16384) ==")
    print(f"{'nframes':>9} {'data_GB':>8} {'status':>7} {'ms/iter':>9} {'truthNMSE':>10}")
    for nnx in BIG_NNX:
        nf = nnx * nnx
        try:
            pool.free_all_blocks()
            truth, illum, tx, ty, data, Nx = simulate(NX, nnx, Dx, B)
            Gplan = Gramiam_plan(tx, ty, nf, NX, NX, Nx, Nx, bw=0)
            cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
            img, _, _, res = Alternating_projections_batched_c(
                True, cp.ones((Nx, Nx), cp.complex64), Gplan, illum, tx, ty, overlap_cuda,
                split_cuda, data, False, MAXIT, None, truth, MAXIT, 1, NUMITER, frame_batch=B)
            cp.cuda.Stream.null.synchronize()
            ms = (time.perf_counter() - t0) / MAXIT * 1e3
            print(f"{nf:>9} {data.nbytes/1e9:>8.2f} {'OK':>7} {ms:>9.1f} {float(res[-1,0]):>10.4f}")
            del truth, illum, data, Gplan, img; pool.free_all_blocks()
        except (cp.cuda.memory.OutOfMemoryError, MemoryError) as ex:
            print(f"{nf:>9} {'-':>8} {'OOM':>7}  ({type(ex).__name__})"); pool.free_all_blocks()
    print("\nEXPECT: batched == production ~1e-6 (recon); runs past production's OOM cap at ~production speed.")
