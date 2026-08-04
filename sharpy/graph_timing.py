"""Captured-graph timing of the batched (sync-off, device) AP core -- the honest per-iter
throughput number the memory calls for ("measure captured-graph not eager",
droplet-gpu-optimization-backlog / gpu-memory-scaling). The batched loop issues many small
kernel launches (per batch: split_cuda, Project_data FFTs, overlap_cuda) + Python overhead;
a CUDA graph replays the whole iteration with ~one launch, exposing the real GPU cost.

Three measurements per size:
  EAGER  = Python loop, synchronize each iter (== what batched_solver.py reports)
  EVENT  = N iters back-to-back on the stream, timed by CUDA events (no per-iter host sync)
  GRAPH  = capture ONE iteration into a cudaGraph, replay N times (launch overhead removed)
The iteration writes into PRE-ALLOCATED buffers (split -> F slice; overlap -> img0; img in
place) so capture sees fixed addresses; the async mempool makes the FFT allocations
capturable. sync is OFF (the eigensolver's host reductions are not graph-capturable).

Run on a GPU node: source ~/sharpy-venv/bin/activate; python -u graph_timing.py
env: NX(128) B(1024) N(20) NNXLIST("64 128")  # nframes = nnx^2
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cupy as cp
from Operators import make_probe, make_translations, Project_data
from wrap_ops import split_cuda, overlap_cuda

NX = int(os.environ.get("NX", 128)); Dx = NX // 4
B = int(os.environ.get("B", 1024)); N = int(os.environ.get("N", 20))
NNXLIST = [int(x) for x in os.environ.get("NNXLIST", "64 128").split()]


def simulate(nx, nnx, Dx):
    nframes = nnx * nnx
    illum, _ = make_probe(nx, nx, r1=0.075, r2=0.255, fx=+20, fy=-20)
    tx, ty = make_translations(Dx, Dx, nnx, nnx, nnx * Dx, nnx * Dx)
    Nx = int(np.ceil(float(cp.max(tx) - cp.min(tx))))
    np.random.seed(42)
    truth = cp.array(np.exp(0.69 * (-1 + 0.5j) * np.random.rand(Nx, Nx).astype(np.float32)).astype(np.complex64))
    illum = cp.array(illum, dtype=cp.complex64)
    tr = (tx + 1j * ty).astype(cp.complex64)
    data = cp.zeros((nframes, nx, nx), dtype=cp.float32)
    for s in range(0, nframes, B):
        e = min(nframes, s + B)
        fb = cp.zeros((e - s, nx, nx), dtype=cp.complex64)
        split_cuda(truth, fb, tr[s:e], illum)
        data[s:e] = (cp.abs(cp.fft.fft2(fb)) ** 2).astype(cp.float32)
    norm = cp.zeros((Nx, Nx), dtype=cp.complex64)
    overlap_cuda(norm, 0, tr, illum)
    reg = 1e-8 * float(cp.max(cp.abs(norm)))
    norm_reg_inv = (1.0 / (norm + reg)).astype(cp.complex64)
    return illum, tr, data, Nx, norm_reg_inv


def make_iter(img, img0, F, tr, illum, data, norm_reg_inv, B):
    nframes, nx, ny = data.shape

    def one():
        img0.fill(0)
        for s in range(0, nframes, B):
            e = min(nframes, s + B)
            fb = F[s:e]                                     # split writes into the F view (no alloc)
            split_cuda(img, fb, tr[s:e], illum)
            fbp, _ = Project_data(fb, data[s:e], compute_residuals=False)
            overlap_cuda(img0, fbp, tr[s:e], illum)
        img[...] = img0 * norm_reg_inv                      # in-place object update
    return one


def bench(nnx):
    nframes = nnx * nnx
    illum, tr, data, Nx, norm_reg_inv = simulate(NX, nnx, Dx)
    img = cp.ones((Nx, Nx), dtype=cp.complex64)
    img0 = cp.zeros((Nx, Nx), dtype=cp.complex64)
    F = cp.zeros((nframes, NX, NX), dtype=cp.complex64)
    one = make_iter(img, img0, F, tr, illum, data, norm_reg_inv, B)

    one(); cp.cuda.Stream.null.synchronize()               # warmup (alloc + JIT)

    # EAGER (per-iter synced)
    cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
    for _ in range(N):
        one(); cp.cuda.Stream.null.synchronize()
    eager = (time.perf_counter() - t0) / N * 1e3

    # EVENT (back-to-back, device time)
    ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
    cp.cuda.Stream.null.synchronize(); ev0.record()
    for _ in range(N):
        one()
    ev1.record(); ev1.synchronize()
    event = cp.cuda.get_elapsed_time(ev0, ev1) / N

    # GRAPH (capture one iter, replay)
    graph_ms, gerr = None, ""
    try:
        s = cp.cuda.Stream(non_blocking=True)
        with s:
            one()                                          # warmup on the capture stream
            s.synchronize()
            s.begin_capture()
            one()
            g = s.end_capture()
        s.synchronize(); t0 = time.perf_counter()
        for _ in range(N):
            g.launch(stream=s)
        s.synchronize()
        graph_ms = (time.perf_counter() - t0) / N * 1e3
    except Exception as ex:
        gerr = repr(ex)[:120]
    gstr = f"{graph_ms:8.2f}" if graph_ms is not None else f"{'n/a':>8}"
    spd = f"{eager/graph_ms:5.2f}x" if graph_ms else "  -  "
    print(f"{nframes:>8} {data.nbytes/1e9:>7.2f} | {eager:8.2f} {event:8.2f} {gstr} | {spd}"
          + (f"  GRAPH_FAIL: {gerr}" if graph_ms is None else ""))
    del illum, tr, data, img, img0, F; cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    try:
        cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool().malloc)   # capturable stream-ordered alloc
        alloc = "async"
    except Exception as ex:
        alloc = f"default ({repr(ex)[:40]})"
    print(f"NX={NX} B={B} N={N} alloc={alloc} | ms/iter (batched sync-off device AP core)")
    print(f"{'nframes':>8} {'data_GB':>7} | {'EAGER':>8} {'EVENT':>8} {'GRAPH':>8} | {'e/g':>6}")
    for nnx in NNXLIST:
        bench(nnx)
    print("\nEXPECT: EVENT < EAGER (per-iter host-sync overhead removed); GRAPH <= EVENT "
          "(kernel-launch + Python overhead removed) -> GRAPH is the honest deployable throughput.")
