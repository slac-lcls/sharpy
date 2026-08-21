#!/usr/bin/env python3
"""Eigensolver A/B on REAL sharpy problems.

gpu_eig_bench.py times the sync eigensolvers on a SYNTHETIC connection Gramian at G=64..160
(n = 4k..25k frames). The question this answers instead: does the fused/momentum/graph
eigensolver (PR #15) do anything at the frame counts WE actually run -- 445-frame angular_only,
~1410-frame jittery -- on the real overlap geometry and real reconstructed object/probe.

Two phases so every config solves the IDENTICAL matrix:
  BUILD=1  -> build H from a real .ptyd geometry + real recon npz, save to H_<TAG>.npz
  BUILD=0  -> load H, run Eigensolver under whatever SHARPY_* flags are set, print time,
              and save the phase vector for cross-config comparison.

The flags are read at Operators import time, so each config must be a separate process.

  PTYD=... NPZ=... TAG=ang BUILD=1 python real_eig_bench.py
  SHARPY_FUSED_EIG=1 TAG=ang LABEL=fused python real_eig_bench.py
"""
import os, time
import numpy as np
import config; config.GPU = True
import cupy as cp
import Operators as ops

xp = cp
W = os.environ.get("WORK", ".")
TAG = os.environ.get("TAG", "ang")
LABEL = os.environ.get("LABEL", "baseline")
NUM_ITER = int(os.environ.get("NUM_ITER", "50"))
REPS = int(os.environ.get("REPS", "10"))
HPATH = os.path.join(W, f"H_{TAG}.npz")


def build():
    import h5py
    ptyd, npz = os.environ["PTYD"], os.environ["NPZ"]
    with h5py.File(ptyd, "r") as f:
        pos = np.asarray(f["info/positions_scan"][:])          # integer pixel positions
    z = np.load(npz)
    obj = xp.asarray(z["obj"]).astype(xp.complex64)
    probe = xp.asarray(z["probe"]).astype(xp.complex64)
    Ny, Nx = int(obj.shape[0]), int(obj.shape[1])
    nx = int(probe.shape[-1]); ny = int(probe.shape[-2])
    tx = xp.asarray(pos[:, 0].astype(float)); ty = xp.asarray(pos[:, 1].astype(float))
    nf = int(tx.size)
    print(f"[build] {TAG}: {nf} frames, frame {ny}x{nx}, canvas {Ny}x{Nx}", flush=True)

    mapid = ops.map_frames(tx, ty, nx, ny, Nx, Ny)
    frames = ops.Illuminate_frames(ops.Splitc(obj, mapid), probe)   # real exit waves
    plan = ops.Gramiam_plan(tx, ty, nf, nx, ny, Nx, Ny, bw=0)
    QQ = ops.Overlapc(xp.tile(xp.abs(probe) ** 2, (nf, 1, 1)), Nx, Ny, mapid)
    inorm = ops.Splitc((1.0 / QQ).astype(xp.complex64), mapid)
    fnorm = ops.Precondition_calc(frames, bw=0)
    H = ops.Gramiam_calc_cuda(frames, plan, probe, inorm, fnorm)
    H = H.tocsr() if hasattr(H, "tocsr") else H
    np.savez(HPATH, data=cp.asnumpy(H.data), indices=cp.asnumpy(H.indices),
             indptr=cp.asnumpy(H.indptr), shape=np.asarray(H.shape), nframes=nf)
    nnz = int(H.data.size)
    print(f"[build] H {H.shape} nnz={nnz} ({nnz / nf:.1f} pairs/frame) -> {HPATH}", flush=True)


def run():
    import cupyx.scipy.sparse as csp
    d = np.load(HPATH)
    H = csp.csr_matrix((xp.asarray(d["data"]), xp.asarray(d["indices"]), xp.asarray(d["indptr"])),
                       shape=tuple(int(s) for s in d["shape"]))
    n = H.shape[0]
    flags = {k: os.environ.get(k, "") for k in
             ("SHARPY_FUSED_EIG", "SHARPY_EIG_MOMENTUM", "SHARPY_EIG_GRAPH", "SHARPY_EIG_WINDOWED")}

    # ⚠ Eigensolver caches its last eigenvector in the module global _eig_v0 and warm-starts the
    # next call from it. Timing REPS back-to-back therefore measures ONE cold solve followed by
    # REPS-1 warm ones and reports a bimodal median -- which is how the first version of this
    # bench produced medians that disagreed run-to-run. Measure the two cases separately:
    #   MODE=cold  reset the cache before every rep   (first sync of a reconstruction)
    #   MODE=warm  keep it                            (every subsequent AP iteration -- the
    #                                                   common case in the loop)
    MODE = os.environ.get("MODE", "warm")

    def reset():
        if hasattr(ops, "_eig_v0"):
            ops._eig_v0 = None

    v = ops.Eigensolver(H, NUM_ITER)                       # warm-up / compile
    cp.cuda.runtime.deviceSynchronize()
    ts = []
    for _ in range(REPS):
        if MODE == "cold":
            reset()
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        v = ops.Eigensolver(H, NUM_ITER)
        cp.cuda.runtime.deviceSynchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    lbl = f"{LABEL}_{MODE}"          # NOT `LABEL = ...`: rebinding a module global inside the
                                     # function makes it local for the whole body -> UnboundLocalError
    ph = np.angle(cp.asnumpy(xp.asarray(v).ravel()))
    np.save(os.path.join(W, f"phase_{TAG}_{lbl}.npy"), ph)
    print(f"{lbl:22s} n={n:6d} | {np.median(ts):8.3f} ms (min {ts.min():.3f}, "
          f"spread {ts.std():.3f}) | flags {flags}", flush=True)


def build_synth():
    """Sparse connection Gramian on a grid scan -- the regime PR #15 was tuned in.

    Our real cells are the OPPOSITE corner of the space: small n, but frames so large relative
    to the canvas that every pair overlaps (H is dense, nnz = n^2). This builds the other corner
    -- large n, small frames, few neighbours each -- with the same sharpy machinery, so the two
    are measured by one code path and the crossover (if any) is visible.
    """
    G = int(os.environ.get("G", "64")); nx = int(os.environ.get("NXF", "32"))
    step = int(os.environ.get("STEP", "8"))
    Nx = Ny = G * step
    rng = np.random.default_rng(0)
    truth = xp.asarray((rng.standard_normal((Ny, Nx)) + 1j * rng.standard_normal((Ny, Nx))
                        ).astype(np.complex64))
    gx, gy = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    tx = xp.asarray(gx.ravel() * step, dtype=float); ty = xp.asarray(gy.ravel() * step, dtype=float)
    nf = int(tx.size)
    probe = xp.asarray((rng.standard_normal((nx, nx)) + 1j * rng.standard_normal((nx, nx))
                        ).astype(np.complex64))
    mapid = ops.map_frames(tx, ty, nx, nx, Nx, Ny)
    frames = ops.Illuminate_frames(ops.Splitc(truth, mapid), probe)
    plan = ops.Gramiam_plan(tx, ty, nf, nx, nx, Nx, Ny, bw=0)
    QQ = ops.Overlapc(xp.tile(xp.abs(probe) ** 2, (nf, 1, 1)), Nx, Ny, mapid)
    inorm = ops.Splitc((1.0 / QQ).astype(xp.complex64), mapid)
    fnorm = ops.Precondition_calc(frames, bw=0)
    H = ops.Gramiam_calc_cuda(frames, plan, probe, inorm, fnorm)
    H = H.tocsr() if hasattr(H, "tocsr") else H
    np.savez(HPATH, data=cp.asnumpy(H.data), indices=cp.asnumpy(H.indices),
             indptr=cp.asnumpy(H.indptr), shape=np.asarray(H.shape), nframes=nf)
    nnz = int(H.data.size)
    print(f"[synth] {TAG}: n={nf} frame {nx}^2 canvas {Ny}x{Nx} | nnz={nnz} "
          f"({nnz / nf:.1f} pairs/frame, density {nnz / nf**2:.4f}) -> {HPATH}", flush=True)


if os.environ.get("SYNTH", "0") == "1":
    build_synth()
elif os.environ.get("BUILD", "0") == "1":
    build()
else:
    run()
