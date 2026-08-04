"""Work-item 1 / memory axis: FRAME-BATCHED AP -- keep the object resident, process the
frames stack in batches of B, accumulate into the resident object. The frames stack is
O(nf*nx^2) = ~90% of resident (unified_solver_test); the main solver
(Alternating_projections_c, Solvers.py:622-673) materializes the WHOLE stack every iter
(frames = zeros(nf,...); split all; project all; frames_old copy; sync; overlap all) -> the
frame-count ceiling. Batching removes it: split/project are per-frame, overlap is a scatter-
SUM (so per-batch partial overlaps sum to the full overlap), and the KD-tree Gramian is
sparse so its inner products assemble in CHUNKS over the pair list (peak = unique frames per
pair-chunk, << nf). Object + data + probe stay resident; frames are DERIVED, never all held.

This CPU prototype proves the batched algorithm is BIT-EQUIVALENT to the full stack, so the
GPU port (Operators/Solvers) is mechanical:
  * ap(B=nf, chunk=nnz)  == the full-stack baseline (single batch).
  * ap(B<nf, chunk<nnz)  == batched; peak frames = max(B, unique-per-chunk).
Both use the SAME chunked-braket sync, so any full-vs-batched diff is pure float sum-ORDER
(the exactness result). A separate FAITHFULNESS check vs the production synchronize_frames_c
(numba Gramian) confirms the prototype reproduces the real solver.

CPU/numpy authoritative.  OMP_NUM_THREADS=1 ... nice -n 19 python3 frame_batch_test.py
env: NX(48) K(12) STEPD(4) MAXIT(40) B(16) PCHUNK(1024) SYNC(1) SOLVER(eigsh)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NX", "48")

import numpy as np
import Operators
from Operators import (Splitc, Overlapc, Project_data, synchronize_frames_c, braket_i,
                       eig_reset, xp)
import sync_bandpass_test as T

nx = T.nx
K = int(os.environ.get("K", 12))
MAXIT = int(os.environ.get("MAXIT", 40))
B = int(os.environ.get("B", 16))
PCHUNK = int(os.environ.get("PCHUNK", 1024))
SYNC = bool(int(os.environ.get("SYNC", 1)))
SOLVER = os.environ.get("SOLVER", "eigsh")


def _solver():
    return T.eigsh_sync if SOLVER == "eigsh" else T._POWER


def build_omega(F, ctx, pair_chunk, num_iter):
    """Assemble the Gramian H from the frame store F in CHUNKS of pair_chunk pairs --
    gathering only the frames each chunk references (peak = unique frames per chunk) -- then
    eigensolve. Mirrors synchronize_frames_c's math (framesl=conj(P)*F, framesr=framesl*
    inorm_split, val = braket / (fn[i]*fn[j])) exactly, but never holds all frames."""
    plan = ctx["Gramiam"]
    cP = ctx["cprobe"]; inorm = ctx["inorm_split"]; fn = ctx["frames_norm"]; bw = plan["bw"]
    col = np.asarray(plan["col"]).astype(np.int64); row = np.asarray(plan["row"]).astype(np.int64)
    dx = plan["dx"]; dy = plan["dy"]
    nnz = col.size
    val = xp.empty(nnz, dtype=xp.complex128)
    peak = 0
    # process pairs in FRAME-LOCALITY order (group by left frame) so each chunk touches few
    # frames; write val to the ORIGINAL pair slot so plan["val2H"] is unaffected.
    order = np.argsort(col, kind="stable")
    for s in range(0, nnz, pair_chunk):
        pidx = order[s:min(nnz, s + pair_chunk)]              # original indices of this chunk
        c = col[pidx]; r = row[pidx]
        U = np.unique(np.concatenate([c, r]))                 # frames this chunk touches
        peak = max(peak, U.size)
        pos = np.empty(int(U.max()) + 1, dtype=np.int64); pos[U] = np.arange(U.size)
        fl = F[U] * cP                                        # local framesl (<= |U| frames)
        fr = fl * inorm[U]                                    # local framesr
        cl = xp.asarray(pos[c]); rl = xp.asarray(pos[r])
        dxc = dx[pidx]; dyc = dy[pidx]
        for t in range(pidx.size):
            val[pidx[t]] = braket_i(t, fl, fr, cl, rl, dxc, dyc, bw) / (fn[c[t]] * fn[r[t]])
    H = plan["val2H"](val)
    Operators.Eigensolver = _solver()
    omega = Operators.Eigensolver(H, num_iter)
    Operators.Eigensolver = T._POWER
    return omega, peak


def ap(ctx, sync, maxit, B, pair_chunk, num_iter=5):
    """Frame-batched AP. B=nf & pair_chunk=nnz -> the full-stack baseline; smaller -> batched.
    Returns (img, curve, peak_frames_resident)."""
    probe, cprobe, mapid = ctx["probe"], ctx["cprobe"], ctx["mapid"]
    Nx, Ny, nf = ctx["Nx"], ctx["Ny"], ctx["nframes"]
    data, norm = ctx["data"], ctx["normalization"]
    eig_reset(); T._eigsh_v0["v"] = None
    img = xp.ones((Ny, Nx), dtype=xp.complex64)
    F = xp.empty((nf, nx, nx), dtype=xp.complex64)            # host store (device streams batches)
    curve = []; peak = 0
    for it in range(maxit):
        for s in range(0, nf, B):                            # PASS A: split+project -> store
            e = min(nf, s + B)
            fb = Splitc(img, mapid[s:e]) * probe
            fb, _ = Project_data(fb, data[s:e])
            F[s:e] = fb; peak = max(peak, e - s)
        omega = None
        if sync:
            omega, pf = build_omega(F, ctx, pair_chunk, num_iter); peak = max(peak, pf)
        img0 = xp.zeros((Ny, Nx), dtype=xp.complex64)
        for s in range(0, nf, B):                            # PASS B: (apply omega) + overlap
            e = min(nf, s + B)
            fb = F[s:e] if omega is None else F[s:e] * omega[s:e]
            img0 = img0 + Overlapc(fb * cprobe, Nx, Ny, mapid[s:e]); peak = max(peak, e - s)
        img = img0 / norm
        curve.append(T.band_err(ctx, img))
    return img, np.array(curve), peak


def ap_production(ctx, sync, maxit, num_iter=5):
    """The real full-stack path (mirrors sync_bandpass.run / Alternating_projections_c):
    full frames stack + production synchronize_frames_c (numba Gramian). Faithfulness ref."""
    from Operators import Illuminate_frames
    Operators.Eigensolver = _solver()
    eig_reset(); T._eigsh_v0["v"] = None
    probe, cprobe, mapid = ctx["probe"], ctx["cprobe"], ctx["mapid"]
    img = xp.ones((ctx["Ny"], ctx["Nx"]), dtype=xp.complex64)
    frames = Illuminate_frames(Splitc(img, mapid), probe)
    curve = []
    for it in range(maxit):
        frames, _ = Project_data(frames, ctx["data"])
        if sync:
            omega = synchronize_frames_c(frames, probe, ctx["frames_norm"], ctx["inorm_split"],
                                         ctx["Gramiam"], ctx["Gramiam"]["bw"], num_iter)
            frames = frames * omega
        img = Overlapc(Illuminate_frames(frames, cprobe), ctx["Nx"], ctx["Ny"], mapid) / ctx["normalization"]
        frames = Illuminate_frames(Splitc(img, mapid), probe)
        curve.append(T.band_err(ctx, img))
    Operators.Eigensolver = T._POWER
    return img, np.array(curve)


def dimg(a, b):
    return float(xp.max(xp.abs(a - b)) / (xp.max(xp.abs(b)) + 1e-30))


if __name__ == "__main__":
    T.STEPD = int(os.environ.get("STEPD", 4)); T.NUMITER = 5
    ctx = T.build(K)
    nf = ctx["nframes"]; nnz = int(np.asarray(ctx["Gramiam"]["col"]).size)
    ov = 100 * (1 - ctx["step"] / nx)
    print(f"NX={nx} K={K} ({nf} frames), img {ctx['Nx']}^2, overlap {ov:.0f}%, MAXIT={MAXIT} "
          f"SYNC={SYNC} SOLVER={SOLVER} | B={B} PCHUNK={PCHUNK} nnz={nnz}")

    img_full, cf, pk_full = ap(ctx, SYNC, MAXIT, nf, nnz)            # single-batch baseline
    img_bat, cb, pk_bat = ap(ctx, SYNC, MAXIT, B, PCHUNK)           # batched
    img_prod, cp = ap_production(ctx, SYNC, MAXIT)                  # production faithfulness ref

    print(f"\n== FRAME-BATCHING EXACTNESS (batched B={B} vs single-batch B={nf}) ==")
    print(f"  final recon rel-diff (max|Δimg|/max|img|) = {dimg(img_bat, img_full):.2e}   "
          f"(pure float sum-ORDER -> ~1e-6 = exact)")
    print(f"  low-band curve max|Δ| over {MAXIT} iters = {float(np.max(np.abs(cb[:,0]-cf[:,0]))):.2e}")
    print(f"\n== FAITHFULNESS (prototype single-batch vs PRODUCTION synchronize_frames_c) ==")
    print(f"  final recon rel-diff = {dimg(img_full, img_prod):.2e}   "
          f"(chunked-braket vs numba Gramian; ~1e-6 = faithful)")
    print(f"  final low-band: batched {cb[-1,0]:.4f} | single {cf[-1,0]:.4f} | production {cp[-1,0]:.4f}")

    print(f"\n== PEAK FRAMES RESIDENT (device model; frames stack = O(peak*nx^2)) ==")
    print(f"  full-stack (production)  peak ~= nf   = {nf:>5}  frames  ({nf*nx*nx*8/1e6:.1f} MB @nx={nx})")
    print(f"  batched                  peak        = {pk_bat:>5}  frames  ({pk_bat*nx*nx*8/1e6:.1f} MB)  "
          f"= max(B={B}, unique-per-pair-chunk)")
    print(f"  reduction factor         ~= nf/peak  = {nf/max(1,pk_bat):.1f}x   "
          f"(independent of nf -> removes the frame-count ceiling)")
    print("\nEXPECT: exactness ~1e-6 (batched == full, split/project per-frame + overlap scatter-SUM + "
          "sparse-Gramian chunked); faithful ~1e-6 vs production; peak frames = O(B) not O(nf).")
