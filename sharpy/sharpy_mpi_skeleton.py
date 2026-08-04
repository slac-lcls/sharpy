#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""sharpy MPI scaffold — Workstream C2 (distributed CuPy + mpi4py).

A STARTING POINT, not finished code (teaching-mode scaffold). The plumbing is
sketched so you can run it and watch the mechanics; the `TODO(...)` blocks are
where the real work — and the learning — go. Drive Claude in Learning mode.

Run the CPU smoke test (no GPU needed; works with or without mpi4py):
    python sharpy_mpi_skeleton.py            # single process
    mpirun -n 4 python sharpy_mpi_skeleton.py
It builds a toy "object accumulation", does it single-process AND distributed,
and asserts they match. Get that green first, then start the TODOs.

Lift the real pieces from Stefano's repos:
  - ptychoMPI  https://github.com/smarkesini/ptychoMPI  (decomposition, AP, same split/overlap kernels)
  - xpack      https://github.com/smarkesini/xpack      (the UPGRADED communicator: large vectors past
               the ~2**31 element count limit + irregular per-rank sizes — use THIS one)
"""
import numpy as np

try:
    from mpi4py import MPI
    _COMM = MPI.COMM_WORLD
    RANK, SIZE = _COMM.Get_rank(), _COMM.Get_size()
except Exception:                       # no mpi4py -> single process, still runnable
    MPI = None; _COMM = None; RANK, SIZE = 0, 1

try:
    import cupy as cp                   # optional; the smoke test stays on CPU
    # On Perlmutter with --gpus-per-task=1, SLURM sets CUDA_VISIBLE_DEVICES so
    # each task sees only its own GPU as device 0.  Always use Device(0).
    # The old RANK % 4 pattern fails at N>1 because device 1+ don't exist in
    # that task's CUDA_VISIBLE_DEVICES view → cp = None → _xp = np.
    cp.cuda.Device(0).use()
except Exception:
    cp = None


# ---------------------------------------------------------------------------
# 1. communicator  (minimal subset for the smoke test; for the real buffers take
#    xpack's communicator — large / irregular-size capable)
# ---------------------------------------------------------------------------
def mpi_allSum(buf, cuda=False):
    """Allreduce(SUM). CUDA-aware path reduces device buffers directly; else stage via host.

    Complex64 arrays are reduced via a float32 view (re/im treated as flat float32 pairs).
    This avoids MPI_C_COMPLEX handling bugs on some implementations (normalization worked
    because it's purely real; the image update is genuinely complex and exposed the issue).
    """
    if _COMM is None or SIZE == 1:
        return buf

    xp = cp if (cp is not None and isinstance(buf, cp.ndarray)) else np

    if cuda and cp is not None and isinstance(buf, cp.ndarray):
        # DONE -- I HAVE IT ON SCRATCH!!!
        # TODO(human): on Perlmutter this needs GPU-aware cray-mpich +
        #              MPICH_GPU_SUPPORT_ENABLED=1; if not available, drop to the host path.
        out = cp.empty_like(buf)
        if buf.dtype == xp.complex64:
            _COMM.Allreduce(buf.view(xp.float32), out.view(xp.float32), op=MPI.SUM)
        else:
            _COMM.Allreduce(buf, out, op=MPI.SUM)
        return out

    # Host fallback path
    host = buf.get() if (cp is not None and isinstance(buf, cp.ndarray)) else np.ascontiguousarray(buf)
    red = np.empty_like(host)
    if host.dtype == np.complex64:
        _COMM.Allreduce(host.view(np.float32), red.view(np.float32), op=MPI.SUM)
    else:
        _COMM.Allreduce(host, red, op=MPI.SUM)
    return cp.asarray(red) if (cp is not None and isinstance(buf, cp.ndarray)) else red


def mpi_bcast(obj, root=0):
    return obj if (_COMM is None or SIZE == 1) else _COMM.bcast(obj, root=root)


def mpi_bcast_array(arr, root=0):
    """Broadcast a numpy array using the MPI buffer protocol (avoids pickle size limits)."""
    if _COMM is None or SIZE == 1:
        return arr
    meta = (arr.shape, arr.dtype.str) if RANK == root else None
    shape, dtype_str = _COMM.bcast(meta, root=root)
    if RANK != root:
        arr = np.empty(shape, dtype=np.dtype(dtype_str))
    _COMM.Bcast(arr, root=root)
    return arr


def mpi_barrier():
    if _COMM is not None:
        _COMM.Barrier()


# TODOO(human, Delight): for the real object/frames, lift xpack's large/irregular-vector
# ops (Gatherv/Scatterv with per-rank counts, plus >2**31-element chunking) to assemble
# and write the full object. The toy test below is small, so plain Allreduce suffices.

# === ADDED: xpack large/irregular communicator implementation ===
def get_chunk_slices(n_slices):
    """Calculates irregular per-rank start/stop offsets for optimal load balance."""
    chunk_size = int(np.ceil(n_slices / SIZE))
    nreduce = (chunk_size * SIZE - n_slices)
    start = np.concatenate((np.arange(SIZE - nreduce) * chunk_size,
                            (SIZE - nreduce) * chunk_size + np.arange(nreduce) * (chunk_size - 1)))
    stop = np.append(start[1:], n_slices)

    start = start.reshape((SIZE, 1))
    stop = stop.reshape((SIZE, 1))
    slices = np.longlong(np.concatenate((start, stop), axis=1))
    return slices


# ---------------------------------------------------------------------------
# 2D spatial decomposition (ptychoMPI pattern)
# ---------------------------------------------------------------------------
import math as _math

def _is_decomposition_in_rank(translation, rank, size, trans_x, trans_y):
    min_tx, max_tx = trans_x
    min_ty, max_ty = trans_y
    size_x = int(_math.sqrt(size))
    size_y = int(_math.ceil(float(size) / size_x))
    last_row = int(size - (size_x * (size_y - 1)))
    my_ix = int(rank % size_x)
    my_iy = int(rank / size_x)
    frac_y = (translation[1] - min_ty) / ((max_ty - min_ty) * (1 + 2.2e-16))
    pos_y  = max(0, min(size_y - 1, int(frac_y * size_y)))
    frac_x = (translation[0] - min_tx) / ((max_tx - min_tx) * (1 + 2.2e-16))
    if pos_y == size_y - 1:
        pos_x = max(0, min(last_row - 1, int(frac_x * last_row)))
    else:
        pos_x = max(0, min(size_x - 1, int(frac_x * size_x)))
    return pos_x == my_ix and pos_y == my_iy


def _calculate_decomposition(rank, size, translations):
    """Return sorted list of global frame indices assigned to `rank`."""
    if size == 1:
        return list(range(len(translations)))
    min_tx, max_tx = translations[:, 0].min(), translations[:, 0].max()
    min_ty, max_ty = translations[:, 1].min(), translations[:, 1].max()
    trans_x = (min_tx, max_tx); trans_y = (min_ty, max_ty)
    return [i for i in range(len(translations))
            if _is_decomposition_in_rank(translations[i], rank, size, trans_x, trans_y)]


def get_2d_decomposition(tx_np, ty_np):
    """
    2D spatial decomposition: each rank gets the frames whose (tx, ty) scan
    position falls in its √SIZE × √SIZE tile of the bounding box.

    Returns a dict with:
      my_frames      : (local_n,) int64  — global frame indices for this rank
      frame_to_rank  : (nframes,) int32  — global frame g → owning rank
      counts         : (SIZE,)    int64  — frames per rank (for Allgatherv/Gatherv)
      global_to_local: dict g → local index within my_frames
      nframes        : int
    """
    nframes = len(tx_np)
    trans   = np.stack([tx_np.ravel(), ty_np.ravel()], axis=1).astype(np.float64)
    my_frames = np.array(_calculate_decomposition(RANK, SIZE, trans), dtype=np.int64)

    local_count = np.array([len(my_frames)], dtype=np.int64)
    counts      = np.empty(SIZE, dtype=np.int64)
    if _COMM is not None:
        _COMM.Allgather(local_count, counts)
    else:
        counts[0] = local_count[0]

    # Allgather all frame index lists so every rank can build frame_to_rank
    all_frames = np.empty(nframes, dtype=np.int64)
    if _COMM is not None:
        _COMM.Allgatherv(my_frames, (all_frames, counts.tolist()))
    else:
        all_frames[:] = my_frames

    frame_to_rank = np.empty(nframes, dtype=np.int32)
    pos = 0
    for r, cnt in enumerate(counts):
        frame_to_rank[all_frames[pos:pos + int(cnt)]] = r
        pos += int(cnt)

    return dict(
        my_frames       = my_frames,
        frame_to_rank   = frame_to_rank,
        counts          = counts,
        global_to_local = {int(g): k for k, g in enumerate(my_frames)},
        nframes         = nframes,
    )

def scatterv(data, chunk_slices, slice_shape):
    """Scatters global data to ranks using a 4-float block type to bypass 2**31 limit."""
    if SIZE == 1:
        return data[chunk_slices[0, 0]:chunk_slices[0, 1], ...]

    dspl = chunk_slices[:, 0]
    cnt = (chunk_slices[:, 1] - chunk_slices[:, 0]).astype(np.int64)  # 1D, not (SIZE,1)
    sdim = np.prod(slice_shape)
    chunk_shape = (np.append(int(cnt[RANK]), slice_shape))
    data_local = np.empty(chunk_shape, dtype='float32')

    # Create chunk datatype to defeat the 32-bit signed integer element count limit
    mpichunktype = MPI.FLOAT.Create_contiguous(4).Commit()
    sdim = sdim * MPI.FLOAT.Get_size() // mpichunktype.Get_size()
    _COMM.Scatterv([data, tuple(cnt * sdim), tuple(dspl * sdim), mpichunktype], data_local)
    mpichunktype.Free()

    return data_local

def gatherv(data_local, chunk_slices, data=None): 
    """Gathers irregular pieces from ranks back to root node with chunking."""
    if SIZE == 1: 
        if data is None:
            data = data_local + 0
        else:
            data[...] = data_local[...]
        return data

    cnt = (chunk_slices[:, 1] - chunk_slices[:, 0]).astype(np.int64)   # 1D counts per rank
    slice_shape = data_local.shape[1:]
    sdim = int(np.prod(slice_shape))
    
    if RANK == 0 and data is None:
        tshape = (np.append(chunk_slices[-1, -1] - chunk_slices[0, 0], slice_shape))
        data = np.empty(tuple(tshape), dtype=data_local.dtype)

    # Gather via contiguous chunk tracking
    mpichunktype = MPI.FLOAT.Create_contiguous(4).Commit()
    sdim = sdim * MPI.FLOAT.Get_size() // mpichunktype.Get_size()
    recvbuf = [data, (cnt * sdim, None), mpichunktype] if RANK == 0 else None
    _COMM.Gatherv(sendbuf=[data_local, mpichunktype], recvbuf=recvbuf)
    mpichunktype.Free()
    
    return data


# ---------------------------------------------------------------------------
# 2. decomposition — 2D spatial tiling by scan position
# ---------------------------------------------------------------------------
def _grid(size):
    import math
    gx = max(1, int(round(math.sqrt(size))))
    gy = int(math.ceil(size / gx))
    return gx, gy


def frame_ranks(translations, size):
    if size == 1:
        return np.zeros(translations.shape[0], dtype=int)
    gx, gy = _grid(size)

    def frac(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo + 1e-12)

    ix = np.clip((frac(translations[:, 0].astype(float)) * gx).astype(int), 0, gx - 1)
    iy = np.clip((frac(translations[:, 1].astype(float)) * gy).astype(int), 0, gy - 1)
    return np.clip(iy * gx + ix, 0, size - 1)


def calculate_decomposition(translations, rank, size):
    return np.where(frame_ranks(translations, size) == rank)[0]


# ---------------------------------------------------------------------------
# 3. toy "object accumulation"
# ---------------------------------------------------------------------------
def _accumulate(shape, translations, values, idx, nx):
    obj = np.zeros(shape, dtype=np.float64)
    for i in idx:                                  
        x, y = int(translations[i, 0]), int(translations[i, 1])
        obj[y:y + nx, x:x + nx] += values[i]
    return obj


# ---------------------------------------------------------------------------
# 2. decomposition — 2D spatial tiling by scan position
#    (adapted from ptychoMPI/common/decomposition.py). Each rank owns the frames
#    whose (x,y) translation lands in its tile of a gx x gy grid.
# ---------------------------------------------------------------------------
def _grid(size):
    import math
    gx = max(1, int(round(math.sqrt(size))))
    gy = int(math.ceil(size / gx))
    return gx, gy


def frame_ranks(translations, size):
    """Owning rank for every frame, by scan position (linear tile id; overflow -> last rank)."""
    if size == 1:
        return np.zeros(translations.shape[0], dtype=int)
    gx, gy = _grid(size)

    def frac(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo + 1e-12)

    ix = np.clip((frac(translations[:, 0].astype(float)) * gx).astype(int), 0, gx - 1)
    iy = np.clip((frac(translations[:, 1].astype(float)) * gy).astype(int), 0, gy - 1)
    return np.clip(iy * gx + ix, 0, size - 1)


def calculate_decomposition(translations, rank, size):
    """Indices of the frames belonging to `rank`."""
    return np.where(frame_ranks(translations, size) == rank)[0]


# ---------------------------------------------------------------------------
# 3. toy "object accumulation" — the part you REPLACE with sharpy.
#    single-process: every frame's patch summed into the object.
#    distributed:    each rank does only its tile's frames, then mpi_allSum.
#    Disjoint partition + additive accumulation => the two MUST be identical.
# ---------------------------------------------------------------------------
def _accumulate(shape, translations, values, idx, nx):
    obj = np.zeros(shape, dtype=np.float64)
    for i in idx:                                  # <- replace with sharpy Overlap(Split(...))
        x, y = int(translations[i, 0]), int(translations[i, 1])
        obj[y:y + nx, x:x + nx] += values[i]
    return obj


def smoke_test():
    rng = np.random.default_rng(0)                 # SAME seed on every rank -> identical data
    nframes, nx, N = 64, 8, 48
    translations = rng.integers(0, N - nx, size=(nframes, 2)).astype(float)
    values = rng.standard_normal((nframes, nx, nx))

    ref = _accumulate((N, N), translations, values, np.arange(nframes), nx)   # single-process reference
    mine = calculate_decomposition(translations, RANK, SIZE)                  # this rank's frame indices

    # Each rank accumulates only its own frames, then allSum gives the global result.
    # gatherv concatenates slices — wrong for image accumulation which is additive.
    # mpi_allSum is the right operation: every rank gets the full summed image.
    local_accumulation = _accumulate((N, N), translations, values, mine, nx).astype('float32')
    total = mpi_allSum(local_accumulation)   # all ranks participate; all get the result

    ok = np.allclose(total, ref.astype('float32'), atol=1e-5)
    counts = _COMM.gather(len(mine), root=0) if _COMM is not None else [len(mine)]

    if RANK == 0:
        print(f"[smoke] ranks={SIZE} frames/rank={counts} covered={sum(counts)}/{nframes} "
              f"match={ok} max|err|={np.max(np.abs(total - ref)):.2e}")
    assert ok, "distributed != single-process — decomposition or allreduce is wrong"
    return ok


# ===========================================================================
# Distributed AP step (replaces toy _accumulate)
# ===========================================================================

# CUDA-aware MPI: enabled automatically when cray-mpich is loaded with
# MPICH_GPU_SUPPORT_ENABLED=1. Without it, Allreduce on GPU pointers reads
# garbage. The flag here follows the same env var so no code change is needed
# when switching — just set the env var and rerun.
import os as _os
_CUDA_AWARE_MPI = _os.environ.get("MPICH_GPU_SUPPORT_ENABLED", "0") == "1"

try:
    from Operators import Project_data
    from wrap_ops import split_cuda, overlap_cuda
    import config as _cfg
    _xp = cp if (cp is not None and _cfg.GPU) else np
    _SHARPY = True
except Exception:
    _SHARPY = False
    _xp = np


def _compute_normalization(illumination, translations_xp, Nx, Ny):
    """
    Pixel coverage map: |illumination|^2 summed at every scan position.
    Each rank computes its local slice, allSum gives the global map.
    Called once before the AP loop — reused every iteration.

    overlap_cuda with frames=0 accumulates coverage without any frame data.
    """
    norm_local = _xp.zeros((Nx, Ny), dtype=_xp.complex64)
    overlap_cuda(norm_local, 0, translations_xp, illumination)
    is_gpu = cp is not None and isinstance(norm_local, cp.ndarray)
    norm_local_cpu = norm_local.get() if is_gpu else norm_local  # GPU→CPU sync (xcale pattern)
    norm_global_cpu = mpi_allSum(norm_local_cpu)
    return _xp.asarray(norm_global_cpu) if is_gpu else norm_global_cpu


def mpi_AP_step(img, local_data, illumination, local_translations,
                nx, ny, Nx, Ny, normalization,
                F_local, frames_norm, decomp,
                sync=True, sync_this=True,
                num_iter=5, frame_batch=1024, reg=1e-8,
                object_halo=None, eps=None, halo_geometry=None):
    """
    One distributed AP iteration with optional Gramian phase sync.

    Two-pass structure (matches Alternating_projections_batched_c):

    PASS A — Split + Project_data per batch → store in F_local.
      img is replicated on every rank. Each rank extracts only ITS patches.
      Frames are stored (not discarded) so the sync phase can see them all.

    SYNC (if sync and sync_this) — Gramian phase synchronization.
      mpi_gramian_sync builds H_local + H_cross across rank boundaries,
      runs distributed power iteration, returns omega (phase correction).
      F_local *= omega folds the correction in before PASS B.

    PASS B — Overlap per batch → AllSum.
      Corrected frames pasted back into a zero buffer, summed across ranks,
      divided by normalization. ONE Allreduce per call (same as before).

    Parameters
    ----------
    F_local     : (local_nframes, nx, ny) complex64 — pre-allocated, mutated
    frames_norm : (local_nframes,)         complex64 — pre-allocated, mutated
    decomp      : dict from get_2d_decomposition (frame_to_rank, counts, global_to_local)
    sync        : enable Gramian sync at all (False keeps pure AP behaviour)
    sync_this   : whether to sync THIS iteration (caller computes ii%sync_interval==0)
    object_halo : dict from mpi_object_halo.setup_object_halo(...), or None.
                  When given, PASS B sums only the overlap strips between
                  ranks (O(halo), independent of rank count) instead of
                  mpi_allSum-ing the whole Nx x Ny canvas -- see
                  mpi_object_halo.py / validate_object_halo.py. The returned
                  img is then only correct inside object_halo['tiles_own'] on
                  each rank (the object is no longer replicated); callers
                  that need a full replicated canvas must keep object_halo=None.
    eps         : pre-computed reg * max(abs(normalization)), or None to
                  compute it here. normalization is static across iterations,
                  so callers running a loop should compute this once outside
                  it instead of forcing a full-canvas device sync every call.
    halo_geometry : dict from mpi_halo.setup_halo_geometry(...), or None.
                  Passed through to mpi_gramian_sync -- see its docstring.
                  Build once (positions-only), reuse across every sync_this
                  call while positions are frozen.
    """
    local_nframes = local_data.shape[0]

    # PASS A: batched split + data projection → F_local
    for s in range(0, local_nframes, frame_batch):
        e = min(local_nframes, s + frame_batch)
        fb = _xp.zeros((e - s, nx, ny), dtype=_xp.complex64)
        split_cuda(img, fb, local_translations[s:e], illumination)
        fb, _ = Project_data(fb, local_data[s:e], compute_residuals=False)
        F_local[s:e] = fb
        frames_norm[s:e] = _xp.linalg.norm(fb, axis=(1, 2)).astype(_xp.complex64)

    # SYNC: distributed Gramian eigenvector → fold phase correction into F_local
    if sync and sync_this:
        omega = mpi_gramian_sync(F_local, illumination, normalization,
                                 local_translations, decomp,
                                 nx, ny, Nx, Ny, num_iter=num_iter,
                                 halo_geometry=halo_geometry)
        F_local *= omega.reshape(-1, 1, 1)

    # PASS B: batched overlap accumulation → AllSum (or halo exchange) → image update
    img0 = _xp.zeros((Nx, Ny), dtype=_xp.complex64)
    for s in range(0, local_nframes, frame_batch):
        e = min(local_nframes, s + frame_batch)
        overlap_cuda(img0, F_local[s:e], local_translations[s:e], illumination)

    if eps is None:
        eps = reg * float(_xp.max(_xp.abs(normalization)).real)

    if object_halo is not None:
        from mpi_object_halo import exchange_object_halo
        exchange_object_halo(img0, object_halo)     # <-- replaces mpi_allSum(img0)
        # Only tiles_own is meaningful after the exchange (see docstring) --
        # dividing the full (Nx,Ny) canvas would touch pixels nobody reads.
        for (x0, x1, y0, y1) in object_halo['tiles_own']:
            img0[y0:y1, x0:x1] /= (normalization[y0:y1, x0:x1] + eps)
        return img0
    else:
        is_gpu = cp is not None and isinstance(img0, cp.ndarray)
        img0_cpu = img0.get() if is_gpu else img0
        img_global_cpu = mpi_allSum(img0_cpu)
        img_global = _xp.asarray(img_global_cpu) if is_gpu else img_global_cpu
        return img_global / (normalization + eps)


def validate_distributed_ap(maxiter=5):
    """
    Validate the distributed AP step against the single-GPU sharpy reference.
    Run on a GPU node:
        srun -n 4 python sharpy_mpi_skeleton.py --validate-ap

    Residual float32 differences (~1e-6) are expected: the distributed version
    sums 4 × 64-frame CPU Allreduce results while the reference accumulates all
    256 frames in a single GPU kernel.  atol=1e-4 gives comfortable headroom.
    """
    if not _SHARPY:
        if RANK == 0:
            print("[validate-ap] sharpy not importable — run on a GPU node")
        return

    from poster_simulate import simulate
    data, illumination, truth, tx, ty, nframes, nx, ny, Nx, Ny = simulate(16)

    # Single-GPU reference — all ranks compute independently (no MPI)
    tx_full = tx if isinstance(tx, _xp.ndarray) else _xp.asarray(tx)
    ty_full = ty if isinstance(ty, _xp.ndarray) else _xp.asarray(ty)
    trans_full = (tx_full + 1j * ty_full).astype(_xp.complex64)

    norm_ref = _xp.zeros((Nx, Ny), dtype=_xp.complex64)
    overlap_cuda(norm_ref, 0, trans_full, illumination)
    eps_ref = 1e-8 * float(_xp.max(_xp.abs(norm_ref)).real)
    img_ref = _xp.ones((Nx, Ny), dtype=_xp.complex64)
    for _ in range(maxiter):
        frames = _xp.zeros((nframes, nx, ny), dtype=_xp.complex64)
        split_cuda(img_ref, frames, trans_full, illumination)
        frames, _ = Project_data(frames, data, compute_residuals=False)
        buf = _xp.zeros((Nx, Ny), dtype=_xp.complex64)
        overlap_cuda(buf, frames, trans_full, illumination)
        img_ref = buf / (norm_ref + eps_ref)

    # Distributed MPI version
    tx_np   = tx.get()   if (cp is not None and isinstance(tx,   cp.ndarray)) else np.array(tx)
    ty_np   = ty.get()   if (cp is not None and isinstance(ty,   cp.ndarray)) else np.array(ty)
    data_np = data.get() if (cp is not None and isinstance(data, cp.ndarray)) else np.array(data)

    decomp        = get_2d_decomposition(tx_np, ty_np)
    mf            = decomp['my_frames']
    local_data_xp = _xp.asarray(data_np[mf])
    local_trans   = (_xp.asarray(tx_np[mf]) + 1j * _xp.asarray(ty_np[mf])).astype(_xp.complex64)

    normalization = _compute_normalization(illumination, local_trans, Nx, Ny)

    local_nframes = local_data_xp.shape[0]
    F_local     = _xp.zeros((local_nframes, nx, ny), dtype=_xp.complex64)
    frames_norm = _xp.zeros(local_nframes, dtype=_xp.complex64)
    img_mpi = _xp.ones((Nx, Ny), dtype=_xp.complex64)
    for ii in range(maxiter):
        img_mpi = mpi_AP_step(img_mpi, local_data_xp, illumination,
                              local_trans, nx, ny, Nx, Ny, normalization,
                              F_local, frames_norm, decomp, sync=False)

    ref_np = img_ref.get() if (cp is not None and isinstance(img_ref, cp.ndarray)) else np.array(img_ref)
    mpi_np = img_mpi.get() if (cp is not None and isinstance(img_mpi, cp.ndarray)) else np.array(img_mpi)
    ok  = np.allclose(mpi_np, ref_np, atol=1e-4)
    err = float(np.max(np.abs(mpi_np - ref_np)))
    if RANK == 0:
        print(f"[validate-ap] ranks={SIZE}  match={ok}  max|err|={err:.2e}")
    assert ok, "MPI AP does not match single-GPU reference"


# ===========================================================================
# Distributed Gramian sync (TODO 2)
# ===========================================================================
from mpi_halo import setup_halo, exchange_v_halo, setup_halo_geometry, exchange_halo_frames


def mpi_gramian_sync(local_frames, illumination, normalization,
                     local_translations, decomp, nx, ny, Nx, Ny, num_iter=5,
                     halo_geometry=None):
    """
    Distributed Gramian phase synchronization with halo exchange.

    1. Build intra-rank H_local.
    2. setup_halo(): Allgather translations → detect cross-rank pairs →
       Isend/Irecv boundary frames → build H_cross (local × n_halo dense).
    3. Power iteration: exchange_v_halo() each step so the matvec
       (H_local @ v + H_cross @ v_halo) uses the full globally-consistent v.
       Gap-aware stop (SHARPY_SYNC_EIGTOL) via global Allreduce.

    halo_geometry : dict from mpi_halo.setup_halo_geometry(...), or None.
                    When given, only the content-dependent half of halo setup
                    (exchange_halo_frames -- current frame data + H_cross)
                    reruns; the positions-only half (3 Allgathervs + a global
                    Gramiam_plan + XOR-filtering boundary pairs) is skipped
                    and the cached geometry reused. Build once, before the AP
                    loop, and rebuild only if positions move (e.g. inside
                    position refinement). None recomputes geometry every call
                    (old, always-correct-but-wasteful behaviour).

    Returns omega: (local_nframes,) complex64 phase-correction vector.
    """
    from Operators import Gramiam_plan, Gramiam_calc_cuda, SYNC_EIGTOL

    local_nframes = local_frames.shape[0]
    local_tx = local_translations.real
    local_ty = local_translations.imag

    plan = Gramiam_plan(local_tx, local_ty, local_nframes, nx, ny, Nx, Ny)
    if plan["col"].size == 0:
        return _xp.ones(local_nframes, dtype=_xp.complex64)

    frames_norm = _xp.linalg.norm(local_frames, axis=(1, 2)).astype(_xp.complex64)
    eps      = 1e-8 * float(_xp.max(_xp.abs(normalization)).real)
    norm_reg = (normalization + eps).astype(_xp.complex64)
    H_local  = Gramiam_calc_cuda(local_frames, plan, illumination, norm_reg, frames_norm)
    # Sparse data may contain NaN for pairs whose probe footprint reaches past
    # the image edge (tx + nx > Nx): the zQQz kernel reads normalization
    # out-of-bounds (returns 0) and divides → NaN.  Zero those entries out.
    if hasattr(H_local, 'data'):
        H_local.data[:] = _xp.nan_to_num(H_local.data, nan=0.0, posinf=0.0, neginf=0.0)

    if halo_geometry is not None:
        halo = exchange_halo_frames(local_frames, local_tx, local_ty, frames_norm,
                                    illumination, norm_reg, halo_geometry, nx, ny, Nx, Ny)
    else:
        halo = setup_halo(local_frames, local_tx, local_ty, frames_norm,
                          illumination, norm_reg, decomp, nx, ny, Nx, Ny)
    H_cross = halo["H_cross"]

    v = _xp.ones((local_nframes, 1), dtype=_xp.complex64)
    init_sq = mpi_allSum(np.array([float(_xp.real(_xp.sum(_xp.abs(v) ** 2)))], np.float32))
    v /= float(init_sq[0]) ** 0.5

    prev_step = None
    for _ in range(num_iter):
        if H_cross is not None:
            v_halo = exchange_v_halo(v, halo)
            vn = H_local @ v + H_cross @ v_halo
        else:
            vn = H_local @ v

        vn_sq = mpi_allSum(np.array([float(_xp.real(_xp.sum(_xp.abs(vn) ** 2)))], np.float32))
        global_norm = float(vn_sq[0]) ** 0.5
        if global_norm < 1e-12: break
        vn /= global_norm

        step_sq = mpi_allSum(np.array([float(_xp.real(_xp.sum(_xp.abs(vn - v) ** 2)))], np.float32))
        step = float(step_sq[0]) ** 0.5
        v = vn
        if step < 1e-12: break
        if prev_step is not None and step < prev_step:
            rho = step / prev_step
            if rho < 1.0 and step * rho / (1.0 - rho) < SYNC_EIGTOL: break
        prev_step = step

    return v.ravel()


def validate_distributed_sync(maxiter=3, num_iter=5):
    """
    Smoke test for mpi_gramian_sync.

    Runs maxiter AP steps to get non-trivial frames, then calls mpi_gramian_sync
    and checks:
      - omega has the right shape
      - all entries are finite and non-zero
      - |omega| range is reported (entries need not be unit-magnitude; only
        the phases are applied in practice)
    """
    if not _SHARPY:
        if RANK == 0:
            print("[validate-sync] sharpy not importable — run on a GPU node")
        return

    from poster_simulate import simulate
    data, illumination, truth, tx, ty, nframes, nx, ny, Nx, Ny = simulate(16)

    tx_np   = tx.get()   if (cp is not None and isinstance(tx,   cp.ndarray)) else np.array(tx)
    ty_np   = ty.get()   if (cp is not None and isinstance(ty,   cp.ndarray)) else np.array(ty)
    data_np = data.get() if (cp is not None and isinstance(data, cp.ndarray)) else np.array(data)

    decomp        = get_2d_decomposition(tx_np, ty_np)
    mf            = decomp['my_frames']
    local_data_xp = _xp.asarray(data_np[mf])
    local_trans   = (_xp.asarray(tx_np[mf]) + 1j * _xp.asarray(ty_np[mf])).astype(_xp.complex64)
    normalization = _compute_normalization(illumination, local_trans, Nx, Ny)

    # Run AP to get non-trivial frames; F_local holds the projected frames after the loop
    local_nframes = local_data_xp.shape[0]
    F_local     = _xp.zeros((local_nframes, nx, ny), dtype=_xp.complex64)
    frames_norm = _xp.zeros(local_nframes, dtype=_xp.complex64)
    img = _xp.ones((Nx, Ny), dtype=_xp.complex64)
    for ii in range(maxiter):
        img = mpi_AP_step(img, local_data_xp, illumination,
                          local_trans, nx, ny, Nx, Ny, normalization,
                          F_local, frames_norm, decomp, sync=False)

    # Distributed sync — F_local already has the projected frames from the last PASS A.
    # Compare the cached-geometry path (halo_geometry built once, like a real
    # caller syncing repeatedly with frozen positions would) against the old
    # always-rebuild path, to confirm the mpi_halo.py split is a pure
    # optimization with no behaviour change.
    halo_geometry = setup_halo_geometry(local_trans.real, local_trans.imag,
                                        decomp, nx, ny, Nx, Ny)
    omega = mpi_gramian_sync(F_local, illumination, normalization,
                             local_trans, decomp, nx, ny, Nx, Ny, num_iter=num_iter,
                             halo_geometry=halo_geometry)
    omega_old = mpi_gramian_sync(F_local, illumination, normalization,
                                 local_trans, decomp, nx, ny, Nx, Ny, num_iter=num_iter)

    assert omega.shape == (local_nframes,), f"wrong shape: {omega.shape}"
    omega_np = omega.get() if (cp is not None and isinstance(omega, cp.ndarray)) else np.array(omega)
    omega_old_np = omega_old.get() if (cp is not None and isinstance(omega_old, cp.ndarray)) else np.array(omega_old)
    geometry_ok = bool(np.allclose(omega_np, omega_old_np, atol=1e-4))
    ok = bool(np.all(np.isfinite(omega_np))) and float(np.max(np.abs(omega_np))) > 1e-12 and geometry_ok

    if RANK == 0:
        amp = np.abs(omega_np)
        print(f"[validate-sync] ranks={SIZE}  ok={ok}  geometry_ok={geometry_ok}  "
              f"|omega| in [{float(amp.min()):.3f}, {float(amp.max()):.3f}]")
    assert ok, ("mpi_gramian_sync returned invalid omega (NaN, Inf, all-zero), "
               "or cached-geometry path diverged from the always-rebuild path")


# ===========================================================================
# TODO 3 — CUDA-aware MPI + xpack Gatherv for large/irregular object assembly.
# ===========================================================================

def mpi_gather_frames(local_frames, decomp):
    """
    Gather distributed complex64 frames from all ranks onto rank 0.

    Why this exists
    ---------------
    mpi_AP_step keeps the reconstructed *image* replicated on every rank via
    allSum, so no gather is needed for the running loop.  But the *frames*
    (one per scan position) live only on their owning rank.  To write them
    to disk at the end we need them all on rank 0.

    Why not plain MPI_Gather
    ------------------------
    With 2D decomp each rank owns a different number of frames (irregular counts),
    so we need Gatherv.  gatherv() handles this with the MPI.FLOAT chunk-type
    trick that keeps per-rank byte counts under 2^31.

    Why the float32 view
    --------------------
    gatherv() is hardwired to MPI.FLOAT chunks (4 bytes each).  Frames are
    complex64 (8 bytes per element).  Viewing complex64 as float32 doubles the
    last dimension: (local_n, nx, ny) c64 → (local_n, nx, ny*2) f32.
    gatherv sees uniform float32 data; rank 0 casts the result back to complex64.

    CUDA-aware path
    ---------------
    If MPICH_GPU_SUPPORT_ENABLED=1 is set and cray-mpich was loaded with GPU
    support, MPI can read directly from GPU memory.  Otherwise we copy to host
    first (the safe default on Perlmutter without the env var).

    Parameters
    ----------
    local_frames : (local_nframes, nx, ny) complex64 array on device or host
    decomp       : dict from get_2d_decomposition

    Returns
    -------
    frames : (nframes, nx, ny) complex64 numpy array on rank 0, None elsewhere
    """
    is_gpu = cp is not None and isinstance(local_frames, cp.ndarray)
    nx, ny = int(local_frames.shape[1]), int(local_frames.shape[2])

    if is_gpu and _CUDA_AWARE_MPI and SIZE > 1:
        # CUDA-aware path: keep data on device and let cray-mpich DMA it
        # directly from GPU memory.  No host staging — that's the whole point.
        # SIZE > 1 guard keeps the SIZE==1 short-circuit in gatherv on CPU.
        frames_f32 = cp.ascontiguousarray(local_frames).view(cp.float32).reshape(
            local_frames.shape[0], nx, ny * 2)
    else:
        # Host-staged fallback (safe without MPICH_GPU_SUPPORT_ENABLED=1).
        frames_cpu = local_frames.get() if is_gpu else np.asarray(local_frames)
        frames_f32 = np.ascontiguousarray(frames_cpu).view(np.float32).reshape(
            frames_cpu.shape[0], nx, ny * 2)

    # Build a fake chunk_slices from decomp counts for gatherv's displacement calc
    counts   = decomp['counts']
    starts   = np.concatenate([[0], np.cumsum(counts[:-1])]).reshape(SIZE, 1)
    stops    = np.cumsum(counts).reshape(SIZE, 1)
    cs_proxy = np.concatenate([starts, stops], axis=1).astype(np.int64)
    gathered_f32 = gatherv(frames_f32, cs_proxy)

    if RANK == 0:
        nframes_total = int(decomp['nframes'])
        return gathered_f32.reshape(nframes_total, nx, ny * 2).view(np.complex64).reshape(
            nframes_total, nx, ny)
    return None


def validate_distributed_gather(maxiter=3):
    """
    Smoke test for mpi_gather_frames.

    Runs maxiter AP steps, gathers the resulting frames onto rank 0, and checks:
      - shape  == (nframes, nx, ny)
      - all entries finite (no NaN / Inf from the gather itself)
      - max amplitude > 0 (frames are non-trivial after AP)

    Run on a GPU node:
        srun -n 4 python sharpy_mpi_skeleton.py --validate-gather
    """
    if not _SHARPY:
        if RANK == 0:
            print("[validate-gather] sharpy not importable — run on a GPU node")
        return

    from poster_simulate import simulate
    data, illumination, truth, tx, ty, nframes, nx, ny, Nx, Ny = simulate(16)

    tx_np   = tx.get()   if (cp is not None and isinstance(tx,   cp.ndarray)) else np.array(tx)
    ty_np   = ty.get()   if (cp is not None and isinstance(ty,   cp.ndarray)) else np.array(ty)
    data_np = data.get() if (cp is not None and isinstance(data, cp.ndarray)) else np.array(data)

    decomp        = get_2d_decomposition(tx_np, ty_np)
    mf            = decomp['my_frames']
    local_data_xp = _xp.asarray(data_np[mf])
    local_trans   = (_xp.asarray(tx_np[mf]) + 1j * _xp.asarray(ty_np[mf])).astype(_xp.complex64)
    normalization = _compute_normalization(illumination, local_trans, Nx, Ny)

    local_nframes = local_data_xp.shape[0]
    F_local     = _xp.zeros((local_nframes, nx, ny), dtype=_xp.complex64)
    frames_norm = _xp.zeros(local_nframes, dtype=_xp.complex64)
    img = _xp.ones((Nx, Ny), dtype=_xp.complex64)
    for ii in range(maxiter):
        img = mpi_AP_step(img, local_data_xp, illumination,
                          local_trans, nx, ny, Nx, Ny, normalization,
                          F_local, frames_norm, decomp, sync=False)

    # F_local has the projected frames from the last iteration — gather directly
    frames = mpi_gather_frames(F_local, decomp)

    if RANK == 0:
        ok    = (frames.shape == (nframes, nx, ny)
                 and bool(np.all(np.isfinite(frames)))
                 and float(np.max(np.abs(frames))) > 0)
        print(f"[validate-gather] ranks={SIZE}  ok={ok}  "
              f"shape={frames.shape}  max={float(np.max(np.abs(frames))):.2e}")
        assert ok, "mpi_gather_frames produced wrong shape, NaN, or all-zero"


# ===========================================================================
# Coarse inter-rank gauge sync (issue #6 follow-up: MPI confirmation)
# ===========================================================================

def mpi_coarse_gauge_sync(obj_t, cov_t):
    """
    One phase correction per MPI rank via the inter-rank overlap Gramian.

    Each rank contributes wo = (|cov_t| * obj_t).ravel() (coverage-weighted
    object patch). Allgather assembles WO (SIZE x Npix) on every rank.
    G = WO @ WO† (SIZE x SIZE) is the coarse inter-rank Gramian.
    Solving (I - Hn)g = 1 (degree-normalised, ones-anchored) gives the gauge.

    Returns conj(g[RANK]) — multiply local obj_t by this to remove its phase.

    This is the distributed equivalent of coarse_gauge() in od2p_coarse_only_test.py.
    Communication: one Allgather of SIZE x Npix complex64 (one vector per rank).
    """
    if SIZE == 1 or _COMM is None:
        return np.complex64(1.0)

    is_gpu = cp is not None and isinstance(obj_t, cp.ndarray)
    obj_np = obj_t.get() if is_gpu else np.asarray(obj_t)
    cov_np = np.abs(cov_t.get() if is_gpu else np.asarray(cov_t)).astype(np.float32)

    wo_local = (cov_np * obj_np).ravel().astype(np.complex64)
    Npix = wo_local.size

    # Allgather via float32 view (matches mpi_allSum's safe pattern)
    wo_f  = wo_local.view(np.float32)
    WO_f  = np.empty((SIZE, 2 * Npix), dtype=np.float32)
    _COMM.Allgather(wo_f, WO_f)
    WO = WO_f.view(np.complex64).reshape(SIZE, Npix)

    # Inter-rank Gramian in float64 for numerical stability
    WO128 = WO.astype(np.complex128)
    G = WO128 @ WO128.conj().T
    A = G - np.diag(np.diag(G))
    deg = np.maximum(np.abs(A).sum(1), 1e-30)
    s   = 1.0 / np.sqrt(deg)
    Hn  = s[:, None] * A * s[None, :]
    g   = np.linalg.solve(np.eye(SIZE) - Hn + 1e-3 * np.eye(SIZE),
                          np.ones(SIZE, dtype=Hn.dtype))
    g   = g / (np.abs(g) + 1e-30)
    g   = g * np.conj(g.sum()) / (abs(g.sum()) + 1e-30)
    return np.complex64(np.conj(g[RANK]))


def validate_coarse_gauge(n_out=14, k_loc=10):
    """
    Confirm distributed coarse gauge sync across MPI ranks.

    Mirrors od2p_coarse_only_test.py: each rank is one tile. The outer loop
    runs k_loc purely LOCAL AP steps (no AllSum), then mpi_coarse_gauge_sync
    applies one inter-rank phase correction per rank, then coverage-weighted
    AllSum blends the corrected patches into the new global object.

    Expected: NMSE < 0.40 (better than no-sync; od2p single-node coarse ~ 0.02).
    Run:  srun -n 4 <python> sharpy_mpi_skeleton.py --validate-coarse
    """
    if not _SHARPY:
        if RANK == 0:
            print("[validate-coarse] sharpy not importable — run on a GPU node")
        return

    from poster_simulate import simulate
    data, illumination, truth, tx, ty, nframes, nx, ny, Nx, Ny = simulate(16)

    tx_np   = tx.get()   if (cp is not None and isinstance(tx,   cp.ndarray)) else np.array(tx)
    ty_np   = ty.get()   if (cp is not None and isinstance(ty,   cp.ndarray)) else np.array(ty)
    data_np = data.get() if (cp is not None and isinstance(data, cp.ndarray)) else np.array(data)

    decomp        = get_2d_decomposition(tx_np, ty_np)
    mf            = decomp['my_frames']
    local_data_xp = _xp.asarray(data_np[mf])
    local_trans   = (_xp.asarray(tx_np[mf]) + 1j * _xp.asarray(ty_np[mf])).astype(_xp.complex64)

    # Local coverage — each rank's own probe footprint, no AllSum
    cov_t = _xp.zeros((Nx, Ny), dtype=_xp.complex64)
    overlap_cuda(cov_t, 0, local_trans, illumination)
    mx        = float(_xp.max(_xp.abs(cov_t))) + 1e-30
    norm_safe = _xp.where(_xp.abs(cov_t) < 1e-6 * mx, _xp.complex64(1.0), cov_t)

    # Global coverage denominator for blend (AllSum of all ranks' |cov_t|)
    is_gpu   = cp is not None and isinstance(cov_t, cp.ndarray)
    cov_cpu  = np.abs(cov_t.get() if is_gpu else np.asarray(cov_t))
    cov_denom = _xp.asarray(mpi_allSum(cov_cpu.astype(np.float32)))
    cov_denom = _xp.where(cov_denom < 1e-30, _xp.float32(1.0), cov_denom).astype(_xp.complex64)

    local_nframes = local_data_xp.shape[0]
    fb   = _xp.zeros((local_nframes, nx, ny), dtype=_xp.complex64)
    gobj = _xp.ones((Nx, Ny), dtype=_xp.complex64)

    for _outer in range(n_out):
        # Extract frames from the global replicated object
        split_cuda(gobj, fb, local_trans, illumination)

        # k_loc purely local AP steps — no AllSum, tile evolves independently
        obj_t = _xp.zeros((Nx, Ny), dtype=_xp.complex64)
        for _inner in range(k_loc):
            fb, _ = Project_data(fb, local_data_xp, compute_residuals=False)
            obj_t = _xp.zeros((Nx, Ny), dtype=_xp.complex64)
            overlap_cuda(obj_t, fb, local_trans, illumination)
            obj_t = obj_t / norm_safe
            split_cuda(obj_t, fb, local_trans, illumination)

        # One phase correction per rank via inter-rank Gramian
        phase = mpi_coarse_gauge_sync(obj_t, cov_t)
        obj_t = obj_t * complex(phase)

        # Coverage-weighted blend → new global object replicated on all ranks
        blend_cpu = (np.abs(cov_t.get() if is_gpu else np.asarray(cov_t))
                     * (obj_t.get() if is_gpu else np.asarray(obj_t)))
        gobj = _xp.asarray(mpi_allSum(blend_cpu.astype(np.complex64))) / cov_denom

    if RANK == 0:
        truth_np = truth.get() if (cp is not None and isinstance(truth, cp.ndarray)) else np.array(truth)
        gobj_np  = gobj.get()  if (cp is not None and isinstance(gobj,  cp.ndarray)) else np.array(gobj)
        nmse = float(np.linalg.norm(gobj_np - truth_np) / (np.linalg.norm(truth_np) + 1e-30))
        ok   = np.isfinite(nmse) and nmse < 0.5
        print(f"[validate-coarse] ranks={SIZE}  ok={ok}  NMSE={nmse:.4f}  "
              f"(od2p single-node coarse~0.02, no-sync~0.40)")
        assert ok, f"coarse gauge NMSE={nmse:.4f} >= 0.5 — inter-rank sync not helping"


# ===========================================================================
# Cadence sync: cheap coarse gauge every iteration, real pixel halo every
# object_sync_interval iterations (mentor's item 3 -- wires two functions
# that already existed, mpi_object_halo's exchange and mpi_coarse_gauge_sync,
# instead of paying a per-iteration pixel-level sync every time).
# ===========================================================================

def mpi_AP_step_cadence(img, local_data, illumination, local_translations,
                        nx, ny, Nx, Ny, normalization, local_coverage,
                        F_local, frames_norm, object_halo, iteration,
                        object_sync_interval=10, frame_batch=1024, reg=1e-8,
                        eps=None):
    """
    PASS A/B with a CADENCE-based object sync, instead of mpi_AP_step's
    unconditional every-iteration reduction. Mirrors validate_coarse_gauge's
    PROVEN structure (NMSE ~0.02-0.03): object_sync_interval-1 purely local
    iterations -- no cross-rank communication AT ALL -- then one iteration
    where the coarse-gauge phase correction and a real merge happen
    TOGETHER. Gauge alone, applied every iteration with no merge in between,
    measured at NMSE 0.4474 (matching NO sync) in an earlier version of this
    function: a whole-tile phase rotation doesn't reconcile pixel VALUES
    between tiles, so without a periodic merge the "cheap" iterations weren't
    doing anything useful. validate_coarse_gauge's 0.02 result comes from
    gauge and its coverage-weighted mpi_allSum blend happening together, at
    the same cadence -- this reuses that exact pairing, just with
    exchange_object_halo (cheap, O(halo)) standing in for the full blend,
    since that's the actual thing being wired in here.

    img is a PER-RANK LOCAL buffer at all times here (never replicated),
    valid only inside object_halo['tiles_own'] after a sync iteration.

    normalization  : (Nx, Ny) globally-summed coverage, used only on sync
                     iterations (for the post-halo division).
    local_coverage : (Nx, Ny) THIS RANK's own |P|^2 scatter-add (build once,
                     same footprint/static assumptions as object_halo --
                     see mpi_coarse_gauge_sync's obj_t/cov_t convention).
    iteration      : caller's loop counter, 0-based. Syncs on
                     iteration % object_sync_interval == object_sync_interval - 1
                     (i.e. after object_sync_interval-1 purely local steps).
    """
    local_nframes = local_data.shape[0]
    is_sync_iter = (iteration % object_sync_interval == object_sync_interval - 1)

    for s in range(0, local_nframes, frame_batch):
        e = min(local_nframes, s + frame_batch)
        fb = _xp.zeros((e - s, nx, ny), dtype=_xp.complex64)
        split_cuda(img, fb, local_translations[s:e], illumination)
        fb, _ = Project_data(fb, local_data[s:e], compute_residuals=False)
        F_local[s:e] = fb
        frames_norm[s:e] = _xp.linalg.norm(fb, axis=(1, 2)).astype(_xp.complex64)

    img0 = _xp.zeros((Nx, Ny), dtype=_xp.complex64)
    for s in range(0, local_nframes, frame_batch):
        e = min(local_nframes, s + frame_batch)
        overlap_cuda(img0, F_local[s:e], local_translations[s:e], illumination)

    local_eps = reg * float(_xp.max(_xp.abs(local_coverage)).real)

    if not is_sync_iter:
        # Purely local -- no cross-rank communication at all, matches
        # validate_coarse_gauge's inner k_loc loop exactly.
        return img0 / (local_coverage + local_eps)

    # Sync boundary: gauge computed on the LOCALLY-normalized object (matching
    # validate_coarse_gauge, which calls mpi_coarse_gauge_sync on obj_t
    # *after* dividing by local coverage, not on the raw accumulation).
    obj_local = img0 / (local_coverage + local_eps)
    phase = mpi_coarse_gauge_sync(obj_local, local_coverage)

    # Apply phase to the RAW (pre-division) accumulation -- multiplying by a
    # constant phase commutes with the later division, but the cross-rank
    # SUM (via halo exchange, next) needs phase-consistent inputs, which only
    # the raw accumulation form supports (dividing first, by DIFFERENT
    # per-rank coverage, then summing != summing then dividing).
    img0 = img0 * complex(phase)

    if eps is None:
        eps = reg * float(_xp.max(_xp.abs(normalization)).real)
    from mpi_object_halo import exchange_object_halo
    exchange_object_halo(img0, object_halo)
    for (x0, x1, y0, y1) in object_halo['tiles_own']:
        img0[y0:y1, x0:x1] /= (normalization[y0:y1, x0:x1] + eps)
    return img0


def validate_cadence_sync(n_iter=60, object_sync_interval=10, nnx=16):
    """
    Correctness/quality gate for mpi_AP_step_cadence -- NOT just a
    performance check. Compares final NMSE against the two baselines the
    mentor cited from od2p_coarse_only_test: coarse-gauge-only reaches
    ~0.02-0.032, no inter-tile sync at all sits at ~0.42-0.53. This cadence
    scheme should land closer to the coarse-gauge number than to full
    per-iteration pixel sync would, while doing far less communication.

    Run: srun -n 4 <python> sharpy_mpi_skeleton.py --validate-cadence
    """
    if not _SHARPY:
        if RANK == 0:
            print("[validate-cadence] sharpy not importable — run on a GPU node")
        return

    from poster_simulate import simulate
    data, illumination, truth, tx, ty, nframes, nx, ny, Nx, Ny = simulate(nnx)

    tx_np   = tx.get()   if (cp is not None and isinstance(tx,   cp.ndarray)) else np.array(tx)
    ty_np   = ty.get()   if (cp is not None and isinstance(ty,   cp.ndarray)) else np.array(ty)
    data_np = data.get() if (cp is not None and isinstance(data, cp.ndarray)) else np.array(data)
    truth_np = truth.get() if (cp is not None and isinstance(truth, cp.ndarray)) else np.array(truth)

    decomp        = get_2d_decomposition(tx_np, ty_np)
    mf            = decomp['my_frames']
    local_data_xp = _xp.asarray(data_np[mf])
    local_trans   = (_xp.asarray(tx_np[mf]) + 1j * _xp.asarray(ty_np[mf])).astype(_xp.complex64)
    normalization = _compute_normalization(illumination, local_trans, Nx, Ny)

    local_coverage = _xp.zeros((Nx, Ny), dtype=_xp.complex64)
    overlap_cuda(local_coverage, 0, local_trans, illumination)

    from mpi_object_halo import setup_object_halo
    object_halo = setup_object_halo(local_trans, nx, ny, Nx, Ny)

    local_nframes = local_data_xp.shape[0]
    F_local     = _xp.zeros((local_nframes, nx, ny), dtype=_xp.complex64)
    frames_norm = _xp.zeros(local_nframes, dtype=_xp.complex64)
    img         = _xp.ones((Nx, Ny), dtype=_xp.complex64)

    def _nmse_checkpoint(img_now, tag):
        is_gpu = cp is not None and isinstance(img_now, cp.ndarray)
        img_cpu = img_now.get() if is_gpu else np.asarray(img_now)
        tile_mask = np.zeros((Nx, Ny), dtype=bool)
        for (x0, x1, y0, y1) in object_halo['tiles_own']:
            tile_mask[y0:y1, x0:x1] = True
        local_tile_cpu = np.where(tile_mask, img_cpu, 0).astype(np.complex64)
        gobj = mpi_allSum(local_tile_cpu)
        if RANK == 0:
            scale = (np.dot(gobj.ravel().conj(), truth_np.ravel())
                    / (np.dot(gobj.ravel().conj(), gobj.ravel()).real + 1e-30))
            aligned = gobj * scale
            nmse = float(np.linalg.norm(aligned - truth_np) / (np.linalg.norm(truth_np) + 1e-30))
            print(f"  [cadence checkpoint] {tag}  NMSE={nmse:.4f}", flush=True)
        return gobj

    for ii in range(n_iter):
        img = mpi_AP_step_cadence(img, local_data_xp, illumination, local_trans,
                                  nx, ny, Nx, Ny, normalization, local_coverage,
                                  F_local, frames_norm, object_halo, ii,
                                  object_sync_interval=object_sync_interval)
        if ii % object_sync_interval == object_sync_interval - 1:
            _nmse_checkpoint(img, f"iter={ii+1}")

    gobj = _nmse_checkpoint(img, "final")

    if RANK == 0:
        scale = (np.dot(gobj.ravel().conj(), truth_np.ravel())
                / (np.dot(gobj.ravel().conj(), gobj.ravel()).real + 1e-30))
        aligned = gobj * scale
        nmse = float(np.linalg.norm(aligned - truth_np) / (np.linalg.norm(truth_np) + 1e-30))
        ok = np.isfinite(nmse) and nmse < 0.15
        print(f"[validate-cadence] ranks={SIZE}  object_sync_interval={object_sync_interval}  "
              f"n_iter={n_iter}  ok={ok}  NMSE={nmse:.4f}  "
              f"(coarse-gauge-only ref ~0.02-0.03, no-sync ref ~0.42-0.53)")
        assert ok, f"cadence sync NMSE={nmse:.4f} — not close to the coarse-gauge baseline"


# ===========================================================================
# Scaling benchmarks — Workstream B/C2 deliverables
# ===========================================================================
import time as _time


def _cuda_sync():
    if cp is not None:
        cp.cuda.runtime.deviceSynchronize()


def mpi_AP_step_timed(img, local_data, illumination, local_translations,
                      nx, ny, Nx, Ny, normalization, F_local, frames_norm,
                      frame_batch=1024, reg=1e-8, object_halo=None, eps=None):
    """
    One distributed AP iteration (no Gramian sync).
    Returns (img_updated, t_compute, t_sync) where:
      t_compute = PASS A + PASS B GPU wall time (local per-rank work)
      t_sync    = GPU->CPU staging + MPI AllReduce/halo-exchange + CPU->GPU staging
    CUDA-syncs before each timing boundary for accurate GPU measurements.

    object_halo : dict from mpi_object_halo.setup_object_halo(...), or None.
                  When given, PASS B sums only the overlap strips between
                  ranks instead of mpi_allSum-ing the whole Nx x Ny canvas --
                  see mpi_object_halo.py. img is then only correct inside
                  object_halo['tiles_own'] on each rank.
    eps         : pre-computed reg * max(abs(normalization)), or None to
                  compute it here. normalization is static across iterations,
                  so callers running a loop should compute this once outside
                  it instead of forcing a full-canvas device sync every call
                  (previously this ran after t_sync stopped, so it wasn't
                  even part of either timed bucket -- still real wasted
                  wall-clock time between iterations, just unattributed).
    """
    local_nframes = local_data.shape[0]

    _cuda_sync()
    t0 = _time.perf_counter()

    # PASS A: split + FFT projection
    for s in range(0, local_nframes, frame_batch):
        e = min(local_nframes, s + frame_batch)
        fb = _xp.zeros((e - s, nx, ny), dtype=_xp.complex64)
        split_cuda(img, fb, local_translations[s:e], illumination)
        fb, _ = Project_data(fb, local_data[s:e], compute_residuals=False)
        F_local[s:e] = fb
        frames_norm[s:e] = _xp.linalg.norm(fb, axis=(1, 2)).astype(_xp.complex64)

    # PASS B: overlap accumulation
    img0 = _xp.zeros((Nx, Ny), dtype=_xp.complex64)
    for s in range(0, local_nframes, frame_batch):
        e = min(local_nframes, s + frame_batch)
        overlap_cuda(img0, F_local[s:e], local_translations[s:e], illumination)

    _cuda_sync()
    t_compute = _time.perf_counter() - t0

    # Sync: GPU->CPU + AllReduce/halo-exchange collective + CPU->GPU
    t0 = _time.perf_counter()
    if object_halo is not None:
        from mpi_object_halo import exchange_object_halo
        exchange_object_halo(img0, object_halo)
        img_global = img0
    else:
        is_gpu = cp is not None and isinstance(img0, cp.ndarray)
        img0_cpu = img0.get() if is_gpu else img0
        img_global_cpu = mpi_allSum(img0_cpu)
        img_global = _xp.asarray(img_global_cpu) if is_gpu else img_global_cpu
    t_sync = _time.perf_counter() - t0

    if eps is None:
        eps = reg * float(_xp.max(_xp.abs(normalization)).real)

    if object_halo is not None:
        # Only tiles_own is meaningful after the exchange -- dividing the
        # full (Nx,Ny) canvas would touch pixels nobody reads.
        for (x0, x1, y0, y1) in object_halo['tiles_own']:
            img_global[y0:y1, x0:x1] /= (normalization[y0:y1, x0:x1] + eps)
        return img_global, t_compute, t_sync
    return img_global / (normalization + eps), t_compute, t_sync


_STRONG_NNX = 64    # 4096 frames — Stefano guide §3

_WEAK_NNX = {       # growing-object ladder — Stefano guide §4
    1:  45,         # 2025 frames
    2:  64,         # 4096 frames  (~2048/rank)
    4:  90,         # 8100 frames  (~2025/rank)
    8: 128,         # 16384 frames (~2048/rank)
}


def _run_ap_benchmark(NNX, nwarmup=5, ntimed=20, tag=""):
    """Simulate NNX*NNX frames, 2D decomp, run AP with compute/sync timing."""
    if not _SHARPY:
        if RANK == 0:
            print(f"[bench{tag}] sharpy not available — run on GPU node")
        return None

    from poster_simulate import simulate

    if RANK == 0:
        print(f"[bench{tag}] ranks={SIZE}  NNX={NNX}  nframes={NNX*NNX}", flush=True)

    data, illumination, truth, tx, ty, nframes, nx, ny, Nx, Ny = simulate(NNX)

    tx_np   = tx.get()   if (cp is not None and isinstance(tx,   cp.ndarray)) else np.array(tx)
    ty_np   = ty.get()   if (cp is not None and isinstance(ty,   cp.ndarray)) else np.array(ty)
    data_np = data.get() if (cp is not None and isinstance(data, cp.ndarray)) else np.array(data)

    decomp        = get_2d_decomposition(tx_np, ty_np)
    mf            = decomp['my_frames']
    local_data_xp = _xp.asarray(data_np[mf])
    local_trans   = (_xp.asarray(tx_np[mf]) + 1j * _xp.asarray(ty_np[mf])).astype(_xp.complex64)

    normalization = _compute_normalization(illumination, local_trans, Nx, Ny)
    # Static across iterations -- compute once instead of every call.
    eps = 1e-8 * float(_xp.max(_xp.abs(normalization)).real)

    # SHARPY_NO_HALO=1 forces the old full-canvas AllReduce path, for
    # apples-to-apples before/after scaling comparisons.
    object_halo = None
    if _os.environ.get("SHARPY_NO_HALO", "0") != "1":
        from mpi_object_halo import setup_object_halo
        object_halo = setup_object_halo(local_trans, nx, ny, Nx, Ny)

    local_nframes = local_data_xp.shape[0]
    F_local     = _xp.zeros((local_nframes, nx, ny), dtype=_xp.complex64)
    frames_norm = _xp.zeros(local_nframes, dtype=_xp.complex64)
    img         = _xp.ones((Nx, Ny), dtype=_xp.complex64)

    # Warmup
    for _ in range(nwarmup):
        img, _, _ = mpi_AP_step_timed(img, local_data_xp, illumination, local_trans,
                                      nx, ny, Nx, Ny, normalization, F_local, frames_norm,
                                      object_halo=object_halo, eps=eps)
    mpi_barrier()

    # Timed iterations — barrier before each so all ranks start together
    times_compute = []
    times_sync    = []
    for _ in range(ntimed):
        mpi_barrier()
        img, tc, ts = mpi_AP_step_timed(img, local_data_xp, illumination, local_trans,
                                        nx, ny, Nx, Ny, normalization, F_local, frames_norm,
                                        object_halo=object_halo, eps=eps)
        times_compute.append(tc)
        times_sync.append(ts)
    mpi_barrier()

    if RANK == 0:
        tc = float(np.median(times_compute))
        ts = float(np.median(times_sync))
        print(f"  frames/rank={local_nframes}  Nx={Nx}  Ny={Ny}")
        print(f"  compute={tc*1000:.1f}ms  sync={ts*1000:.1f}ms  "
              f"total={(tc+ts)*1000:.1f}ms/iter", flush=True)
        return dict(ranks=SIZE, NNX=NNX, nframes=NNX*NNX, Nx=Nx, Ny=Ny,
                    local_nframes=local_nframes, tc=tc, ts=ts, total=tc+ts)
    return None


def _run_ap_benchmark_h5(h5path, nwarmup=5, ntimed=20, tag=""):
    """Like _run_ap_benchmark but loads real data from an H5 file."""
    if not _SHARPY:
        if RANK == 0:
            print(f"[bench{tag}] sharpy not available — run on GPU node")
        return None

    # ONLY rank 0 touches the filesystem — avoids Lustre MDS stalls on other nodes.
    # A bcast of a found-flag syncs everyone before any MPI collective is attempted.
    data_np = illum_np = tx_np = ty_np = None
    if RANK == 0:
        import h5py, os
        if os.path.exists(h5path):
            with h5py.File(h5path, "r") as fid:
                data_np  = np.array(fid["data"],           dtype=np.float32)
                illum_np = np.array(fid["probe"],          dtype=np.complex64)
                tx_np    = np.array(fid["translations_x"], dtype=np.float32).reshape(-1)
                ty_np    = np.array(fid["translations_y"], dtype=np.float32).reshape(-1)
        else:
            print(f"[bench{tag}] {h5path} not found — falling back to simulate", flush=True)

    found = mpi_bcast(data_np is not None)   # small bool, pickle fine
    if not found:
        return None

    data_np  = mpi_bcast_array(data_np)
    illum_np = mpi_bcast_array(illum_np)
    tx_np    = mpi_bcast_array(tx_np)
    ty_np    = mpi_bcast_array(ty_np)

    nframes, nx, ny = data_np.shape
    Nx = int(tx_np.max() - tx_np.min()) + nx
    Ny = int(ty_np.max() - ty_np.min()) + ny

    if RANK == 0:
        print(f"[bench{tag}] ranks={SIZE}  nframes={nframes}  Nx={Nx}  Ny={Ny}", flush=True)

    illumination  = _xp.asarray(illum_np)
    decomp        = get_2d_decomposition(tx_np, ty_np)
    mf            = decomp['my_frames']
    local_data_xp = _xp.asarray(data_np[mf])
    local_trans   = (_xp.asarray(tx_np[mf]) + 1j * _xp.asarray(ty_np[mf])).astype(_xp.complex64)

    normalization = _compute_normalization(illumination, local_trans, Nx, Ny)
    # Static across iterations -- compute once instead of every call.
    eps = 1e-8 * float(_xp.max(_xp.abs(normalization)).real)

    # SHARPY_NO_HALO=1 forces the old full-canvas AllReduce path, for
    # apples-to-apples before/after scaling comparisons.
    object_halo = None
    if _os.environ.get("SHARPY_NO_HALO", "0") != "1":
        from mpi_object_halo import setup_object_halo
        object_halo = setup_object_halo(local_trans, nx, ny, Nx, Ny)

    local_nframes = local_data_xp.shape[0]
    F_local     = _xp.zeros((local_nframes, nx, ny), dtype=_xp.complex64)
    frames_norm = _xp.zeros(local_nframes, dtype=_xp.complex64)
    img         = _xp.ones((Nx, Ny), dtype=_xp.complex64)

    for _ in range(nwarmup):
        img, _, _ = mpi_AP_step_timed(img, local_data_xp, illumination, local_trans,
                                      nx, ny, Nx, Ny, normalization, F_local, frames_norm,
                                      object_halo=object_halo, eps=eps)
    mpi_barrier()

    times_compute = []
    times_sync    = []
    for _ in range(ntimed):
        mpi_barrier()
        img, tc, ts = mpi_AP_step_timed(img, local_data_xp, illumination, local_trans,
                                        nx, ny, Nx, Ny, normalization, F_local, frames_norm,
                                        object_halo=object_halo, eps=eps)
        times_compute.append(tc)
        times_sync.append(ts)
    mpi_barrier()

    if RANK == 0:
        tc = float(np.median(times_compute))
        ts = float(np.median(times_sync))
        print(f"  frames/rank={local_nframes}")
        print(f"  compute={tc*1000:.1f}ms  sync={ts*1000:.1f}ms  "
              f"total={(tc+ts)*1000:.1f}ms/iter", flush=True)
        return dict(ranks=SIZE, nframes=nframes, Nx=Nx, Ny=Ny,
                    local_nframes=local_nframes, tc=tc, ts=ts, total=tc+ts)
    return {}   # non-None: signals H5 benchmark completed (non-rank-0 sentinel)


_STRONG_H5 = "/sdf/home/d/dnyanhet/sharpy_fresh/sharpy/refine_illum_large.h5"


def benchmark_mpi_strong(nwarmup=5, ntimed=50):
    """Strong scaling: fixed real dataset, vary ranks."""
    result = _run_ap_benchmark_h5(_STRONG_H5, nwarmup=nwarmup, ntimed=ntimed, tag="-strong")
    if result is None:
        # H5 unavailable — fall back to synthetic 4096-frame data
        _run_ap_benchmark(_STRONG_NNX, nwarmup=nwarmup, ntimed=ntimed, tag="-strong")


_WEAK_H5 = {    # growing-object ladder — guide §4, generated with NX=128 DX=40
    1: "/sdf/home/d/dnyanhet/refine_45.h5",   # 2025 frames, 1908x1908
    2: "/sdf/home/d/dnyanhet/refine_64.h5",   # 4096 frames, 2668x2668
    4: "/sdf/home/d/dnyanhet/refine_90.h5",   # 8100 frames, 3708x3708
    8: "/sdf/home/d/dnyanhet/refine_128.h5",  # 16384 frames, 5228x5228
}


def benchmark_mpi_weak(nwarmup=5, ntimed=50):
    """Weak scaling: growing-object H5 ladder per Stefano guide §4."""
    h5path = _WEAK_H5.get(SIZE)
    if h5path is None:
        if RANK == 0:
            print(f"[bench-weak] no H5 entry for SIZE={SIZE}")
        return
    result = _run_ap_benchmark_h5(h5path, nwarmup=nwarmup, ntimed=ntimed, tag="-weak")
    if result is None:
        NNX = _WEAK_NNX.get(SIZE, _STRONG_NNX)
        _run_ap_benchmark(NNX, nwarmup=nwarmup, ntimed=ntimed, tag="-weak")


if __name__ == "__main__":
    import sys
    if "--validate-ap" in sys.argv:
        validate_distributed_ap()
    elif "--validate-sync" in sys.argv:
        validate_distributed_sync()
    elif "--validate-gather" in sys.argv:
        validate_distributed_gather()
    elif "--validate-coarse" in sys.argv:
        validate_coarse_gauge()
    elif "--validate-cadence" in sys.argv:
        validate_cadence_sync()
    elif "--benchmark-strong" in sys.argv:
        benchmark_mpi_strong()
    elif "--benchmark-weak" in sys.argv:
        benchmark_mpi_weak()
    else:
        smoke_test()
    mpi_barrier()
