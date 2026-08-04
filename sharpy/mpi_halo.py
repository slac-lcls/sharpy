# -*- coding: utf-8 -*-
"""
mpi_halo.py — halo exchange helpers for distributed Gramian sync (TODO 2).

Two public functions:
  setup_halo(...)       — one-time setup: detect cross-rank pairs, exchange
                          boundary frames, build H_cross.
  exchange_v_halo(...)  — called each power-iteration step to exchange the
                          eigenvector entries for halo frames.

Halo strategy (scales to 256x256+):
  Only the boundary frames that spatially overlap across a rank boundary are
  exchanged.  For a typical ptychography scan the probe footprint overlaps
  ~5-10 neighboring scan positions, so the halo is O(n_boundary) << O(N_total).
"""
import numpy as np

try:
    from mpi4py import MPI
    _COMM = MPI.COMM_WORLD
    RANK  = _COMM.Get_rank()
    SIZE  = _COMM.Get_size()
except Exception:
    MPI = None; _COMM = None; RANK = 0; SIZE = 1

try:
    import cupy as _cp
    _xp = _cp
    _GPU = True
except Exception:
    _cp = None
    _xp = np
    _GPU = False


def _to_np(a, dtype):
    return (a.get() if (_GPU and isinstance(a, _cp.ndarray)) else np.asarray(a)).astype(dtype)


def setup_halo_geometry(local_tx_xp, local_ty_xp, decomp, nx, ny, Nx, Ny):
    """
    Positions-only part of halo setup (steps 1-3 of the original setup_halo).
    Call once, before the AP loop; rebuild only if positions move (e.g. inside
    position refinement) -- everything here depends only on WHERE frames are,
    not on their current content, so it's safe to reuse across every sync call
    while positions are frozen.

    Steps
    -----
    1. Allgather translations → global_tx/global_ty indexed by global frame id.
    2. Build global Gramiam_plan → find all pairs including cross-rank.
    3. XOR-filter to boundary pairs (exactly one frame is local).

    Parameters
    ----------
    local_tx_xp    : (local_nframes,) float32 x-translations on device
    local_ty_xp    : (local_nframes,) float32 y-translations on device
    decomp         : dict from get_2d_decomposition (my_frames, frame_to_rank,
                     counts, global_to_local, nframes)
    nx, ny         : frame dimensions
    Nx, Ny         : image dimensions

    Returns
    -------
    geometry : dict with keys
        halo_global  — sorted int64 array of global halo frame indices
        send_idx     — dict {q: sorted list of LOCAL indices to send to rank q}
        recv_idx     — dict {q: sorted list of GLOBAL indices from rank q}
        halo_map     — dict {global_idx: position in halo array}
        global_tx/global_ty — (nframes_total,) float32, every frame's position
                     (needed by exchange_halo_frames to build the extended plan)
    """
    from Operators import Gramiam_plan

    my_frames      = decomp['my_frames']
    frame_to_rank  = decomp['frame_to_rank']
    global_to_local = decomp['global_to_local']
    counts         = decomp['counts'].tolist()
    nframes_total  = int(decomp['nframes'])

    # 1. Allgather translations, then reorder to global frame index.
    # With 2D decomp, allgatherv fills in rank order (rank0 frames, rank1 frames, ...).
    # We reorder so that global_tx[g] = translation of global frame g.
    all_tx_raw = np.empty(nframes_total, np.float32)
    all_ty_raw = np.empty(nframes_total, np.float32)
    all_idx    = np.empty(nframes_total, np.int64)
    _COMM.Allgatherv(my_frames.astype(np.int64), (all_idx, counts))
    _COMM.Allgatherv(_to_np(local_tx_xp, np.float32).ravel(), (all_tx_raw, counts))
    _COMM.Allgatherv(_to_np(local_ty_xp, np.float32).ravel(), (all_ty_raw, counts))
    global_tx = np.empty(nframes_total, np.float32)
    global_ty = np.empty(nframes_total, np.float32)
    global_tx[all_idx] = all_tx_raw
    global_ty[all_idx] = all_ty_raw

    # 2. Global plan: finds all pairs including cross-rank
    gplan = Gramiam_plan(_xp.asarray(global_tx), _xp.asarray(global_ty),
                         nframes_total, nx, ny, Nx, Ny)
    col_g = np.asarray(gplan["col"].get() if _GPU else gplan["col"], int)
    row_g = np.asarray(gplan["row"].get() if _GPU else gplan["row"], int)

    # 3. XOR filter: exactly one of (col, row) belongs to this rank
    my_set = set(my_frames.tolist())
    i_loc  = np.array([int(c) in my_set for c in col_g])
    j_loc  = np.array([int(r) in my_set for r in row_g])
    bdy    = i_loc ^ j_loc

    null_geometry = dict(halo_global=np.array([], np.int64),
                         send_idx={}, recv_idx={}, halo_map={},
                         global_tx=global_tx, global_ty=global_ty)
    if not np.any(bdy):
        return null_geometry

    bp_mine   = np.where(i_loc[bdy], col_g[bdy], row_g[bdy])
    bp_theirs = np.where(i_loc[bdy], row_g[bdy], col_g[bdy])

    def rank_of(g):
        return int(frame_to_rank[int(g)])

    send_sets = {q: set() for q in range(SIZE)}
    recv_sets = {q: set() for q in range(SIZE)}
    for gm, gt in zip(bp_mine, bp_theirs):
        q = rank_of(gt)
        send_sets[q].add(global_to_local[int(gm)])   # local index, not gm-s
        recv_sets[q].add(int(gt))

    send_idx = {q: sorted(v) for q, v in send_sets.items()}
    recv_idx = {q: sorted(v) for q, v in recv_sets.items()}
    halo_global = np.array(sorted(set().union(*recv_sets.values())), np.int64)
    halo_map = {int(g): k for k, g in enumerate(halo_global)}

    return dict(halo_global=halo_global, send_idx=send_idx, recv_idx=recv_idx,
                halo_map=halo_map, global_tx=global_tx, global_ty=global_ty)


def exchange_halo_frames(local_frames, local_tx_xp, local_ty_xp, frames_norm,
                         illumination, norm_reg, geometry, nx, ny, Nx, Ny):
    """
    Content-dependent part of halo setup (steps 4-5 of the original
    setup_halo) -- call every sync, using a geometry built once by
    setup_halo_geometry(). Exchanges the CURRENT frame data for the
    (already-known) boundary pairs and rebuilds H_cross from it, since frame
    content (unlike position) genuinely changes every AP iteration.

    Steps
    -----
    4. Isend/Irecv only those boundary frames.
    5. Build extended plan (local + halo) → extract H_cross dense block.

    Parameters
    ----------
    local_frames   : (local_nframes, nx, ny) complex64, this rank's CURRENT frames
    local_tx_xp    : (local_nframes,) float32 x-translations on device
    local_ty_xp    : (local_nframes,) float32 y-translations on device
    frames_norm    : (local_nframes,) complex64 per-frame L2 norms
    illumination   : (nx, ny) complex64 probe
    norm_reg       : (Nx, Ny) normalization + eps  (NaN-safe denominator)
    geometry       : dict from setup_halo_geometry(...)
    nx, ny         : frame dimensions
    Nx, Ny         : image dimensions

    Returns
    -------
    halo_state : dict with keys (same shape as the old setup_halo(), so
                 exchange_v_halo works unchanged)
        H_cross      — (local_nframes x n_halo) dense array on device, or None
        halo_global  — sorted int64 array of global halo frame indices
        send_idx     — dict {q: sorted list of LOCAL indices to send to rank q}
        recv_idx     — dict {q: sorted list of GLOBAL indices from rank q}
        halo_map     — dict {global_idx: position in halo array}
    """
    from Operators import Gramiam_plan, Gramiam_calc_cuda

    halo_global = geometry['halo_global']
    send_idx    = geometry['send_idx']
    recv_idx    = geometry['recv_idx']
    halo_map    = geometry['halo_map']
    global_tx   = geometry['global_tx']
    global_ty   = geometry['global_ty']
    n_halo      = len(halo_global)
    local_nframes = local_frames.shape[0]

    null_state = dict(H_cross=None, halo_global=halo_global,
                      send_idx=send_idx, recv_idx=recv_idx, halo_map=halo_map)
    if n_halo == 0:
        return null_state

    # 4. Exchange boundary frames via Isend/Irecv
    local_cpu = _to_np(local_frames, np.complex64)
    halo_cpu  = np.empty((n_halo, nx, ny), np.complex64)
    sbufs, reqs, rbufs = [], [], {}

    for q in range(SIZE):
        if q == RANK or not recv_idx[q]: continue
        buf = np.empty(len(recv_idx[q]) * nx * ny, np.complex64)
        reqs.append(_COMM.Irecv(buf, source=q, tag=q * SIZE + RANK))
        rbufs[q] = (buf, recv_idx[q])

    for q in range(SIZE):
        if q == RANK or not send_idx[q]: continue
        sb = np.ascontiguousarray(local_cpu[send_idx[q]].ravel())
        sbufs.append(sb)
        reqs.append(_COMM.Isend(sb, dest=q, tag=RANK * SIZE + q))

    MPI.Request.Waitall(reqs)

    for q, (buf, gidxs) in rbufs.items():
        fq = buf.reshape(len(gidxs), nx, ny)
        for k, g in enumerate(gidxs):
            halo_cpu[halo_map[g]] = fq[k]

    halo_frames = _xp.asarray(halo_cpu)
    halo_norm   = _xp.linalg.norm(halo_frames, axis=(1, 2)).astype(_xp.complex64)

    # 5. Extended plan (local + halo) → H_cross
    # local_tx_xp may be (N,1,1) from make_translations; ravel to 1D before concat.
    ext_tx = _xp.asarray(np.concatenate([_to_np(local_tx_xp, np.float32).ravel(),
                                         global_tx[halo_global].ravel().astype(np.float32)]))
    ext_ty = _xp.asarray(np.concatenate([_to_np(local_ty_xp, np.float32).ravel(),
                                         global_ty[halo_global].ravel().astype(np.float32)]))
    ext_f    = _xp.concatenate([local_frames, halo_frames])
    ext_norm = _xp.concatenate([frames_norm, halo_norm])
    ext_plan = Gramiam_plan(ext_tx, ext_ty, local_nframes + n_halo, nx, ny, Nx, Ny)
    ext_H    = Gramiam_calc_cuda(ext_f, ext_plan, illumination, norm_reg, ext_norm)

    # Cross-block: local rows × halo cols (dense; n_halo is small at scale)
    # A small fraction of entries may be NaN from boundary pixel edge cases in
    # the zQQz kernel (same root cause as H_local before the eps fix, but
    # harder to fully prevent for cross-rank shifts).
    #Zero them out — they represent pairs with negligible physical overlap anyway

    H_cross = ext_H.toarray()[:local_nframes, local_nframes:]
    H_cross = _xp.nan_to_num(H_cross, nan=0.0, posinf=0.0, neginf=0.0)

    return dict(H_cross=H_cross, halo_global=halo_global,
                send_idx=send_idx, recv_idx=recv_idx, halo_map=halo_map)


def setup_halo(local_frames, local_tx_xp, local_ty_xp, frames_norm,
               illumination, norm_reg, decomp, nx, ny, Nx, Ny):
    """
    Backward-compatible one-shot wrapper: setup_halo_geometry() +
    exchange_halo_frames() in one call. Prefer calling them separately when
    syncing repeatedly with frozen positions -- see their docstrings.
    """
    geometry = setup_halo_geometry(local_tx_xp, local_ty_xp, decomp, nx, ny, Nx, Ny)
    return exchange_halo_frames(local_frames, local_tx_xp, local_ty_xp, frames_norm,
                                illumination, norm_reg, geometry, nx, ny, Nx, Ny)


def exchange_v_halo(v, halo_state):
    """
    Exchange eigenvector entries for halo frames (called each power-iteration step).

    Parameters
    ----------
    v          : (local_nframes, 1) complex64 on device — this rank's v slice
    halo_state : dict returned by setup_halo

    Returns
    -------
    v_halo : (n_halo, 1) complex64 on device, or None if no halo
    """
    halo_global = halo_state["halo_global"]
    n_halo = len(halo_global)
    if n_halo == 0:
        return None

    send_idx = halo_state["send_idx"]
    recv_idx = halo_state["recv_idx"]
    halo_map = halo_state["halo_map"]

    v_cpu    = _to_np(v, np.complex64).ravel()
    v_halo   = np.empty(n_halo, np.complex64)
    sbufs, reqs, rbufs = [], [], {}

    for q in range(SIZE):
        if q == RANK or not recv_idx.get(q): continue
        buf = np.empty(len(recv_idx[q]), np.complex64)
        reqs.append(_COMM.Irecv(buf, source=q, tag=q * SIZE + RANK + 10000))
        rbufs[q] = (buf, recv_idx[q])

    for q in range(SIZE):
        if q == RANK or not send_idx.get(q): continue
        sb = np.ascontiguousarray(v_cpu[send_idx[q]])
        sbufs.append(sb)
        reqs.append(_COMM.Isend(sb, dest=q, tag=RANK * SIZE + q + 10000))

    MPI.Request.Waitall(reqs)

    for q, (buf, gidxs) in rbufs.items():
        for k, g in enumerate(gidxs):
            v_halo[halo_map[g]] = buf[k]

    return _xp.asarray(v_halo.reshape(-1, 1))
