# -*- coding: utf-8 -*-
"""
mpi_object_halo.py — halo exchange for the distributed object accumulation
(PASS B of mpi_AP_step). The object twin of mpi_halo.py.

Two public functions:
  setup_object_halo(local_trans, nx, ny, Nx, Ny)  — one-time setup: find this
                                  rank's footprint rect(s), Allgather them,
                                  find which other ranks' rects overlap them.
  exchange_object_halo(img0, object_halo)         — sum only the overlap
                                  strips across ranks, in place.

The problem
-----------
Each rank's frames land inside a contiguous tile of the canvas (2D spatial
decomposition), so overlap_cuda only ever *writes* nonzero values inside that
rank's footprint (the tight region containing every frame footprint the rank
owns). mpi_allSum(img0) reduces the FULL Nx x Ny canvas every iteration --
O(Nx*Ny), independent of rank count. But outside the union of per-rank
footprints the canvas is zero on every rank, and inside a single rank's
footprint (away from where footprints overlap) only that rank has a nonzero
value. Only the thin strips where two ranks' footprints overlap need summing
across ranks.

KEY IDENTITY
------------
For any pixel p:
    mpi_allSum(img0)[p] = sum over ALL ranks r of img0_r[p]
Every rank r has img0_r[p] == 0 unless p is inside footprint(r) (overlap_cuda
never writes outside a rank's own frame footprints). So the sum above only
has nonzero terms for ranks r whose footprint contains p -- i.e. ranks whose
footprint intersects footprint(RANK) at p (RANK's own term) union every OTHER
rank q whose footprint also covers p. Summing footprint(RANK) against every q
with footprint(RANK) ∩ footprint(q) != empty, restricted to that intersection
rectangle, reproduces exactly that sum -- pixels covered by more than two
ranks (e.g. a 4-tile corner) get one term added per overlapping neighbour,
same as the AllReduce would add one term per rank. This is exact, not an
approximation -- provided each pairwise exchange sends the RAW (pre-halo)
contribution (see "snapshot-then-add" below).

Periodic wraparound
--------------------
overlap.cu / split.cu index with `% img_width` / `% img_height`, so a frame
whose footprint would extend past the Ny/Nx edge wraps around and writes at
the START of that axis instead. A footprint bounding box that just clips at
the canvas edge (instead of wrapping) silently drops that contribution --
found by validate_object_halo.py on real GPU/MPI runs (poster_simulate sizes
the canvas tightly, so frames near the max translation sit right at the
edge and wrap). _footprint_rects splits a rank's footprint into up to 2 rects
per axis (main span + wrapped remainder) when this happens, so a rank's full
footprint is represented as 1-4 disjoint rects, all pairwise-intersected
against every other rank's rects.

Four things a naive port breaks (see README):
  1. Translations are complex here (tx + 1j*ty), not (N,2). _footprint_rects
     reads .real/.imag.  .real pairs with the Ny (column/width) axis, .imag
     with the Nx (row/height) axis -- matches overlap.cu / split.cu indexing.
  2. The object stops being replicated. After exchange_object_halo, a rank
     is only correct inside its OWN footprint (tiles_own) -- that's the
     point. Do not re-replicate; split_cuda next iteration only ever reads
     inside a rank's own frame footprints, i.e. inside tiles_own (including
     any wrapped piece, since split_cuda wraps too).
  3. Snapshot-then-add: every neighbour is sent this rank's RAW contribution
     (captured before any received strip is added back in), otherwise a
     pixel at a multi-tile corner double-counts.
  4. Normalization tiles identically (it's a static |P|^2 scatter-add with
     the same footprint structure) -- exchange_object_halo works on it
     unchanged; rebuild object_halo only when positions move.
"""
import numpy as np

try:
    from mpi4py import MPI
    _COMM = MPI.COMM_WORLD
    RANK = _COMM.Get_rank()
    SIZE = _COMM.Get_size()
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


def _wrap_spans(v0, v1, N):
    """
    Span(s) covering [v0, v1) wrapped onto a circle of circumference N
    (matches overlap.cu's `% N` indexing).

    Normally a single (v0, min(v1,N)) span. If v1 > N, the footprint wraps:
    adds a second (0, v1-N) span for the wrapped remainder -- these two are
    always disjoint as long as the footprint is narrower than the canvas
    (v1-v0 < N), which is the ordinary case. If the footprint is as wide as
    (or wider than) the whole canvas (v1-v0 >= N), it trivially covers
    everything -- collapse to the single full-width span rather than
    emitting two spans that would overlap each other.
    """
    if v1 - v0 >= N:
        return [(0, N)]
    v1c = min(v1, N)
    spans = [(v0, v1c)]
    if v1 > N:
        spans.append((0, v1 - N))
    return spans


def _footprint_rects(local_trans, nx, ny, Nx, Ny):
    """
    All axis-aligned rectangles covering this rank's frame footprint,
    accounting for overlap.cu's periodic wraparound. Usually a single rect;
    splits into up to 4 (2 x-spans * 2 y-spans) when the aggregate footprint
    extends past the Ny/Nx edge on either axis and wraps.

    This is a conservative (possibly loose, never missing real writes)
    superset of the true per-frame footprint union: the true set is
    ∪_i (X(frame_i) × Y(frame_i)); this returns (∪_i X(frame_i)) ×
    (∪_i Y(frame_i)), which always contains the true set. That's all the
    KEY IDENTITY needs -- not tightness.

    local_trans : (local_nframes,) complex64 — tx + 1j*ty (device or host)
    Returns a list of (x0, x1, y0, y1) rects (0-4 of them; empty list if
    local_nframes == 0).
    """
    if local_trans.shape[0] == 0:
        return []

    tx = _to_np(local_trans.real, np.float64)
    ty = _to_np(local_trans.imag, np.float64)

    x0 = max(0, int(np.floor(tx.min())))
    x1 = int(np.ceil(tx.max())) + ny
    y0 = max(0, int(np.floor(ty.min())))
    y1 = int(np.ceil(ty.max())) + nx

    x_spans = _wrap_spans(x0, x1, Ny)
    y_spans = _wrap_spans(y0, y1, Nx)

    rects = []
    for xs0, xs1 in x_spans:
        for ys0, ys1 in y_spans:
            if xs1 > xs0 and ys1 > ys0:
                rects.append((xs0, xs1, ys0, ys1))
    return rects


_MAX_RECTS = 4   # 2 x-spans * 2 y-spans, worst case


def setup_object_halo(local_trans, nx, ny, Nx, Ny):
    """
    One-time halo setup (call once, before the AP loop; rebuild only if
    positions move, e.g. inside position refinement).

    Returns object_halo : dict
      tiles_own : list of (x0, x1, y0, y1) — this rank's own footprint
                  rect(s) (usually 1, up to 4 if wrapped). After
                  exchange_object_halo, only these regions are correct.
      pairs     : list of (q, (x0, x1, y0, y1)) — overlap rectangles to
                  exchange with other ranks. A given q may appear more than
                  once if wraparound creates multiple overlap regions with
                  the same neighbour; entries are grouped and sorted per
                  neighbour so both sides derive the same message order
                  (exchange_object_halo relies on this for tagging).
    """
    my_rects = _footprint_rects(local_trans, nx, ny, Nx, Ny)
    padded = (my_rects + [(0, 0, 0, 0)] * _MAX_RECTS)[:_MAX_RECTS]
    local_flat = np.array(padded, dtype=np.int64).ravel()

    all_flat = np.empty((SIZE, _MAX_RECTS * 4), dtype=np.int64)
    if _COMM is not None and SIZE > 1:
        _COMM.Allgather(local_flat, all_flat)
    else:
        all_flat[0] = local_flat

    pairs = []
    for q in range(SIZE):
        if q == RANK:
            continue
        q_rects_flat = all_flat[q].reshape(_MAX_RECTS, 4)
        q_rects = [tuple(int(v) for v in r) for r in q_rects_flat if r[1] > r[0] and r[3] > r[2]]

        q_pairs = []
        for (x0, x1, y0, y1) in my_rects:
            for (qx0, qx1, qy0, qy1) in q_rects:
                ix0, ix1 = max(x0, qx0), min(x1, qx1)
                iy0, iy1 = max(y0, qy0), min(y1, qy1)
                if ix1 > ix0 and iy1 > iy0:
                    q_pairs.append((ix0, ix1, iy0, iy1))
        q_pairs.sort()   # canonical order -- both ranks agree on it independently
        pairs.extend((q, rect) for rect in q_pairs)

    return dict(tiles_own=my_rects, pairs=pairs, Nx=Nx, Ny=Ny)


def exchange_object_halo(img0, object_halo):
    """
    Sum the overlap strips across ranks, in place. Replaces mpi_allSum(img0)
    in PASS B. After this call img0 is only correct inside
    object_halo['tiles_own'] -- see point 2 above.

    Snapshot-then-add: every rect is captured from img0 BEFORE any received
    strip is added back in (all Isend buffers are built up front), so a
    pixel touched by more than one neighbour (e.g. a 4-tile corner) gets
    each neighbour's RAW contribution added exactly once -- matching what
    mpi_allSum would have summed there.

    Tags encode a per-neighbour message index (0, 1, ...) on top of the
    (q, RANK) pair, since wraparound can produce more than one overlap rect
    with the same neighbour -- MPI only guarantees message order (not
    content) is preserved for repeated (source, tag) pairs, so without a
    unique index two same-neighbour messages posted in different relative
    orders on each side could get swapped into the wrong rectangle.
    """
    pairs = object_halo["pairs"]
    if not pairs or _COMM is None or SIZE == 1:
        return img0

    is_gpu = _GPU and isinstance(img0, _cp.ndarray)
    img0_cpu = img0.get() if is_gpu else img0

    k_by_q = {}
    entries = []
    for q, rect in pairs:
        k = k_by_q.get(q, 0)
        k_by_q[q] = k + 1
        entries.append((q, rect, k))

    reqs, sbufs, rbufs = [], [], {}
    for q, (ix0, ix1, iy0, iy1), k in entries:
        rbuf = np.empty((iy1 - iy0, ix1 - ix0), dtype=np.complex64)
        reqs.append(_COMM.Irecv(rbuf, source=q, tag=q * SIZE + RANK + 30000 + k * 1000000))
        rbufs[(q, k)] = rbuf

    for q, (ix0, ix1, iy0, iy1), k in entries:
        sbuf = np.ascontiguousarray(img0_cpu[iy0:iy1, ix0:ix1])
        sbufs.append(sbuf)
        reqs.append(_COMM.Isend(sbuf, dest=q, tag=RANK * SIZE + q + 30000 + k * 1000000))

    MPI.Request.Waitall(reqs)

    for q, (ix0, ix1, iy0, iy1), k in entries:
        img0_cpu[iy0:iy1, ix0:ix1] += rbufs[(q, k)]

    if is_gpu:
        img0[...] = _cp.asarray(img0_cpu)
    return img0


def halo_comm_bytes(object_halo):
    """Bytes this rank sends (== receives) per exchange_object_halo call."""
    total = 0
    for _, (ix0, ix1, iy0, iy1) in object_halo["pairs"]:
        total += (ix1 - ix0) * (iy1 - iy0) * 8   # complex64 = 8 bytes
    return total


# ---------------------------------------------------------------------------
# PASS B wiring (mpi_AP_step, sharpy_mpi_skeleton.py) — for reference:
#
#   from mpi_object_halo import setup_object_halo, exchange_object_halo
#   object_halo = setup_object_halo(local_trans, nx, ny, Nx, Ny)  # once
#
#   img0 = _xp.zeros((Nx, Ny), dtype=_xp.complex64)
#   for s in range(0, local_nframes, frame_batch):
#       e = min(local_nframes, s + frame_batch)
#       overlap_cuda(img0, F_local[s:e], local_translations[s:e], illumination)
#   exchange_object_halo(img0, object_halo)      # <-- replaces mpi_allSum(img0)
#   eps = reg * float(_xp.max(_xp.abs(normalization)).real)
#   return img0 / (normalization + eps)
# ---------------------------------------------------------------------------
