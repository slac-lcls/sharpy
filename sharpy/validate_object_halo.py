#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_object_halo.py — green/red correctness gate for mpi_object_halo.py,
plus a comm-volume measurement (full AllReduce vs halo strips).

Run this BEFORE wiring exchange_object_halo into mpi_AP_step:
    srun -n 4  python validate_object_halo.py
    srun -n 16 python validate_object_halo.py

If it prints False, stop -- the bbox or the sum is wrong for your setup,
don't wire it into the loop.

Imports the existing get_2d_decomposition, mpi_allSum, _compute_normalization
from sharpy_mpi_skeleton.py, and Operators, so it has to run where those do
(GPU node, sharpy importable).
"""
import numpy as np

from sharpy_mpi_skeleton import (
    RANK, _COMM, cp, _xp, _SHARPY,
    get_2d_decomposition, mpi_allSum, _compute_normalization,
)
from mpi_object_halo import setup_object_halo, exchange_object_halo, halo_comm_bytes


def _to_np(a):
    return a.get() if (cp is not None and isinstance(a, cp.ndarray)) else np.asarray(a)


def main(nnx=16):
    if not _SHARPY:
        if RANK == 0:
            print("[validate-object-halo] sharpy not importable — run on a GPU node")
        return

    from Operators import Project_data
    from wrap_ops import overlap_cuda, split_cuda
    from poster_simulate import simulate

    data, illumination, truth, tx, ty, nframes, nx, ny, Nx, Ny = simulate(nnx)

    tx_np   = _to_np(tx)
    ty_np   = _to_np(ty)
    data_np = _to_np(data)

    decomp        = get_2d_decomposition(tx_np, ty_np)
    mf            = decomp["my_frames"]
    local_data_xp = _xp.asarray(data_np[mf])
    local_trans   = (_xp.asarray(tx_np[mf]) + 1j * _xp.asarray(ty_np[mf])).astype(_xp.complex64)
    local_nframes = local_data_xp.shape[0]

    # One PASS A (split + data projection) against a flat initial guess, to
    # get realistic frames for PASS B -- same shape of work mpi_AP_step does.
    img_initial = _xp.ones((Nx, Ny), dtype=_xp.complex64)
    F_local = _xp.zeros((local_nframes, nx, ny), dtype=_xp.complex64)
    for s in range(0, local_nframes, 1024):
        e = min(local_nframes, s + 1024)
        fb = _xp.zeros((e - s, nx, ny), dtype=_xp.complex64)
        split_cuda(img_initial, fb, local_trans[s:e], illumination)
        fb, _ = Project_data(fb, local_data_xp[s:e], compute_residuals=False)
        F_local[s:e] = fb

    # --- object accumulation: reference (full AllReduce) vs halo ----------
    img0_ref = _xp.zeros((Nx, Ny), dtype=_xp.complex64)
    for s in range(0, local_nframes, 1024):
        e = min(local_nframes, s + 1024)
        overlap_cuda(img0_ref, F_local[s:e], local_trans[s:e], illumination)
    img0_ref_cpu = _to_np(img0_ref)
    ref_full = mpi_allSum(img0_ref_cpu)   # ground truth, replicated on every rank

    object_halo = setup_object_halo(local_trans, nx, ny, Nx, Ny)

    img0_halo = _xp.zeros((Nx, Ny), dtype=_xp.complex64)
    for s in range(0, local_nframes, 1024):
        e = min(local_nframes, s + 1024)
        overlap_cuda(img0_halo, F_local[s:e], local_trans[s:e], illumination)
    exchange_object_halo(img0_halo, object_halo)
    img0_halo_cpu = _to_np(img0_halo)

    # --- normalization: same halo structure, static |P|^2 scatter-add -----
    norm_ref_full = _to_np(_compute_normalization(illumination, local_trans, Nx, Ny))

    norm_halo = _xp.zeros((Nx, Ny), dtype=_xp.complex64)
    overlap_cuda(norm_halo, 0, local_trans, illumination)
    exchange_object_halo(norm_halo, object_halo)
    norm_halo_cpu = _to_np(norm_halo)

    # --- per-tile comparison (object stays distributed after the exchange) -
    # tiles_own may be more than one rect if this rank's footprint wraps
    # around the canvas edge (overlap.cu's periodic %-indexing) -- check all.
    tiles_own = object_halo["tiles_own"]
    obj_err = norm_err = 0.0
    for (x0, x1, y0, y1) in tiles_own:
        obj_err  = max(obj_err,  float(np.max(np.abs(img0_halo_cpu[y0:y1, x0:x1] - ref_full[y0:y1, x0:x1]))))
        norm_err = max(norm_err, float(np.max(np.abs(norm_halo_cpu[y0:y1, x0:x1] - norm_ref_full[y0:y1, x0:x1]))))
    obj_ok  = obj_err  < 1e-3
    norm_ok = norm_err < 1e-3

    local_ok = obj_ok and norm_ok
    all_ok = bool(_COMM.allreduce(local_ok, op=_and_op())) if _COMM is not None else local_ok
    max_err = max(obj_err, norm_err)
    max_err = _COMM.allreduce(max_err, op=_max_op()) if _COMM is not None else max_err

    # --- comm-volume measurement -------------------------------------------
    full_bytes_per_rank = Nx * Ny * 8            # complex64 buffer AllReduce'd today
    halo_bytes_this_rank = halo_comm_bytes(object_halo)
    full_total  = _COMM.allreduce(full_bytes_per_rank, op=_sum_op()) if _COMM is not None else full_bytes_per_rank
    halo_total  = _COMM.allreduce(halo_bytes_this_rank, op=_sum_op()) if _COMM is not None else halo_bytes_this_rank

    if RANK == 0:
        speedup = full_total / max(halo_total, 1)
        print(f"CORRECTNESS  object+norm match on every tile : {all_ok}   max|err|={max_err:.2e}")
        print(f"COMM / iter  full AllReduce : {full_total / 1e6:8.2f} MB (all ranks)")
        print(f"             halo strips    : {halo_total / 1e6:8.2f} MB -> {speedup:.1f}x less")

    assert all_ok, "object/normalization halo exchange does not match full AllReduce"


def _sum_op():
    from mpi4py import MPI
    return MPI.SUM


def _max_op():
    from mpi4py import MPI
    return MPI.MAX


def _and_op():
    from mpi4py import MPI
    return MPI.LAND


if __name__ == "__main__":
    main()
