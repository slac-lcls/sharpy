"""Regression tests for the sync eigensolver map (Eigensolver_invit modes) and the
canonical-format dependency of the Gramian.

Locks in the 2026-07-04 findings:
  * Eigensolver_invit mode="cg" (matrix-free inverse iteration), mode="si"
    (matrix-free shift-invert Lanczos), and mode="direct" (splu) all recover the
    SAME per-frame consensus phase on a near-degenerate connection Gramian.
  * The returned omega is unit-modulus.
  * mapu2all builds H in CANONICAL CSR format (fast H@v SpMV path; cupy #3430) and
    val2H preserves it.
CPU / numpy authoritative.  python -m pytest tests/test_sync_eigensolver.py
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Operators
from Operators import Eigensolver_invit, eig_reset, Gramiam_plan


def _degenerate_gramian(g=16, seed=0):
    """A near-degenerate connection Gramian: 2D grid-graph adjacency (zero diagonal,
    Hermitian) with STRUCTURED per-frame weights s -> non-uniform consensus, small
    Fiedler gap. Returns a complex128 CSR (the CPU Gramian dtype)."""
    n = g * g
    rows, cols = [], []
    for i in range(g):
        for j in range(g):
            u = i * g + j
            for di, dj in ((1, 0), (0, 1)):
                if i + di < g and j + dj < g:
                    v = (i + di) * g + (j + dj)
                    rows += [u, v]
                    cols += [v, u]
    data = np.ones(len(rows), dtype=np.complex128)
    K = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    # structured weights (dark blobs) -> non-uniform degree, ones a poor eigsh anchor
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    s = np.ones((g, g))
    for _ in range(3):
        cy, cx = rng.uniform(0, g, 2)
        s *= 1.0 - 0.8 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0 ** 2))
    s = s.ravel().astype(np.complex128)
    H = (sp.diags(s) @ K @ sp.diags(s)).tocsr()
    return H


def _omega(H, mode):
    eig_reset()
    return Eigensolver_invit(H, eps=1e-3, steps=2, tol=1e-8, mode=mode).ravel()


def _align(a, b):
    return float(abs(np.vdot(a, b)) / a.size)


def test_invit_modes_agree():
    """cg (inverse iteration), si (shift-invert Lanczos), direct (splu) agree."""
    H = _degenerate_gramian()
    o_cg = _omega(H, "cg")
    o_si = _omega(H, "si")
    o_dir = _omega(H, "direct")
    assert _align(o_cg, o_si) > 0.999, _align(o_cg, o_si)
    assert _align(o_cg, o_dir) > 0.999, _align(o_cg, o_dir)


def test_invit_omega_unit_modulus():
    H = _degenerate_gramian()
    for mode in ("cg", "si", "direct"):
        o = _omega(H, mode)
        assert np.allclose(np.abs(o), 1.0, atol=1e-6), (mode, np.abs(np.abs(o) - 1).max())


def test_invit_recovers_consensus():
    """On this graph the bottom Laplacian mode is the smooth Perron consensus:
    invit-from-ones must land on it (near-constant phase, |sum omega|/n ~ 1)."""
    H = _degenerate_gramian()
    o = _omega(H, "si")
    assert abs(o.sum()) / o.size > 0.9, abs(o.sum()) / o.size


def test_gramian_canonical_format():
    """The plan's val2H H is canonical CSR (the fast H@v SpMV path depends on it;
    cupy #3430). Built through the real Gramiam_plan -> val2H path."""
    g, step, nx = 6, 8, 16
    tx, ty = np.meshgrid(np.arange(g) * step, np.arange(g) * step, indexing="ij")
    tx = tx.ravel().astype(float)
    ty = ty.ravel().astype(float)
    nf = tx.size
    Nx = Ny = (g - 1) * step + nx
    plan = Gramiam_plan(tx, ty, nf, nx, nx, Nx, Ny)
    nnz = plan["col"].size
    H = plan["val2H"](np.ones(nnz, dtype=np.complex128))
    # canonical = sorted indices + no duplicate (row, col) entries
    assert getattr(H, "has_canonical_format", getattr(H, "_has_canonical_format", True)), \
        "Gramian H must be canonical CSR (fast SpMV; cupy #3430)"
    nnz_before = H.nnz
    Hd = H.copy()
    Hd.sum_duplicates()
    assert Hd.nnz == nnz_before, "H has duplicate entries -> not canonical"
