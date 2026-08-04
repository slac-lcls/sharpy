"""
Validate the Numba <bra|ket> fast path in Gramiam_calc against the explicit
braket reference (the assembled Gramian matrix H must match).
"""

import os
import sys

import numpy as np
import scipy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

if config.GPU:
    import pytest
    pytest.skip("CPU-only test", allow_module_level=True)

import Operators
from Operators import Gramiam_calc, Gramiam_plan, braket_i


def _ref_H(framesl, framesr, plan, frames_norm):
    """Reference Gramian: explicit braket_i loop + the same CPU assembly."""
    col, row = plan["col"], plan["row"]
    dx, dy, bw = plan["dx"], plan["dy"], plan["bw"]
    nframes = framesl.shape[0]
    nnz = len(col)
    val = np.empty(nnz, dtype=np.complex128)
    for ii in range(nnz):
        val[ii] = braket_i(ii, framesl, framesr, col, row, dx, dy, bw)
        val[ii] /= frames_norm[col[ii]] * frames_norm[row[ii]]
    H = sp.sparse.coo_matrix((val.ravel(), (col, row)), shape=(nframes, nframes))
    H = H + sp.sparse.triu(H, 1).getH()
    return H.tocsr()


def test_gramiam_numba_matches_braket():
    assert Operators._HAVE_NUMBA, "numba not available"
    rng = np.random.default_rng(0)

    nx = ny = 32
    nnx = nny = 8
    step = 5
    Nx = Ny = nnx * step
    nframes = nnx * nny

    framesl = (rng.standard_normal((nframes, nx, ny))
               + 1j * rng.standard_normal((nframes, nx, ny))).astype(np.complex128)
    framesr = (rng.standard_normal((nframes, nx, ny))
               + 1j * rng.standard_normal((nframes, nx, ny))).astype(np.complex128)
    frames_norm = (rng.random(nframes) + 0.5).astype(np.float64)

    tx, ty = np.meshgrid(np.arange(nnx) * step, np.arange(nny) * step, indexing="ij")
    # Gramiam_plan now works on CPU (xp.fuse guarded); exercises plan + calc.
    plan = Gramiam_plan(tx.ravel().astype(float), ty.ravel().astype(float),
                        nframes, nx, ny, Nx, Ny, bw=0)

    H_ref = _ref_H(framesl, framesr, plan, frames_norm)
    H_fast = Gramiam_calc(framesl, framesr, plan, frames_norm)  # numba path

    diff = np.max(np.abs((H_fast - H_ref).toarray()))
    scale = np.max(np.abs(H_ref.toarray()))
    print(f"max |H_fast - H_ref| / scale = {diff/scale:.2e}")
    assert diff / scale < 1e-12


if __name__ == "__main__":
    test_gramiam_numba_matches_braket()
    print("OK")
