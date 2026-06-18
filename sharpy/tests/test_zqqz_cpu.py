"""
Validate the OpenMP C Gramian kernel (src/zqqz_omp.c via zqqz_cpu.py)
against the braket reference. Skips if no C compiler is available.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

if config.GPU:
    pytest.skip("CPU-only test", allow_module_level=True)

import zqqz_cpu

try:
    zqqz_cpu._build()  # needs a C compiler
except Exception as e:  # noqa
    pytest.skip(f"no C compiler for zqqz_omp: {e}", allow_module_level=True)

from Operators import braket_i
from position_retrieval import position_plan


def test_zqqz_cpu_matches_braket():
    rng = np.random.default_rng(1)
    nx = ny = 32
    nnx = nny = 8
    step = 5
    Nx = Ny = nnx * step
    nf = nnx * nny

    fl = (rng.standard_normal((nf, nx, ny))
          + 1j * rng.standard_normal((nf, nx, ny))).astype(np.complex128)
    fr = (rng.standard_normal((nf, nx, ny))
          + 1j * rng.standard_normal((nf, nx, ny))).astype(np.complex128)
    fn = (rng.random(nf) + 0.5).astype(np.float64)

    tx, ty = np.meshgrid(np.arange(nnx) * step, np.arange(nny) * step, indexing="ij")
    plan = position_plan(tx.ravel().astype(float), ty.ravel().astype(float),
                         nf, nx, ny, Nx, Ny)
    col, row, dx, dy, bw = plan["col"], plan["row"], plan["dx"], plan["dy"], plan["bw"]

    ref = np.array([braket_i(ii, fl, fr, col, row, dx, dy, bw)
                    / (fn[col[ii]] * fn[row[ii]]) for ii in range(len(col))])
    valc = zqqz_cpu.gramiam_val_omp(fl, fr, col, row, dx, dy, bw, fn)

    err = np.max(np.abs(valc - ref)) / np.max(np.abs(ref))
    print(f"max rel error (OpenMP C vs braket): {err:.2e}")
    assert err < 1e-12


def test_zqqz_cpu_coupled_matches_ref():
    """Coupled OpenMP C kernel vs the Python _braket_coupled_ref."""
    from position_retrieval import _braket_coupled_ref

    rng = np.random.default_rng(2)
    nx = ny = 32
    nnx = nny = 8
    step = 5
    Nx = Ny = nnx * step
    nf = nnx * nny

    def cx(s):
        return (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(np.complex128)

    frames, pL, pR, qq = cx((nf, nx, ny)), cx((nf, nx, ny)), cx((nf, nx, ny)), cx((nf, nx, ny))
    tx, ty = np.meshgrid(np.arange(nnx) * step, np.arange(nny) * step, indexing="ij")
    plan = position_plan(tx.ravel().astype(float), ty.ravel().astype(float),
                         nf, nx, ny, Nx, Ny)
    col, row, dx, dy, bw = plan["col"], plan["row"], plan["dx"], plan["dy"], plan["bw"]

    abr, bar = _braket_coupled_ref(frames, pL, pR, qq, col, row, dx, dy, bw)
    abc, bac = zqqz_cpu.braket_coupled_omp(frames, pL, pR, qq, col, row, dx, dy, bw)
    sc = max(np.max(np.abs(abr)), np.max(np.abs(bar)))
    err = max(np.max(np.abs(abc - abr)), np.max(np.abs(bac - bar))) / sc
    print(f"max rel error (coupled OpenMP C vs ref): {err:.2e}")
    assert err < 1e-12


if __name__ == "__main__":
    test_zqqz_cpu_matches_braket()
    test_zqqz_cpu_coupled_matches_ref()
    print("OK")
