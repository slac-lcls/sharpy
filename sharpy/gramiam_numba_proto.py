"""
Prototype: OpenMP-style CPU Gramian kernel via Numba (mirrors zQQz.cu).

Same strategy as the GPU Gramiam_calc_cuda, but the CUDA "block per pair +
threads per pixel + BlockReduce" collapses on CPU to "one parallel iteration
per overlapping pair + a serial inner sum over the overlap" -- exactly what
the OpenMP C version would do. Numba's njit(parallel=True)/prange provides the
threading with no build step.

Validates against the existing braket reference (Operators.braket_i) and times
both. If this matches, the loop body is the line-for-line reference for an
OpenMP C port for Perlmutter.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: must be CPU (config.GPU = False) for this prototype
from numba import njit, prange

from Operators import braket_i
from position_retrieval import position_plan  # CPU-safe plan (no xp.fuse)


# ---- the kernel: mirrors zQQz.cu dotp(), one pair per parallel iteration ----
@njit(parallel=True, cache=True, fastmath=True)
def gramiam_val_numba(framesl, framesr, col, row, dx, dy, bw, frames_norm, out):
    nnz = col.shape[0]
    nx = framesl.shape[1]  # square frames (ket uses nx for both axes)
    for ii in prange(nnz):
        c = col[ii]
        r = row[ii]
        dxi = dx[ii]
        dyi = dy[ii]
        # overlap window: left frame uses (-dx,-dy), right frame uses (dx,dy)
        lr0 = max(0, -dyi) + bw
        lc0 = max(0, -dxi) + bw
        rr0 = max(0, dyi) + bw
        rc0 = max(0, dxi) + bw
        hgt = nx - abs(dyi) - 2 * bw
        wid = nx - abs(dxi) - 2 * bw
        s = 0.0 + 0.0j
        for a in range(hgt):
            for b in range(wid):
                s += np.conj(framesl[c, lr0 + a, lc0 + b]) * framesr[r, rr0 + a, rc0 + b]
        out[ii] = s / (frames_norm[c] * frames_norm[r])


# ---- reference: the existing Python braket loop (what Gramiam_calc does) ----
def gramiam_val_ref(framesl, framesr, col, row, dx, dy, bw, frames_norm):
    nnz = len(col)
    val = np.empty(nnz, dtype=np.complex128)
    for ii in range(nnz):
        a = braket_i(ii, framesl, framesr, col, row, dx, dy, bw)
        val[ii] = a / (frames_norm[col[ii]] * frames_norm[row[ii]])
    return val


def main():
    assert not config.GPU, "set config.GPU = False for this CPU prototype"
    rng = np.random.default_rng(0)

    nx = ny = 32
    nnx = nny = 16          # 256 frames
    step = 4
    Nx = Ny = nnx * step
    nframes = nnx * nny

    framesl = (rng.standard_normal((nframes, nx, ny))
               + 1j * rng.standard_normal((nframes, nx, ny))).astype(np.complex128)
    framesr = (rng.standard_normal((nframes, nx, ny))
               + 1j * rng.standard_normal((nframes, nx, ny))).astype(np.complex128)
    frames_norm = (rng.random(nframes) + 0.5).astype(np.float64)

    tx, ty = np.meshgrid(np.arange(nnx) * step, np.arange(nny) * step, indexing="ij")
    plan = position_plan(tx.ravel().astype(float), ty.ravel().astype(float),
                         nframes, nx, ny, Nx, Ny, bw=0)
    col = np.ascontiguousarray(plan["col"]).astype(np.int64)
    row = np.ascontiguousarray(plan["row"]).astype(np.int64)
    dx = np.ascontiguousarray(plan["dx"]).astype(np.int64)
    dy = np.ascontiguousarray(plan["dy"]).astype(np.int64)
    bw = int(plan["bw"])
    nnz = col.size
    print(f"frames={nframes}, frame={nx}x{ny}, overlapping pairs nnz={nnz}")

    # reference
    t0 = time.time()
    val_ref = gramiam_val_ref(framesl, framesr, col, row, dx, dy, bw, frames_norm)
    t_ref = time.time() - t0

    # numba: warm up (JIT compile) then time
    out = np.empty(nnz, dtype=np.complex128)
    gramiam_val_numba(framesl, framesr, col, row, dx, dy, bw, frames_norm, out)  # compile
    t0 = time.time()
    gramiam_val_numba(framesl, framesr, col, row, dx, dy, bw, frames_norm, out)
    t_numba = time.time() - t0

    err = np.max(np.abs(out - val_ref)) / np.max(np.abs(val_ref))
    print(f"max relative error (numba vs braket ref): {err:.2e}")
    print(f"time  ref(python braket loop): {t_ref*1e3:8.2f} ms")
    print(f"time  numba (parallel):        {t_numba*1e3:8.2f} ms")
    print(f"speedup: {t_ref / t_numba:.0f}x")


if __name__ == "__main__":
    main()
