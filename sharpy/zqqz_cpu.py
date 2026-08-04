"""
ctypes wrapper for the OpenMP CPU Gramian kernel (src/zqqz_omp.c).

Builds the shared library on first use (parallel with -fopenmp where the
compiler supports it, serial otherwise) and exposes `gramiam_val_omp`, which
returns the per-pair <bra|ket> values -- the same thing the Numba
_braket_val_numba kernel computes, so it is a drop-in alternative for the CPU
Gramian and the line-for-line C counterpart for the GPU zQQz.cu.

This is the OpenMP-C path intended for Perlmutter (gcc -fopenmp). On macOS
(Apple clang, no -fopenmp) it builds serially -- still correct, just single
threaded.
"""

import ctypes
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src", "zqqz_omp.c")
_LIB = os.path.join(_HERE, "src", "zqqz_omp.so")

# build flag-sets to try, most-parallel first.
# -ffast-math enables limited-range complex multiply (C99's default full
# complex mul is ~10x slower); matches the Numba kernel's fastmath=True.
_BUILDS = [
    ["gcc", "-O3", "-ffast-math", "-fopenmp", "-shared", "-fPIC"],   # Linux / Perlmutter
    ["cc", "-O3", "-ffast-math", "-Xpreprocessor", "-fopenmp", "-lomp",
     "-shared", "-fPIC"],                                            # macOS + brew libomp
    ["cc", "-O3", "-ffast-math", "-Wno-unknown-pragmas",
     "-shared", "-fPIC"],                                            # serial fallback
]


def _build():
    last = None
    for flags in _BUILDS:
        try:
            subprocess.run(flags + [_SRC, "-o", _LIB],
                           check=True, capture_output=True, text=True)
            return flags[0]
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            last = e
    raise RuntimeError(f"could not build {_SRC}: {last}")


def _load():
    if not os.path.exists(_LIB) or os.path.getmtime(_SRC) > os.path.getmtime(_LIB):
        _build()
    lib = ctypes.CDLL(_LIB)
    c128 = np.ctypeslib.ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS")
    i64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.zqqz_braket.restype = None
    lib.zqqz_braket.argtypes = [
        c128, c128,            # framesl, framesr
        i64, i64, i64, i64,    # col, row, dx, dy
        ctypes.c_int,          # bw
        ctypes.c_long,         # nnz
        ctypes.c_int,          # nx
        f64,                   # frames_norm
        c128,                  # out
    ]
    lib.zqqz_braket_coupled.restype = None
    lib.zqqz_braket_coupled.argtypes = [
        c128, c128, c128, c128,  # frames, pL, pR, qq
        i64, i64, i64, i64,      # col, row, dx, dy
        ctypes.c_int,            # bw
        ctypes.c_long,           # nnz
        ctypes.c_int,            # nx
        c128, c128,              # ab, ba (out)
    ]
    return lib


_LIB_HANDLE = None


def gramiam_val_omp(framesl, framesr, col, row, dx, dy, bw, frames_norm):
    """Per-pair <bra|ket> Gramian values via the OpenMP C kernel.

    framesl, framesr : (nframes, nx, nx) complex128
    col, row, dx, dy : (nnz,) integer plan arrays
    frames_norm      : (nframes,) real
    returns          : (nnz,) complex128
    """
    global _LIB_HANDLE
    if _LIB_HANDLE is None:
        _LIB_HANDLE = _load()

    framesl = np.ascontiguousarray(framesl, dtype=np.complex128)
    framesr = np.ascontiguousarray(framesr, dtype=np.complex128)
    col = np.ascontiguousarray(col, dtype=np.int64)
    row = np.ascontiguousarray(row, dtype=np.int64)
    dx = np.ascontiguousarray(dx, dtype=np.int64)
    dy = np.ascontiguousarray(dy, dtype=np.int64)
    frames_norm = np.ascontiguousarray(frames_norm, dtype=np.float64)
    nx = framesl.shape[1]
    nnz = col.size
    out = np.empty(nnz, dtype=np.complex128)

    _LIB_HANDLE.zqqz_braket(framesl, framesr, col, row, dx, dy,
                            int(bw), int(nnz), int(nx), frames_norm, out)
    return out


def braket_coupled_omp(frames, pL, pR, qq, col, row, dx, dy, bw):
    """Coupled <bra|ket> (both orientations) via the OpenMP C kernel.

    C/OpenMP counterpart of position_retrieval._braket_coupled_numba: frames
    weighted by left/right probes pL, pR and normalization qq. Returns
    (ab, ba), each (nnz,) complex128 -- the off-diagonal O11/O22/Ox entries.
    """
    global _LIB_HANDLE
    if _LIB_HANDLE is None:
        _LIB_HANDLE = _load()

    frames = np.ascontiguousarray(frames, dtype=np.complex128)
    pL = np.ascontiguousarray(pL, dtype=np.complex128)
    pR = np.ascontiguousarray(pR, dtype=np.complex128)
    qq = np.ascontiguousarray(qq, dtype=np.complex128)
    col = np.ascontiguousarray(col, dtype=np.int64)
    row = np.ascontiguousarray(row, dtype=np.int64)
    dx = np.ascontiguousarray(dx, dtype=np.int64)
    dy = np.ascontiguousarray(dy, dtype=np.int64)
    nx = frames.shape[1]
    nnz = col.size
    ab = np.empty(nnz, dtype=np.complex128)
    ba = np.empty(nnz, dtype=np.complex128)

    _LIB_HANDLE.zqqz_braket_coupled(frames, pL, pR, qq, col, row, dx, dy,
                                    int(bw), int(nnz), int(nx), ab, ba)
    return ab, ba


if __name__ == "__main__":
    print("built with:", _build())
    print("library at:", _LIB)
