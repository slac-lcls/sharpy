"""
Regression test for Overlapc / Overlapc0 partial-coverage handling.

A padded image (frames do not cover every pixel) used to crash Overlapc:
numpy_groupies.aggregate returned only max(mapid)+1 entries, so the
reshape to (Nx, Ny) failed. Passing size/minlength=Nx*Ny zero-fills the
uncovered pixels. This is the geometry that broke the Fig 7 reproduction.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

if config.GPU:
    import cupy as cp

    xp = cp

    def tonp(a):
        return cp.asnumpy(a)
else:
    xp = np

    def tonp(a):
        return np.asarray(a)

from Operators import map_frames, Splitc, Overlapc, Overlapc0


def _padded_problem():
    """A non-periodic scan on a padded image, leaving uncovered corners."""
    nx = ny = 16
    nnx = nny = 4
    step = 6
    Nx = Ny = 64  # much larger than the scan extent -> uncovered border

    # center the small scan in the big image
    off = (Nx - step * (nnx - 1) - nx) // 2
    tx, ty = np.meshgrid(
        off + np.arange(nnx) * step, off + np.arange(nny) * step, indexing="ij"
    )
    translations_x = xp.asarray(tx.ravel().astype(np.float64))
    translations_y = xp.asarray(ty.ravel().astype(np.float64))

    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)
    return nx, ny, Nx, Ny, mapid


def test_overlapc_partial_coverage():
    nx, ny, Nx, Ny, mapid = _padded_problem()
    nframes = mapid.shape[0]
    frames = xp.ones((nframes, nx, ny), dtype=xp.complex64)

    img = Overlapc(frames, Nx, Ny, mapid)  # must not raise
    assert img.shape == (Nx, Ny)

    # corner pixel (0,0) is not covered by any frame -> exactly zero
    assert float(tonp(xp.abs(img[0, 0]))) == 0.0
    # total mass conserved: sum over image == number of frame pixels
    assert abs(float(tonp(xp.sum(img.real))) - nframes * nx * ny) < 1e-3


def test_overlapc0_partial_coverage_and_complex():
    nx, ny, Nx, Ny, mapid = _padded_problem()
    nframes = mapid.shape[0]
    # complex frames to check the 1j reconstruction
    frames = xp.ones((nframes, nx, ny), dtype=xp.complex64) * (1 + 2j)

    img = Overlapc0(frames, Nx, Ny, mapid)  # must not raise
    assert img.shape == (Nx, Ny)
    assert float(tonp(xp.abs(img[0, 0]))) == 0.0

    # Overlapc0 should agree with Overlapc (now that 1j is fixed)
    img_ref = Overlapc(frames, Nx, Ny, mapid)
    assert float(tonp(xp.max(xp.abs(img - img_ref)))) < 1e-3


if __name__ == "__main__":
    test_overlapc_partial_coverage()
    test_overlapc0_partial_coverage_and_complex()
    print("OK")
