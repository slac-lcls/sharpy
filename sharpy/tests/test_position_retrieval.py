"""
Fig. 7 reproduction test for diagonal position retrieval.

Mirrors the setup in arXiv:1209.4924 Fig. 7:
  * 16x16 frames, 32x32 frame size, hexagonal-ish packing,
  * known truth image and probe,
  * unknown random per-frame shifts xi (sub-pixel to ~2.5 resolution
    elements) applied to the data, then recovered.

The test asserts that the position error metric
    eps_xi = sum |xi - xi_hat|^2 / sum |xi|^2
decreases substantially after a few diagonal updates, with the true
image and probe held fixed (isolating the position solver).

Runs on CPU (NumPy) by default; flip config.GPU for the CuPy path.
"""

import os
import sys

import numpy as np
import pytest

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

from Operators import make_probe, map_frames, Splitc, Overlapc, Illuminate_frames
from position_retrieval import probe_derivatives, position_solve_diag


def _build_problem(seed=0, sigma_pix=1.0):
    """Construct a small ptychography problem with known shifts.

    Returns frames_data (with shifts baked in), probe, truth image,
    integer translations, and the true sub-pixel shifts xi.
    """
    rng = np.random.default_rng(seed)

    nx = ny = 32
    nnx = nny = 16  # 16x16 scan
    step = 4  # integer step (pixels)

    # Image big enough to hold the scan with wrap-around.
    Nx = Ny = nnx * step

    # Probe: a smooth disk-like illumination (make_probe gives a zone-plate-ish probe).
    # On the refine_illumination branch make_probe returns (probe, lens_mask).
    probe = make_probe(nx, ny)
    if isinstance(probe, tuple):
        probe = probe[0]
    probe = xp.asarray(probe, dtype=xp.complex64)

    # Truth: smooth random complex image.
    a = rng.standard_normal((Nx, Ny)) + 1j * rng.standard_normal((Nx, Ny))
    # low-pass it so it's well sampled by the probe
    A = np.fft.fft2(a)
    yy, xxg = np.meshgrid(np.fft.fftfreq(Ny), np.fft.fftfreq(Nx))
    A *= np.exp(-(xxg ** 2 + yy ** 2) / (2 * (0.08 ** 2)))
    truth = xp.asarray(np.fft.ifft2(A).astype(np.complex64))

    # Integer scan positions (square packing; close enough for the test).
    tx, ty = np.meshgrid(np.arange(nnx) * step, np.arange(nny) * step)
    translations_x = xp.asarray(tx.ravel().astype(np.float64))
    translations_y = xp.asarray(ty.ravel().astype(np.float64))
    nframes = translations_x.size

    # Unknown true sub-pixel shifts.
    xi_x = xp.asarray(rng.standard_normal(nframes) * sigma_pix)
    xi_y = xp.asarray(rng.standard_normal(nframes) * sigma_pix)

    # Build data WITH the shifts baked in, by shifting the probe per frame
    # using the same Taylor model (consistent forward model for the test).
    dp = probe_derivatives(probe)
    from position_retrieval import taylor_shift_probe

    st = taylor_shift_probe(dp, xi_x, xi_y)
    probe_shifted = st["O"]  # (nframes, nx, ny)

    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)
    frames_truth = Splitc(truth, mapid)
    frames_data = frames_truth * probe_shifted

    return dict(
        frames_data=frames_data,
        probe=probe,
        dp=dp,
        truth=truth,
        mapid=mapid,
        Nx=Nx,
        Ny=Ny,
        nframes=nframes,
        xi_x=xi_x,
        xi_y=xi_y,
    )


def _eps_xi(xi_x, xi_y, xhat_x, xhat_y):
    num = xp.sum((xi_x - xhat_x) ** 2 + (xi_y - xhat_y) ** 2)
    den = xp.sum(xi_x ** 2 + xi_y ** 2)
    return float(tonp(num / den))


def test_position_retrieval_reduces_error():
    """Diagonal solver should shrink the position error metric (Fig. 7)."""
    p = _build_problem(seed=1, sigma_pix=1.0)

    # Start from zero shift estimate (positions assumed at integer grid).
    xhat_x = xp.zeros(p["nframes"])
    xhat_y = xp.zeros(p["nframes"])

    eps0 = _eps_xi(p["xi_x"], p["xi_y"], xhat_x, xhat_y)

    # The data already equals probe_shifted * truth; with the true image
    # known, iterate the position solver a handful of times.
    for _ in range(15):
        xhat_x, xhat_y = position_solve_diag(
            p["frames_data"],
            p["dp"],
            p["truth"],
            p["mapid"],
            p["Nx"],
            p["Ny"],
            xhat_x,
            xhat_y,
            max_step=0.5,
        )

    eps1 = _eps_xi(p["xi_x"], p["xi_y"], xhat_x, xhat_y)

    print(f"eps_xi: {eps0:.4e} -> {eps1:.4e}")
    # Expect a large reduction once positions are recovered.
    assert eps1 < 0.1 * eps0


if __name__ == "__main__":
    test_position_retrieval_reduces_error()
    print("OK")
