"""
Loop-integration test for position retrieval (Fig. 7 style, scaled down).

Unlike test_position_retrieval.py (which isolates the solver math with the
true image held fixed), this runs the *joint* recovery: starting from a
zero shift estimate and a poor image, recover both the image and the
per-frame sub-pixel shifts from intensity-only data inside the
alternating-projections loop (Solvers.Alternating_projections_position).

This is the in-code analogue of horse_shoe_x1.m.  Kept small so it runs on
CPU in a few seconds.
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

from Operators import map_frames, Splitc
from position_retrieval import probe_derivatives, taylor_shift_probe
import Solvers


def _make_data(seed=0, sigma_pix=1.0):
    rng = np.random.default_rng(seed)

    nx = ny = 16
    nnx = nny = 8
    step = 3
    Nx = Ny = nnx * step

    # Compact smooth Gaussian probe (good overlap, smooth Taylor derivatives).
    gx, gy = np.meshgrid(np.arange(nx) - nx / 2, np.arange(ny) - ny / 2, indexing="ij")
    sigma = nx / 4.0
    probe = np.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2)).astype(np.complex64)
    probe = xp.asarray(probe)

    # Smooth random complex truth image.
    a = rng.standard_normal((Nx, Ny)) + 1j * rng.standard_normal((Nx, Ny))
    A = np.fft.fft2(a)
    fx, fy = np.meshgrid(np.fft.fftfreq(Nx), np.fft.fftfreq(Ny), indexing="ij")
    A *= np.exp(-(fx ** 2 + fy ** 2) / (2 * 0.10 ** 2))
    truth = xp.asarray(np.fft.ifft2(A).astype(np.complex64))

    tx, ty = np.meshgrid(np.arange(nnx) * step, np.arange(nny) * step, indexing="ij")
    translations_x = xp.asarray(tx.ravel().astype(np.float64))
    translations_y = xp.asarray(ty.ravel().astype(np.float64))
    nframes = translations_x.size

    xi_x = xp.asarray(rng.standard_normal(nframes) * sigma_pix)
    xi_y = xp.asarray(rng.standard_normal(nframes) * sigma_pix)

    # Generate intensity data with the shifts baked into the probe.
    dp = probe_derivatives(probe)
    probe_shifted = taylor_shift_probe(dp, xi_x, xi_y)["O"]
    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)
    frames = Splitc(truth, mapid) * probe_shifted
    frames_data = xp.abs(xp.fft.fft2(frames)) ** 2  # intensity

    return dict(
        frames_data=frames_data,
        probe=probe,
        truth=truth,
        translations_x=translations_x,
        translations_y=translations_y,
        nx=nx,
        ny=ny,
        Nx=Nx,
        Ny=Ny,
        xi_x=xi_x,
        xi_y=xi_y,
    )


def test_position_loop_recovers_shifts():
    p = _make_data(seed=3, sigma_pix=0.7)

    # Nonzero random start (a zero image is a fixed point of the magnitude projection).
    rng = np.random.default_rng(123)
    img0 = xp.asarray(
        (rng.standard_normal((p["Nx"], p["Ny"]))
         + 1j * rng.standard_normal((p["Nx"], p["Ny"]))).astype(np.complex64)
    )

    img, frames, xhat_x, xhat_y, res = Solvers.Alternating_projections_position(
        img0,
        p["probe"],
        p["frames_data"],
        p["translations_x"],
        p["translations_y"],
        p["nx"],
        p["ny"],
        p["Nx"],
        p["Ny"],
        maxiter=400,
        position_start=80,
        position_every=1,
        img_truth=p["truth"],
        xi_x_truth=p["xi_x"],
        xi_y_truth=p["xi_y"],
        residuals_interval=1,
    )

    eps_xi = tonp(res[:, 3])
    # error metric just before position updates kick in vs at the end
    eps_before = float(eps_xi[79])
    eps_after = float(eps_xi[-1])
    print(f"eps_xi: {eps_before:.4e} -> {eps_after:.4e}")
    print(f"final image MSE: {float(tonp(res[-1, 0])):.4e}")

    # positions should improve by at least an order of magnitude
    assert eps_after < 0.1 * eps_before


if __name__ == "__main__":
    test_position_loop_recovers_shifts()
    print("OK")
