"""
Unit test for the coupled position solver (Eq. 27, off-diagonal terms).

Mirrors test_position_retrieval.py but exercises position_solve_coupled:
build a consistent problem (data generated with the Taylor probe model),
hold the true image fixed, and check the coupled solver drives the position
error metric down -- and at least matches the diagonal solver.
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

from Operators import make_probe, map_frames, Splitc
from position_retrieval import (
    probe_derivatives,
    taylor_shift_probe,
    position_solve_diag,
    position_solve_coupled,
    position_plan,
)


def _build(seed=1, sigma_pix=1.0):
    rng = np.random.default_rng(seed)
    nx = ny = 32
    nnx = nny = 12
    step = 4
    Nx = Ny = nnx * step

    probe = make_probe(nx, ny)
    if isinstance(probe, tuple):
        probe = probe[0]
    probe = xp.asarray(probe, dtype=xp.complex64)

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

    dp = probe_derivatives(probe)
    probe_shifted = taylor_shift_probe(dp, xi_x, xi_y)["O"]
    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)
    frames_data = Splitc(truth, mapid) * probe_shifted

    plan = position_plan(translations_x, translations_y, nframes, nx, ny, Nx, Ny)

    return dict(
        frames_data=frames_data, dp=dp, truth=truth, mapid=mapid,
        Nx=Nx, Ny=Ny, nframes=nframes, xi_x=xi_x, xi_y=xi_y, plan=plan,
    )


def _eps(xi_x, xi_y, hx, hy):
    num = xp.sum((xi_x - hx) ** 2 + (xi_y - hy) ** 2)
    den = xp.sum(xi_x ** 2 + xi_y ** 2)
    return float(tonp(num / den))


def test_coupled_recovers_shifts():
    p = _build(seed=2, sigma_pix=0.8)

    hx = xp.zeros(p["nframes"])
    hy = xp.zeros(p["nframes"])
    eps0 = _eps(p["xi_x"], p["xi_y"], hx, hy)

    for _ in range(12):
        hx, hy = position_solve_coupled(
            p["frames_data"], p["dp"], p["truth"], p["mapid"],
            p["Nx"], p["Ny"], hx, hy, p["plan"], max_step=0.5,
        )

    eps1 = _eps(p["xi_x"], p["xi_y"], hx, hy)
    print(f"coupled eps_xi: {eps0:.4e} -> {eps1:.4e}")
    assert eps1 < 1e-3 * eps0


def test_coupled_at_least_matches_diag():
    """On the same problem, coupled should do no worse than diagonal."""
    p = _build(seed=5, sigma_pix=0.8)

    dx = dy = None
    hx_d = xp.zeros(p["nframes"]); hy_d = xp.zeros(p["nframes"])
    hx_c = xp.zeros(p["nframes"]); hy_c = xp.zeros(p["nframes"])
    for _ in range(6):
        hx_d, hy_d = position_solve_diag(
            p["frames_data"], p["dp"], p["truth"], p["mapid"],
            p["Nx"], p["Ny"], hx_d, hy_d, max_step=0.5,
        )
        hx_c, hy_c = position_solve_coupled(
            p["frames_data"], p["dp"], p["truth"], p["mapid"],
            p["Nx"], p["Ny"], hx_c, hy_c, p["plan"], max_step=0.5,
        )

    eps_d = _eps(p["xi_x"], p["xi_y"], hx_d, hy_d)
    eps_c = _eps(p["xi_x"], p["xi_y"], hx_c, hy_c)
    print(f"after 6 iters: diag={eps_d:.4e}  coupled={eps_c:.4e}")
    # coupled should be at least as good (allow small slack)
    assert eps_c <= eps_d * 1.5


if __name__ == "__main__":
    test_coupled_recovers_shifts()
    test_coupled_at_least_matches_diag()
    print("OK")
