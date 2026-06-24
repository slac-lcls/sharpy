"""
IN-LOOP test of the multiplex coarse pre-localizer (Section III.A, arXiv:1209.4924)
wired into the joint AP + position-retrieval solver.

The standalone position_multiplex_capture_test.py faked the consensus with the
TRUE image. This is the real test: recover BOTH the image and large per-frame
shifts from INTENSITY data, where the pre-localizer must work off the current
(imperfect) consensus object inside Alternating_projections_position. Compare:
  - prelocalize=False : exact-re-linearization refinement only (fails > ~2 px)
  - prelocalize=True  : one-shot multiplex comb (consensus object) -> migrate
                        integers into the map -> exact refine

Headline metric is the gauge-invariant shift RMSE (shift_rmse); image NMSE is
phase-aligned but not position-gauge-aligned, so it's indicative only.

Run:  python position_multiplex_loop_test.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: F401

from Operators import Splitc
from position_retrieval import shift_probe_fourier, shift_rmse
from position_capture_test import build_scene
from Solvers import Alternating_projections_position


def make_intensity_data(scene, xi_x, xi_y):
    z = Splitc(scene["truth"], scene["mapid"]) * \
        shift_probe_fourier(scene["probe"], xi_x, xi_y)
    return (np.abs(np.fft.fft2(z)) ** 2).astype(np.float64)


def img_nmse(rec, truth):
    """Global-phase/scale-invariant NMSE (does NOT remove the position gauge)."""
    s = np.vdot(rec, truth) / (np.vdot(rec, rec) + 1e-30)
    return float(np.linalg.norm(s * rec - truth) / np.linalg.norm(truth))


def run(scene, xi_x, xi_y, prelocalize, radius, maxiter=250, position_start=40):
    nx = scene["nx"]; Nx = scene["Nx"]; Ny = scene["Ny"]
    data = make_intensity_data(scene, xi_x, xi_y)
    img0 = np.ones((Nx, Ny), dtype=np.complex64)
    img, frames, hx, hy, res = Alternating_projections_position(
        img0, scene["probe"], data,
        scene["tx"].astype(float).copy(), scene["ty"].astype(float).copy(),
        nx, nx, Nx, Ny, maxiter,
        position_start=position_start, position_every=1, max_step=0.5,
        method="diag", diag_method="exact", replan=True,
        prelocalize=prelocalize, prelocalize_radius=radius,
        img_truth=scene["truth"], xi_x_truth=xi_x, xi_y_truth=xi_y,
    )
    return shift_rmse(xi_x, xi_y, hx, hy), img_nmse(img, scene["truth"])


if __name__ == "__main__":
    scene = build_scene()
    rng = np.random.default_rng(0)
    nf = scene["nframes"]
    bx = rng.standard_normal(nf); by = rng.standard_normal(nf)
    bx -= bx.mean(); by -= by.mean()
    s = np.sqrt(bx ** 2 + by ** 2).max()

    print(f"joint AP + position from INTENSITY, fixed scan, {nf} frames, "
          f"image {scene['Nx']}x{scene['Ny']}, i.i.d. shifts")
    print("consensus-object multiplex prelocalize vs exact-refine-only\n")
    print(f"{'max|xi| px':>10} | {'prelocalize':>11} | {'shift RMSE':>11} {'img NMSE':>10}")
    print("-" * 52)
    for mx in (2.0, 3.0, 4.0):
        xx = bx / s * mx; yy = by / s * mx
        radius = int(np.ceil(mx)) + 1
        for pl in (False, True):
            dr, nm = run(scene, xx, yy, pl, radius)
            print(f"{mx:>10.2f} | {str(pl):>11} | {dr:>11.3e} {nm:>10.3e}")
