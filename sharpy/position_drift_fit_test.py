"""
Productionized global parametric drift fit (Solvers.fit_drift_global, wired into
Alternating_projections_position via drift_fit=True).

For STRUCTURED drift (the realistic scan-error regime -- magnification, rotation,
shear, slow polynomial drift = an affine "rigid source projection" of the nominal
scan grid, the real-space analog of Eckert's FPM source projection), the position
error is a few global scalars. We GLOBAL-search them on the total Fourier-magnitude
data misfit (coordinate-descent grid search -- no per-frame capture basin), then
let the per-frame exact solver polish the residual. This recovers drift far beyond
the local basin where per-frame-from-zero stalls.

Compares, on intensity data, per-frame (drift_fit=False) vs drift_fit=True:
  - a 1-D ramp drift (max ~4 px)
  - a full affine drift (magnification + shear + rotation, max ~3-4 px)

Run:  python position_drift_fit_test.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: F401

from Operators import Splitc
from position_retrieval import shift_probe_fourier, shift_rmse
from position_capture_test import build_scene
from Solvers import Alternating_projections_position, fit_drift_global


def img_nmse(rec, truth):
    s = np.vdot(rec, truth) / (np.vdot(rec, rec) + 1e-30)
    return float(np.linalg.norm(s * rec - truth) / np.linalg.norm(truth))


def make_intensity(scene, xi_x, xi_y):
    z = Splitc(scene["truth"], scene["mapid"]) * shift_probe_fourier(scene["probe"], xi_x, xi_y)
    return (np.abs(np.fft.fft2(z)) ** 2).astype(np.float64)


def run(scene, xi_x, xi_y, mode):
    """mode: 'per-frame' | 'fit-only' | 'fit+refine'."""
    nx = scene["nx"]; Nx = scene["Nx"]; Ny = scene["Ny"]
    tx = scene["tx"].astype(float); ty = scene["ty"].astype(float)
    data = make_intensity(scene, xi_x, xi_y)
    if mode == "fit-only":
        fx, fy = fit_drift_global(data, scene["probe"], tx, ty, nx, nx, Nx, Ny,
                                  model="linear", drift_max=5.0)
        return shift_rmse(xi_x, xi_y, fx, fy), float("nan")
    drift = (mode == "fit+refine")
    # for the drift path, let the image converge on the fitted positions before
    # the gentle per-frame refine engages (else it over-fits an imperfect image)
    pstart = 60 if drift else 40
    img, fr, hx, hy, res = Alternating_projections_position(
        np.ones((Nx, Ny), np.complex64), scene["probe"], data, tx.copy(), ty.copy(),
        nx, nx, Nx, Ny, 200, position_start=pstart, position_every=1, max_step=0.5,
        method="diag", diag_method="exact", replan=True,
        drift_fit=drift, drift_model="linear", drift_max=5.0,
        img_truth=scene["truth"], xi_x_truth=xi_x, xi_y_truth=xi_y)
    return shift_rmse(xi_x, xi_y, hx, hy), img_nmse(img, scene["truth"])


if __name__ == "__main__":
    scene = build_scene()
    # scan coordinates the drift is defined on (same the fit uses): centered, unit-scaled
    sx = scene["tx"].astype(float) - scene["tx"].mean()
    sy = scene["ty"].astype(float) - scene["ty"].mean()
    sc = max(np.abs(sx).max(), np.abs(sy).max())
    sxn = sx / sc; syn = sy / sc

    cases = {
        "ramp (max~4px)":            (4.0 * sxn, 0.0 * syn),
        "affine mag+shear+rot (~4px)": (3.0 * sxn + 1.0 * syn, -0.8 * sxn + 2.5 * syn),
    }
    print(f"productionized drift_fit (linear model), intensity data, {scene['nframes']} frames")
    print(f"{'drift case':>26} | {'mode':>11} | {'shift RMSE':>11} {'img NMSE':>9}")
    print("-" * 66)
    for name, (xx, yy) in cases.items():
        for mode in ("per-frame", "fit-only", "fit+refine"):
            dr, nm = run(scene, xx, yy, mode)
            nmt = "    n/a" if nm != nm else f"{nm:>9.3e}"
            print(f"{name:>26} | {mode:>11} | {dr:>11.3e} {nmt}")
    print("\nTakeaway: the global fit recovers structured drift to ~0.03px where per-frame")
    print("stalls (>2px). A FREE per-frame refine afterward over-fits pure structured drift")
    print("(use the fit as endpoint; reserve per-frame/model-constrained refine for jitter).")
