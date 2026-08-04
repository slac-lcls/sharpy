"""
Multiplex-comb COARSE position pre-localizer (Section III.A of arXiv:1209.4924)
tested as a capture-range extender BEYOND the exact-re-linearization basin.

Section IV (Taylor/exact) position retrieval is the LOCAL linearization of the
incoherent MULTIPLEX model of Section III.A: a recorded intensity is the
incoherent (intensity) sum of shifted-probe frames weighted by |omega|^2,
  a_n^2 = Omega( |F z|^2 . B|omega|^2 )            (Eq.24)
which is LINEAR in the weights. So we can localize each frame's position by a
convex NON-NEGATIVE solve over a COMB (dictionary) of candidate shifted probes:
the capture range becomes the comb SPAN, not the Taylor radius ~1/k_max.

Test (same harness as position_capture_test): fixed TRUE image, exact-Fourier-
shift data, i.i.d. shifts. Per frame, NNLS over an integer grid of candidate
shifts on the detector intensity |F(O_n P(c))|^2 -> argmax weight = coarse cell
-> warm-start the EXACT solver. Compare:
  - exact (from 0)    : production capture baseline (fails > ~2.5 px)
  - multiplex coarse  : integer-cell RMSE from the NNLS comb alone
  - multiplex + exact : coarse warm-start, then exact refine

NOTE the localizer uses the (fixed) true object frames, consistent with the
capture-test convention; in an AP loop it would use the current consensus. The
intensity localizer needs object CONTRAST within each frame -- for a flat object
|F(O P(c))|^2 is shift-invariant and the comb is degenerate (same reason
position retrieval needs object structure).

Run:  python position_multiplex_capture_test.py
"""

import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: F401  (CPU/GPU switch; this test is CPU)

from Operators import Splitc
from position_retrieval import shift_probe_fourier, position_solve_diag, shift_rmse
from position_capture_test import build_scene


def shift_probe_scalar(probe, sx, sy):
    """Exact band-limited shift of the 2D probe by a single (sx, sy)."""
    return shift_probe_fourier(probe, np.array([sx], float), np.array([sy], float))[0]


def build_comb(radius):
    return [(cx, cy) for cx in range(-radius, radius + 1)
                     for cy in range(-radius, radius + 1)]


def build_dictionary(scene, cand):
    """Intensity dictionary D[j] = |FFT(O_n . P(c_j))|^2, shape (ncand, nf, npix)."""
    O = Splitc(scene["truth"], scene["mapid"])              # (nf, nx, ny), fixed
    nf, nx, ny = O.shape
    D = np.empty((len(cand), nf, nx * ny), np.float32)
    for j, (cx, cy) in enumerate(cand):
        Pj = shift_probe_scalar(scene["probe"], cx, cy)
        fr = O * Pj
        D[j] = (np.abs(np.fft.fft2(fr)) ** 2).reshape(nf, nx * ny).astype(np.float32)
    return D


def comb_localize(scene, xi_x, xi_y, D, cand):
    """Make exact-shift intensity data, NNLS per frame over the comb -> coarse cell."""
    frames = Splitc(scene["truth"], scene["mapid"]) * \
        shift_probe_fourier(scene["probe"], xi_x, xi_y)
    I = (np.abs(np.fft.fft2(frames)) ** 2).reshape(frames.shape[0], -1).astype(np.float64)
    nf = frames.shape[0]
    cand_arr = np.array(cand, float)
    cx = np.zeros(nf); cy = np.zeros(nf)
    for n in range(nf):
        A = D[:, n, :].T.astype(np.float64)                # (npix, ncand)
        w, _ = nnls(A, I[n])
        j = int(np.argmax(w))
        cx[n], cy[n] = cand_arr[j]
    return frames, cx, cy


def refine_exact(scene, frames, hx, hy, niter=40):
    for _ in range(niter):
        hx, hy = position_solve_diag(frames, scene["dp"], scene["truth"], scene["mapid"],
                                     scene["Nx"], scene["Ny"], hx, hy, method="exact")
    return hx, hy


def exact_from_zero(scene, xi_x, xi_y, niter=60):
    frames = Splitc(scene["truth"], scene["mapid"]) * \
        shift_probe_fourier(scene["probe"], xi_x, xi_y)
    hx = np.zeros(scene["nframes"]); hy = np.zeros(scene["nframes"])
    hx, hy = refine_exact(scene, frames, hx, hy, niter)
    return shift_rmse(xi_x, xi_y, hx, hy)


if __name__ == "__main__":
    scene = build_scene()
    rng = np.random.default_rng(0)
    nf = scene["nframes"]
    bx = rng.standard_normal(nf); by = rng.standard_normal(nf)
    bx -= bx.mean(); by -= by.mean()
    s = np.sqrt(bx ** 2 + by ** 2).max()

    mags = [2.0, 2.5, 3.0, 4.0]
    radius = int(np.ceil(max(mags))) + 1
    cand = build_comb(radius)
    print(f"zone-plate probe, fixed TRUE image, {nf} frames, i.i.d. shifts")
    print(f"multiplex comb: integer grid radius {radius} ({len(cand)} candidates), "
          f"NNLS on |F(O P(c))|^2\n")
    print(f"{'max|xi| px':>10} | {'exact(from0)':>13} {'cell-correct':>12} "
          f"{'coarse Dr':>11} {'mux+exact':>11}")
    print("-" * 64)
    D = build_dictionary(scene, cand)
    for mx in mags:
        xx = bx / s * mx; yy = by / s * mx
        e0 = exact_from_zero(scene, xx, yy)
        frames, cx, cy = comb_localize(scene, xx, yy, D, cand)
        frac = np.mean((cx == np.round(xx)) & (cy == np.round(yy)))
        coarse = shift_rmse(xx, yy, cx, cy)
        hx, hy = refine_exact(scene, frames, cx.copy(), cy.copy())
        fine = shift_rmse(xx, yy, hx, hy)
        print(f"{mx:>10.2f} | {e0:>13.3e} {frac:>11.0%} {coarse:>11.3e} {fine:>11.3e}")
