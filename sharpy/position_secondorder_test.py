"""
Is the second-order ("S") correction in Section IV position retrieval really
negligible (as the paper states and both sharpy and PhilippPelz/dptycho assume
by commenting it out)?

Two distinct second-order effects are conflated, so we test both explicitly,
against the production first-order solver and against exact re-linearization.
Note that `taylor_shift_probe` ALREADY recenters the probe to 2nd order each
iteration, so the "first-order" solver carries 2nd-order curvature implicitly;
what is actually dropped is:

  (a) the Newton curvature Hessian term  -<zu, P''.psi>  (the commented-out "S"
      diagonal/cross terms). Gauss-Newton drops it -> negligible when the
      residual zu is small. Variant: "taylor+S".

  (b) a quadratic-in-increment STEP: keep the dxi^2 terms in the per-frame model
      and solve the small nonlinear LS (a few inner GN iters), instead of one
      linear step. Variant: "order2".

  vs "taylor" (production: 1st-order step, 2nd-order recentering) and "exact"
  (exact band-limited shifted probe + shifted spectral derivatives -> no Taylor
  truncation at all).

Fixed TRUE image, exact-Fourier-shift data, i.i.d. shifts (the capture harness
from position_capture_test.py). Reports an in-range accuracy point and a capture
sweep.

Run:  python position_secondorder_test.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: F401  (CPU/GPU switch; this test is CPU)

from Operators import Splitc, Overlapc, map_frames
from position_retrieval import (taylor_shift_probe, shift_probe_fourier,
                                position_solve_diag, shift_rmse)
from position_capture_test import build_scene


def _fsum(a):
    return np.sum(a, axis=(1, 2))


def _common(frames, probe_O, probe_x, probe_y, xrec0, mapid, Nx, Ny, reg):
    """psi = phi z, residual zu, and derivative-weighted frames zR1/zR2."""
    QQinv = 1.0 / (Overlapc(np.abs(probe_O) ** 2, Nx, Ny, mapid) + reg)
    psi_img = QQinv * (Overlapc(frames * np.conj(probe_O), Nx, Ny, mapid) + reg * xrec0)
    psi = Splitc(psi_img, mapid)
    zR1 = probe_x * psi
    zR2 = probe_y * psi
    zu = frames - probe_O * psi
    return psi, zR1, zR2, zu


def _solve2x2(H11, H22, H12, r1, r2):
    det = H11 * H22 - H12 ** 2
    eps = np.finfo(det.dtype).eps
    det = np.where(np.abs(det) < eps, eps, det)
    dx = (H22 * r1 - H12 * r2) / det
    dy = (H11 * r2 - H12 * r1) / det
    return dx, dy


def _trust(xi_x, xi_y, dxi_x, dxi_y, max_step):
    r = np.sqrt(dxi_x ** 2 + dxi_y ** 2)
    scale = np.where(r > 0, np.minimum(r, max_step) / np.where(r > 0, r, 1.0), 0.0)
    return xi_x + dxi_x * scale, xi_y + dxi_y * scale


def solve_taylor_S(frames, dp, xrec0, mapid, Nx, Ny, xi_x, xi_y,
                   reg=1e-10, max_step=0.5):
    """First-order step + the dropped Newton "S" Hessian term  -<zu, P''.psi>.

    The curvature coefficients are constant under 2nd-order recentering, so the
    second derivative of the model w.r.t. the increment is  C_xx = P_xx.psi
    (= 2*dp["xx"].psi), C_yy = P_yy.psi, C_xy = P_xy.psi (= dp["xy"].psi)."""
    st = taylor_shift_probe(dp, xi_x, xi_y)
    psi, zR1, zR2, zu = _common(frames, st["O"], st["x"], st["y"],
                                xrec0, mapid, Nx, Ny, reg)
    H11 = _fsum(np.abs(zR1) ** 2).real
    H22 = _fsum(np.abs(zR2) ** 2).real
    H12 = _fsum(np.real(np.conj(zR1) * zR2))
    rhs1 = _fsum(np.real(np.conj(zu) * zR1))
    rhs2 = _fsum(np.real(np.conj(zu) * zR2))
    # Newton curvature correction (the commented-out S terms): H -= Re<zu, C_ab>
    Cxx = 2.0 * dp["xx"] * psi      # = P_xx . psi
    Cyy = 2.0 * dp["yy"] * psi      # = P_yy . psi
    Cxy = dp["xy"] * psi            # = P_xy . psi
    H11 = H11 - _fsum(np.real(np.conj(zu) * Cxx))
    H22 = H22 - _fsum(np.real(np.conj(zu) * Cyy))
    H12 = H12 - _fsum(np.real(np.conj(zu) * Cxy))
    dxi_x, dxi_y = _solve2x2(H11, H22, H12, rhs1, rhs2)
    return _trust(xi_x, xi_y, dxi_x, dxi_y, max_step)


def solve_order2(frames, dp, xrec0, mapid, Nx, Ny, xi_x, xi_y,
                 reg=1e-10, max_step=0.5, inner=4):
    """Quadratic-in-increment model, solved by a few inner Gauss-Newton iters.

    model(d) = d_x zR1 + d_y zR2 + d_x^2 Sxx + d_x d_y Sxy + d_y^2 Syy
    with Sxx = dp["xx"].psi, Sxy = dp["xy"].psi, Syy = dp["yy"].psi (coeffs as
    stored: xx,yy already halved, xy not -> exactly the Taylor coefficients)."""
    st = taylor_shift_probe(dp, xi_x, xi_y)
    psi, zR1, zR2, zu = _common(frames, st["O"], st["x"], st["y"],
                                xrec0, mapid, Nx, Ny, reg)
    Sxx = dp["xx"] * psi
    Sxy = dp["xy"] * psi
    Syy = dp["yy"] * psi
    nf = frames.shape[0]
    dx = np.zeros(nf); dy = np.zeros(nf)
    for _ in range(inner):
        dxc = dx[:, None, None]; dyc = dy[:, None, None]
        model = (dxc * zR1 + dyc * zR2
                 + dxc ** 2 * Sxx + dxc * dyc * Sxy + dyc ** 2 * Syy)
        rr = zu - model
        Jx = zR1 + 2.0 * dxc * Sxx + dyc * Sxy
        Jy = zR2 + 2.0 * dyc * Syy + dxc * Sxy
        H11 = _fsum(np.abs(Jx) ** 2).real
        H22 = _fsum(np.abs(Jy) ** 2).real
        H12 = _fsum(np.real(np.conj(Jx) * Jy))
        g1 = _fsum(np.real(np.conj(rr) * Jx))
        g2 = _fsum(np.real(np.conj(rr) * Jy))
        ddx, ddy = _solve2x2(H11, H22, H12, g1, g2)
        dx = dx + ddx; dy = dy + ddy
    return _trust(xi_x, xi_y, dx, dy, max_step)


def solve_exact(frames, dp, xrec0, mapid, Nx, Ny, xi_x, xi_y,
                reg=1e-10, max_step=0.5):
    """Exact re-linearization: exact band-limited shifted probe + derivatives."""
    probe_O = shift_probe_fourier(dp["O"], xi_x, xi_y)
    probe_x = shift_probe_fourier(dp["x"], xi_x, xi_y)
    probe_y = shift_probe_fourier(dp["y"], xi_x, xi_y)
    _, zR1, zR2, zu = _common(frames, probe_O, probe_x, probe_y,
                              xrec0, mapid, Nx, Ny, reg)
    H11 = _fsum(np.abs(zR1) ** 2).real
    H22 = _fsum(np.abs(zR2) ** 2).real
    H12 = _fsum(np.real(np.conj(zR1) * zR2))
    rhs1 = _fsum(np.real(np.conj(zu) * zR1))
    rhs2 = _fsum(np.real(np.conj(zu) * zR2))
    dxi_x, dxi_y = _solve2x2(H11, H22, H12, rhs1, rhs2)
    return _trust(xi_x, xi_y, dxi_x, dxi_y, max_step)


SOLVERS = {
    "taylor":   lambda *a: position_solve_diag(*a[:8], max_step=0.5, method="taylor"),
    "taylor+S": lambda *a: solve_taylor_S(*a),
    "order2":   lambda *a: solve_order2(*a),
    "exact":    lambda *a: solve_exact(*a),
}


def recover(scene, xi_x, xi_y, method, niter=80):
    probe = scene["probe"]; dp = scene["dp"]
    Nx, Ny, nx = scene["Nx"], scene["Ny"], scene["nx"]
    mapid = scene["mapid"]
    frames = Splitc(scene["truth"], mapid) * shift_probe_fourier(probe, xi_x, xi_y)
    hx = np.zeros(scene["nframes"]); hy = np.zeros(scene["nframes"])
    solve = SOLVERS[method]
    for _ in range(niter):
        hx, hy = solve(frames, dp, scene["truth"], mapid, Nx, Ny, hx, hy)
    return shift_rmse(xi_x, xi_y, hx, hy)


if __name__ == "__main__":
    scene = build_scene()
    rng = np.random.default_rng(0)
    nf = scene["nframes"]
    base_x = rng.standard_normal(nf); base_y = rng.standard_normal(nf)
    base_x -= base_x.mean(); base_y -= base_y.mean()
    s = np.sqrt(base_x ** 2 + base_y ** 2).max()

    print(f"zone-plate probe, fixed TRUE image, {nf} frames, i.i.d. shifts, niter=80")
    print("Delta_r (shift RMSE) vs max|xi|; columns = solver variant\n")
    methods = ["taylor", "taylor+S", "order2", "exact"]
    print(f"{'max|xi| px':>10} | " + " ".join(f"{m:>11}" for m in methods))
    print("-" * (12 + 13 * len(methods)))
    for mx in (0.4, 0.8, 1.5, 2.0, 2.5, 3.0, 4.0):
        xx = base_x / s * mx; yy = base_y / s * mx
        vals = [recover(scene, xx, yy, m) for m in methods]
        print(f"{mx:>10.2f} | " + " ".join(f"{v:>11.3e}" for v in vals))
