"""
Position retrieval for ptychography.

Implements Section IV ("Position retrieval") of

    S. Marchesini, A. Schirotzek, C. Yang, H.-t. Wu, F. Maia,
    "Augmented projections for ptychographic imaging",
    Inverse Problems 29 (2013) 115009.  arXiv:1209.4924

Given a set of frames z (the current estimate, e.g. after the Fourier
magnitude projection) and the current image estimate, recover an unknown
per-frame translation xi = (xi_x, xi_y) of the probe by a second-order
Taylor expansion of the illumination operator and a least-squares solve.

This is the CuPy/NumPy port of the MATLAB reference `fit_shift.m` /
`Poverlap_branch2.m` / `doraar0_shift.m`.  It runs unchanged on CPU
(NumPy) or GPU (CuPy) via `config.GPU`, exactly like the rest of sharpy.

Milestone 1 (this file): the *diagonal* solver -- the off-diagonal
overlap coupling terms O11/O22/Ox of Eq. (27) are dropped, so each
frame's shift is an independent 2x2 least-squares solve (the paper notes
the higher-order coupling can be neglected in practice).  The fully
coupled sparse solve (reusing the Gramian plan) is a later milestone.

Conventions match the MATLAB:
  * probe derivatives via finite differences (np.gradient), with the
    second-derivative Taylor coefficients halved;
  * a Tikhonov term `reg * xrec0` regularizes the image least-squares;
  * a trust region caps the per-frame step at `max_step` pixels.
"""

import numpy as np

import config

if config.GPU:
    import cupy as cp

    xp = cp
else:
    xp = np

from Operators import Splitc, Overlapc, map_frames


def probe_derivatives(probe):
    """First and second spatial derivatives of the probe (Taylor coefficients).

    Mirrors the MATLAB `gradient`-based construction in Poverlap_branch2.m:

        dp.O = probe
        [dp.x, dp.y]  = gradient(probe)
        [dp.xx, dp.xy] = gradient(dp.x)
        [~,    dp.yy] = gradient(dp.y)
        dp.xx /= 2;  dp.yy /= 2        # second-order Taylor coefficients

    Returns a dict with keys O, x, y, xx, xy, yy, each (nx, ny).

    Note on axes: np.gradient(a) returns [d/d-axis0, d/d-axis1].  We treat
    axis 0 as x and axis 1 as y to match the MATLAB column/row convention
    used throughout sharpy's map_frames.
    """
    O = probe
    gx, gy = xp.gradient(O)
    gxx, gxy = xp.gradient(gx)
    _, gyy = xp.gradient(gy)
    return {
        "O": O,
        "x": gx,
        "y": gy,
        "xx": gxx / 2.0,
        "xy": gxy,
        "yy": gyy / 2.0,
    }


def _bcast(stack2d, per_frame):
    """Multiply a (nx, ny) array by a per-frame (nframes,) vector -> (nframes, nx, ny)."""
    return stack2d[xp.newaxis, :, :] * per_frame[:, xp.newaxis, xp.newaxis]


def taylor_shift_probe(dp, xi_x, xi_y):
    """Recenter the derivative stack about the current shift estimate.

    Port of MATLAB `Taylor_shift`: given per-frame shifts (xi_x, xi_y),
    produce per-frame shifted probe stacks (each (nframes, nx, ny)):

        O <- O + x*dx + y*dy + xx*dx^2 + xy*dx*dy + yy*dy^2
        x <- x + 2*xx*dx + xy*dy
        y <- y + 2*yy*dy + xy*dx

    Returns a dict of stacks {O, x, y} (the only ones the diagonal solver
    needs); xx/xy/yy second-order terms are kept in `dp` for recentering
    but their frame products are neglected (paper: negligible in practice).
    """
    O = dp["O"]
    px = dp["x"]
    py = dp["y"]
    pxx = dp["xx"]
    pxy = dp["xy"]
    pyy = dp["yy"]

    O_s = (
        O[xp.newaxis]
        + _bcast(px, xi_x)
        + _bcast(py, xi_y)
        + _bcast(pxx, xi_x ** 2)
        + _bcast(pxy, xi_x * xi_y)
        + _bcast(pyy, xi_y ** 2)
    )
    x_s = px[xp.newaxis] + _bcast(pxx, 2 * xi_x) + _bcast(pxy, xi_y)
    y_s = py[xp.newaxis] + _bcast(pyy, 2 * xi_y) + _bcast(pxy, xi_x)
    return {"O": O_s, "x": x_s, "y": y_s}


def position_solve_diag(
    frames,
    dp,
    xrec0,
    mapid,
    Nx,
    Ny,
    xi_x,
    xi_y,
    reg=1e-10,
    max_step=0.5,
):
    """One diagonal position-retrieval update.

    Parameters
    ----------
    frames : (nframes, nx, ny) complex
        Current frame estimate z (e.g. output of the data projection).
    dp : dict
        Probe Taylor coefficients from `probe_derivatives` (unshifted).
    xrec0 : (Nx, Ny) complex
        Current image estimate, used for the Tikhonov regularization.
    mapid : (nframes, nx, ny) int
        Frame<->image index map from `map_frames`.
    Nx, Ny : int
        Image dimensions.
    xi_x, xi_y : (nframes,) float
        Current shift estimate; updated and returned.
    reg : float
        Tikhonov weight toward xrec0 (MATLAB reg=1e-10).
    max_step : float
        Trust-region cap on per-frame step, in pixels (MATLAB 0.5).

    Returns
    -------
    xi_x, xi_y : (nframes,) float
        Updated shift estimate.
    """
    # Recenter the Taylor expansion about the current shift.
    st = taylor_shift_probe(dp, xi_x, xi_y)
    probe_O = st["O"]  # (nframes, nx, ny)
    probe_x = st["x"]
    probe_y = st["y"]

    # Image least squares:  psi_img = QQinv * (Q*[z] + reg*xrec0),  psi = split(psi_img)
    #   QQinv = 1 / (sum_overlap |probe|^2 + reg)
    QQinv = 1.0 / (Overlapc(xp.abs(probe_O) ** 2, Nx, Ny, mapid) + reg)
    psi_img = QQinv * (Overlapc(frames * xp.conj(probe_O), Nx, Ny, mapid) + reg * xrec0)
    psi = Splitc(psi_img, mapid)  # (nframes, nx, ny)

    # Derivative-weighted frames z_R1 = R1 phi* z = probe_x * psi, etc.
    zR1 = probe_x * psi
    zR2 = probe_y * psi

    # Residual zu = [I - PQ] z = z - probe_O * psi
    zu = frames - probe_O * psi

    # Per-frame reductions over the (nx, ny) axes.
    def fsum(a):
        return xp.sum(a, axis=(1, 2))

    # Diagonal 2x2 system entries (factor of 2 from +c.c. cancels with RHS).
    H11 = fsum(xp.abs(zR1) ** 2).real
    H22 = fsum(xp.abs(zR2) ** 2).real
    H12 = fsum(xp.real(xp.conj(zR1) * zR2))

    rhs1 = fsum(xp.real(xp.conj(zu) * zR1))
    rhs2 = fsum(xp.real(xp.conj(zu) * zR2))

    # Solve the per-frame 2x2 systems in closed form.
    det = H11 * H22 - H12 ** 2
    # guard against degenerate frames (no signal / no overlap)
    eps = xp.finfo(det.dtype).eps
    det = xp.where(xp.abs(det) < eps, eps, det)
    dxi_x = (H22 * rhs1 - H12 * rhs2) / det
    dxi_y = (H11 * rhs2 - H12 * rhs1) / det

    # Trust region: cap the per-frame step at max_step pixels.
    r = xp.sqrt(dxi_x ** 2 + dxi_y ** 2)
    scale = xp.where(r > 0, xp.minimum(r, max_step) / xp.where(r > 0, r, 1.0), 0.0)
    dxi_x = dxi_x * scale
    dxi_y = dxi_y * scale

    return xi_x + dxi_x, xi_y + dxi_y
