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

import os
import warnings

import numpy as np

import config

if config.GPU:
    import cupy as cp

    xp = cp
    import cupyx.scipy.sparse as _sparse
    import cupyx.scipy.sparse.linalg as _splinalg
else:
    xp = np
    import scipy.sparse as _sparse
    import scipy.sparse.linalg as _splinalg

from Operators import Splitc, Overlapc, map_frames, ket

try:
    from numba import njit as _njit, prange as _prange
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


# Overlap-kernel backend selection. This is a runtime *performance* choice
# (which implementation computes the overlap inner products), kept separate
# from config.GPU (the fundamental NumPy/CuPy array-library switch).
#   "auto"   -> numba if available else python  (omp is opt-in; needs a C compiler)
#   "python" -> the pure-Python reference (_braket_coupled_ref)
#   "numba"  -> the parallel Numba kernel (_braket_coupled_numba)
#   "omp"    -> the OpenMP C kernel (zqqz_cpu.braket_coupled_omp), for Perlmutter
# Resolution order, most specific first:
#   per-call `backend=` arg  >  set_kernel_backend()/SHARPY_KERNEL env  >  "auto".
_KERNEL_BACKEND = os.environ.get("SHARPY_KERNEL", "auto")


def set_kernel_backend(name):
    """Set the default overlap-kernel backend: 'auto'|'python'|'numba'|'omp'."""
    global _KERNEL_BACKEND
    _KERNEL_BACKEND = name


def get_kernel_backend():
    return _KERNEL_BACKEND


def _resolve_backend(backend=None):
    name = backend or _KERNEL_BACKEND
    if name == "auto":
        return "numba" if (_HAVE_NUMBA and not config.GPU) else "python"
    return name


def probe_derivatives(probe, method="fourier"):
    """First and second spatial derivatives of the probe (Taylor coefficients).

    Used by taylor_shift_probe as: O + x*xi + y*eta + xx*xi^2 + xy*xi*eta +
    yy*eta^2, so the returned coefficients are dx, dy, dxx/2, dxy, dyy/2.

    method="fourier" (default): spectral derivatives via a linear ramp in
        Fourier space, dx f = IFFT(i*kx*F). These are the EXACT derivatives
        within the probe's band limit, so the 2nd-order Taylor model is the
        best quadratic fit to a true sub-pixel shift -- markedly better than
        finite differences, especially near the band edge (the lens aperture
        sets k_max, which limits how large a shift the Taylor model can
        represent: it needs k_max*shift <~ 1).
    method="fd": np.gradient finite differences (the original MATLAB
        Poverlap_branch2.m construction; kept as reference).

    Axes: axis 0 = x, axis 1 = y, matching sharpy's map_frames convention.
    """
    O = probe
    if method == "fd":
        gx, gy = xp.gradient(O)
        gxx, gxy = xp.gradient(gx)
        _, gyy = xp.gradient(gy)
        return {"O": O, "x": gx, "y": gy, "xx": gxx / 2.0, "xy": gxy, "yy": gyy / 2.0}

    # spectral (Fourier-ramp) derivatives -- exact within the band limit
    nx, ny = O.shape
    kx = (2 * np.pi * xp.fft.fftfreq(nx)).reshape(nx, 1)
    ky = (2 * np.pi * xp.fft.fftfreq(ny)).reshape(1, ny)
    F = xp.fft.fft2(O)

    def d(mult):
        return xp.fft.ifft2(mult * F)

    return {
        "O": O,
        "x": d(1j * kx),
        "y": d(1j * ky),
        "xx": d(-(kx ** 2)) / 2.0,
        "xy": d(-(kx * ky)),
        "yy": d(-(ky ** 2)) / 2.0,
    }


def apodization_mask(nx, ny, support_frac=0.5, taper_frac=0.4):
    """Smooth radial mask that is 1 in the center and 0 at the frame borders.

    Multiplying the probe by this enforces the oversampling/anti-aliasing
    condition: the exit wave (probe*object) must fill <= 1/2 the frame so that
    |FT|^2 (whose support is the exit-wave autocorrelation, twice as wide) does
    not alias. support_frac=0.5 -> probe support ~ half the frame width; a
    raised-cosine taper (taper_frac of the radius) keeps it smooth (zero at the
    edges also makes the probe shift cleanly, no boundary wrap).
    """
    yy, xx = xp.meshgrid(xp.arange(nx) - nx / 2.0,
                         xp.arange(ny) - ny / 2.0, indexing="ij")
    r = xp.sqrt(xx ** 2 + yy ** 2)
    r_out = support_frac * (nx / 2.0)          # zero beyond this radius
    r_in = r_out * (1.0 - taper_frac)          # unity inside this radius
    t = xp.clip((r_out - r) / (r_out - r_in), 0.0, 1.0)
    return (0.5 * (1.0 - xp.cos(xp.pi * t))).astype(xp.float64)


def apodize_probe(probe, support_frac=0.5, taper_frac=0.4):
    """Apodize a probe to zero at the borders (~<= 1/2 frame support)."""
    m = apodization_mask(probe.shape[0], probe.shape[1], support_frac, taper_frac)
    return (probe * m).astype(probe.dtype)


def shift_probe_fourier(probe, xi_x, xi_y):
    """Exact band-limited sub-pixel shift of the probe via a Fourier phase ramp.

    probe(x+xi) = IFFT( FFT(probe) * exp(i*(kx*xi_x + ky*xi_y)) ), per frame.
    Returns a (nframes, nx, ny) stack. This is the *true* shift (all Taylor
    orders), useful for generating honest test data -- as opposed to
    taylor_shift_probe, which is the 2nd-order model the solver inverts.
    Same +xi sign convention as taylor_shift_probe.
    """
    nx, ny = probe.shape
    kx = (2 * np.pi * xp.fft.fftfreq(nx)).reshape(1, nx, 1)
    ky = (2 * np.pi * xp.fft.fftfreq(ny)).reshape(1, 1, ny)
    F = xp.fft.fft2(probe)[xp.newaxis, :, :]
    ramp = xp.exp(1j * (kx * xi_x[:, xp.newaxis, xp.newaxis]
                        + ky * xi_y[:, xp.newaxis, xp.newaxis]))
    return xp.fft.ifft2(F * ramp, axes=(1, 2))


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
    method="taylor",
):
    """One diagonal position-retrieval update.

    Parameters
    ----------
    frames : (nframes, nx, ny) complex
        Current frame estimate z (e.g. output of the data projection).
    dp : dict
        Probe derivatives from `probe_derivatives` (unshifted); needs dp["O"],
        dp["x"], dp["y"] (the latter two are the spectral first derivatives).
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
    method : {"taylor", "exact"}
        How to re-linearize the probe at the current shift. "taylor" = the
        2nd-order Taylor model (cheap; valid only within ~1/k_max). "exact" =
        the exact band-limited shifted probe and shifted spectral derivatives
        (shift_probe_fourier; shift commutes with d/dx for a band-limited probe),
        which is exact at the current estimate -> machine-precision in-range
        recovery and a larger capture radius (~1.7x) for a few extra FFTs.

    Returns
    -------
    xi_x, xi_y : (nframes,) float
        Updated shift estimate.
    """
    # Re-linearize the probe about the current shift (Taylor model or exact).
    if method == "exact":
        probe_O = shift_probe_fourier(dp["O"], xi_x, xi_y)
        probe_x = shift_probe_fourier(dp["x"], xi_x, xi_y)
        probe_y = shift_probe_fourier(dp["y"], xi_x, xi_y)
    else:
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


# ---------------------------------------------------------------------------
# Comparison baseline: gradient descent on the data discrepancy
# (Guizar-Sicairos & Fienup, Opt. Express 2008, Eq. 21)
# ---------------------------------------------------------------------------
def position_solve_gradient(
    frames_data,
    image,
    dp,
    mapid,
    xi_x,
    xi_y,
    max_step=0.5,
    n_linesearch=12,
):
    """One position update by gradient descent on the *data* discrepancy.

    The Fienup-group alternative to our Section IV solver: instead of a
    Gauss-Newton step on the projection residual, minimize the detector-space
    squared error directly.  We use the **amplitude** metric (gamma = 1/2 in
    Guizar-Sicairos & Fienup 2008, Eq. 17):

        eps = sum_n || |FFT(O_n * P(xi_n))|  -  sqrt(I_n) ||^2

    The square root is the variance-stabilizing transform for Poisson
    (photon-counting) noise, so amplitude least-squares -- not intensity
    least-squares -- is the statistically appropriate objective; it is also
    better-conditioned and matches sharpy's magnitude projection.

    Take a steepest-descent step with a backtracking line search.
    Parameterized by the same per-frame sub-pixel probe shift `xi` as
    `position_solve_diag`/`_coupled`, so the three are directly comparable.

    O_n = Split(image) are the (fixed-image) object frames; the gradient uses
    the same probe derivatives, via
        d eps / d xi_x_n = 2 sum_{u,v} (1 - sqrt(I_n)/|F_n|) Re[ conj(F_n) F(O_n P_x) ].

    Parameters mirror the other solvers; `image` is the current image estimate,
    `frames_data` the measured intensities I_n.
    """
    O = Splitc(image, mapid)  # object frames (fixed for this update)
    sqrtI = xp.sqrt(frames_data)
    tiny = xp.asarray(1e-12, dtype=sqrtI.dtype)

    def model(xix, xiy):
        st = taylor_shift_probe(dp, xix, xiy)
        f = O * st["O"]
        F = xp.fft.fft2(f)
        A = xp.abs(F)                      # |F_n|
        r = A - sqrtI                      # amplitude residual (gamma = 1/2)
        s = xp.sum(r ** 2)
        eps = float(s) if xp is np else float(s.get())
        return st, F, A, r, eps

    st, F, A, r, eps0 = model(xi_x, xi_y)

    # gradient w.r.t. per-frame shift (Eq. 21, amplitude metric, probe-shift
    # parameterization):  d eps/d xi = 2 sum (r/|F|) Re[conj(F) F(O P_xi)]
    Fx = xp.fft.fft2(O * st["x"])
    Fy = xp.fft.fft2(O * st["y"])
    w = r / xp.maximum(A, tiny)            # = 1 - sqrt(I)/|F|
    gx = 2.0 * xp.sum(w * xp.real(xp.conj(F) * Fx), axis=(1, 2))
    gy = 2.0 * xp.sum(w * xp.real(xp.conj(F) * Fy), axis=(1, 2))

    # steepest-descent direction = -g; initial step caps the largest per-frame
    # move at max_step, then backtrack until the data error decreases.
    gnorm = xp.sqrt(gx ** 2 + gy ** 2)
    gmax = float(xp.max(gnorm)) if xp is np else float(xp.max(gnorm).get())
    if gmax == 0.0:
        return xi_x, xi_y
    alpha = max_step / gmax

    for _ in range(n_linesearch):
        nx_x = xi_x - alpha * gx
        nx_y = xi_y - alpha * gy
        _, _, _, _, eps1 = model(nx_x, nx_y)
        if eps1 < eps0:
            return nx_x, nx_y
        alpha *= 0.5

    return xi_x, xi_y  # line search failed to improve; leave unchanged


def shift_rmse(xi_x_true, xi_y_true, xi_x_hat, xi_y_hat):
    """Global-shift-invariant RMSE of the recovered shifts (Fienup Eq. 15).

    (Delta r)^2 = var(x - x_hat) + var(y - y_hat), removing the global
    translation ambiguity (the reconstruction is invariant to a constant
    shift of all positions). Returns Delta r in pixels.
    """
    dxv = xi_x_true - xi_x_hat
    dyv = xi_y_true - xi_y_hat
    dr2 = (xp.var(dxv) + xp.var(dyv))
    return float(dr2 ** 0.5) if xp is np else float((dr2 ** 0.5).get())


# ---------------------------------------------------------------------------
# Milestone 3: fully coupled solver (off-diagonal overlap terms, Eq. 27)
# ---------------------------------------------------------------------------
def position_plan(translations_x, translations_y, nframes, nx, ny, Nx, Ny, bw=0):
    """Neighbor-overlap plan for the coupled position solve.

    Same geometry as Operators.Gramiam_plan -- the off-diagonal coupling
    terms O11/O22/Ox of Eq. (27) live on the same overlapping-frame-pair
    sparsity pattern as the Gramian -- but computed without the CuPy-only
    `xp.fuse` kernel, so it runs on CPU too.  Returns the plan dict with
    keys col, row, dx, dy, bw.
    """
    dx = translations_x.ravel(order="F").reshape(nframes, 1)
    dy = translations_y.ravel(order="F").reshape(nframes, 1)
    dx = xp.subtract(dx, xp.transpose(dx))
    dy = xp.subtract(dy, xp.transpose(dy))

    # periodic-boundary wrap
    dx = -(dx + Nx * ((dx < (-Nx / 2)).astype(int) - (dx > (Nx / 2)).astype(int)))
    dy = -(dy + Ny * ((dy < (-Ny / 2)).astype(int) - (dy > (Ny / 2)).astype(int)))

    # overlapping pairs, upper triangle (i <= j)
    col, row = xp.where(
        xp.triu((abs(dy) < nx - 2 * bw) * (abs(dx) < ny - 2 * bw))
    )

    dx = dx[row, col]
    dy = dy[row, col]

    return {"col": col.astype(int), "row": row.astype(int), "dx": dx, "dy": dy, "bw": bw}


if _HAVE_NUMBA:

    @_njit(parallel=True, cache=True, fastmath=True)
    def _braket_coupled_numba(frames, pL, pR, qq, col, row, dx, dy, bw, ab, ba):
        """Parallel coupled <bra|ket> over all pairs (generalized zQQz, both
        orientations). For pair ii = (a=col, b=row), probes applied inline:

            ab[ii] = sum_overlap conj(frames_a conj(pL_a)) (frames_b conj(pR_b) qq_b)
            ba[ii] = sum_overlap conj(frames_b conj(pL_b)) (frames_a conj(pR_a) qq_a)

        Same strategy as the GPU kernel: one parallel iteration per pair, a
        serial inner sum over the overlap. CPU twin of zQQz2.cu.
        """
        nnz = col.shape[0]
        nx = frames.shape[1]
        for ii in _prange(nnz):
            a = col[ii]
            b = row[ii]
            dxi = dx[ii]
            dyi = dy[ii]
            wn_r = max(0, -dyi) + bw   # "-shift" window (bra of a)
            wn_c = max(0, -dxi) + bw
            wp_r = max(0, dyi) + bw    # "+shift" window (ket of b)
            wp_c = max(0, dxi) + bw
            hgt = nx - abs(dyi) - 2 * bw
            wid = nx - abs(dxi) - 2 * bw
            sab = 0.0 + 0.0j
            sba = 0.0 + 0.0j
            for i in range(hgt):
                for j in range(wid):
                    la_r = wn_r + i; la_c = wn_c + j      # frame a window
                    rb_r = wp_r + i; rb_c = wp_c + j      # frame b window
                    fa = frames[a, la_r, la_c]
                    fb = frames[b, rb_r, rb_c]
                    sab += (np.conj(fa) * pL[a, la_r, la_c]
                            * fb * np.conj(pR[b, rb_r, rb_c]) * qq[b, rb_r, rb_c])
                    sba += (np.conj(fb) * pL[b, rb_r, rb_c]
                            * fa * np.conj(pR[a, la_r, la_c]) * qq[a, la_r, la_c])
            ab[ii] = sab
            ba[ii] = sba


def _braket_coupled(frames, pL, pR, qq, col, row, dx, dy, bw, backend=None):
    """Coupled overlap inner products for both pair orientations.

    Dispatches on the selected backend (see set_kernel_backend / _resolve_backend):
    "numba" (parallel), "omp" (OpenMP C), or "python" reference. On GPU the
    array (CuPy) reference is always used (numba/omp are CPU-only).
    """
    if config.GPU:
        return _braket_coupled_ref(frames, pL, pR, qq, col, row, dx, dy, bw)

    name = _resolve_backend(backend)

    if name == "omp":
        try:
            from zqqz_cpu import braket_coupled_omp
            return braket_coupled_omp(frames, pL, pR, qq, col, row, dx, dy, bw)
        except Exception as e:
            warnings.warn(f"omp kernel unavailable ({e}); falling back")
            name = "numba" if _HAVE_NUMBA else "python"

    if name == "numba" and _HAVE_NUMBA:
        nnz = len(col)
        ab = np.empty(nnz, dtype=np.complex128)
        ba = np.empty(nnz, dtype=np.complex128)
        _braket_coupled_numba(
            np.ascontiguousarray(frames),
            np.ascontiguousarray(pL),
            np.ascontiguousarray(pR),
            np.ascontiguousarray(qq),
            np.ascontiguousarray(col).astype(np.int64),
            np.ascontiguousarray(row).astype(np.int64),
            np.ascontiguousarray(dx).astype(np.int64),
            np.ascontiguousarray(dy).astype(np.int64),
            int(bw), ab, ba,
        )
        return ab, ba

    return _braket_coupled_ref(frames, pL, pR, qq, col, row, dx, dy, bw)


def _braket_coupled_ref(frames, pL, pR, qq, col, row, dx, dy, bw):
    """Coupled overlap inner products with the probes applied *inline*.

    CPU mirror of the generalized zQQz kernel (left/right illumination): for
    each neighbor pair, compute the overlap inner product of
    ``frames * conj(pL)`` (left) with ``frames * conj(pR) * qq`` (right)
    without ever materializing those full frame-sized products -- only the
    small overlap slices are formed.  This is the memory-saving form:
    inputs are frames, the two derivative-probe stacks pL/pR, and the split
    normalization qq, all of which we already have.

    For pair ii = (a=col, b=row):
        ab[ii] = sum_overlap conj(frames_a conj(pL_a)) * (frames_b conj(pR_b) qq_b)
        ba[ii] = sum_overlap conj(frames_b conj(pL_b)) * (frames_a conj(pR_a) qq_a)
    """
    nnz = len(col)
    ab = xp.empty(nnz, dtype=xp.complex128)
    ba = xp.empty(nnz, dtype=xp.complex128)
    for ii in range(nnz):
        a = int(col[ii])
        b = int(row[ii])
        dxi = dx[ii]
        dyi = dy[ii]

        # left frame a, right frame b
        fa = ket(frames[a], -dxi, -dyi, bw)
        pLa = ket(pL[a], -dxi, -dyi, bw)
        fb = ket(frames[b], dxi, dyi, bw)
        pRb = ket(pR[b], dxi, dyi, bw)
        qb = ket(qq[b], dxi, dyi, bw)
        ab[ii] = xp.sum(xp.conj(fa * xp.conj(pLa)) * (fb * xp.conj(pRb) * qb))

        # left frame b, right frame a (reversed displacement)
        fb2 = ket(frames[b], dxi, dyi, bw)
        pLb = ket(pL[b], dxi, dyi, bw)
        fa2 = ket(frames[a], -dxi, -dyi, bw)
        pRa = ket(pR[a], -dxi, -dyi, bw)
        qa = ket(qq[a], -dxi, -dyi, bw)
        ba[ii] = xp.sum(xp.conj(fb2 * xp.conj(pLb)) * (fa2 * xp.conj(pRa) * qa))
    return ab, ba


def _block_from_pairs(ab, ba, col, row, diag, nframes):
    """Assemble a (nframes, nframes) sparse block from neighbor-pair values.

    Off-diagonal pairs (a != b) contribute ab at [a, b] and ba at [b, a];
    diagonal pairs (a == b) contribute ab once at [a, a].  `diag` is added
    to the main diagonal.  Returns a real (2*Re) sparse matrix, matching the
    +c.c. in Eq. (27).
    """
    col = xp.asarray(col)
    row = xp.asarray(row)
    offd = col != row

    rows = xp.concatenate([col[offd], row[offd], xp.arange(nframes)])
    cols = xp.concatenate([row[offd], col[offd], xp.arange(nframes)])
    # diagonal pair value (a==b) sits in ab where col==row
    diag_pair = xp.zeros(nframes, dtype=ab.dtype)
    isdiag = ~offd
    diag_pair[col[isdiag].astype(int)] = ab[isdiag]
    vals = xp.concatenate([ab[offd], ba[offd], diag_pair + diag])

    M = _sparse.coo_matrix((vals, (rows, cols)), shape=(nframes, nframes))
    M = M.tocsr()
    return (M + M.conj().T) * 0.5 * 2.0  # = M.real symmetrized (2*Re via +c.c.)


def position_solve_coupled(
    frames,
    dp,
    xrec0,
    mapid,
    Nx,
    Ny,
    xi_x,
    xi_y,
    plan,
    reg=1e-10,
    max_step=0.5,
    backend=None,
):
    """One *coupled* position-retrieval update (Eq. 27, off-diagonal terms).

    Same inputs as `position_solve_diag` plus a neighbor-overlap `plan`
    (from `position_plan`).  Builds the full 2k x 2k system

        [H1  Hx] [xi_x]   [rhs1]
        [Hx' H2] [xi_y] = [rhs2]

    where H1/H2/Hx include the off-diagonal overlap-coupling terms
    O11/O22/Ox, Jacobi-preconditions it, and solves it with a sparse solver.
    The second-order S terms are neglected (paper: negligible in practice).
    """
    st = taylor_shift_probe(dp, xi_x, xi_y)
    probe_O = st["O"]
    probe_x = st["x"]
    probe_y = st["y"]

    QQinv = 1.0 / (Overlapc(xp.abs(probe_O) ** 2, Nx, Ny, mapid) + reg)
    psi_img = QQinv * (Overlapc(frames * xp.conj(probe_O), Nx, Ny, mapid) + reg * xrec0)
    psi = Splitc(psi_img, mapid)
    QQinv_split = Splitc(QQinv, mapid)

    zR1 = probe_x * psi
    zR2 = probe_y * psi
    zu = frames - probe_O * psi

    def fsum(a):
        return xp.sum(a, axis=(1, 2))

    # diagonal corrections (same as the diagonal solver)
    ccx = fsum(xp.abs(zR1) ** 2).real
    ccy = fsum(xp.abs(zR2) ** 2).real
    cxy = fsum(xp.real(xp.conj(zR1) * zR2))

    rhs1 = fsum(xp.real(xp.conj(zu) * zR1))
    rhs2 = fsum(xp.real(xp.conj(zu) * zR2))

    col, row = plan["col"], plan["row"]
    dx, dy, bw = plan["dx"], plan["dy"], plan["bw"]
    nframes = frames.shape[0]

    # off-diagonal overlap terms O11, O22, Ox (note the leading minus sign),
    # computed with the probes applied inline -- no Lx/Ly/Rx/Ry intermediates
    # (CPU mirror of the generalized left/right-illumination zQQz kernel).
    ab11, ba11 = _braket_coupled(frames, probe_x, probe_x, QQinv_split, col, row, dx, dy, bw, backend)
    ab22, ba22 = _braket_coupled(frames, probe_y, probe_y, QQinv_split, col, row, dx, dy, bw, backend)
    abx, bax = _braket_coupled(frames, probe_x, probe_y, QQinv_split, col, row, dx, dy, bw, backend)

    H1 = _block_from_pairs(-ab11, -ba11, col, row, ccx, nframes)
    H2 = _block_from_pairs(-ab22, -ba22, col, row, ccy, nframes)
    Hx = _block_from_pairs(-abx, -bax, col, row, cxy, nframes)

    H1 = H1.real
    H2 = H2.real
    Hx = Hx.real

    # assemble [[H1, Hx], [Hx', H2]]
    H = _sparse.bmat([[H1, Hx], [Hx.T, H2]], format="csr")
    rhs = xp.concatenate([2.0 * rhs1, 2.0 * rhs2])

    # Jacobi precondition: D = diag(1/sqrt([ccx; ccy]))
    d = 1.0 / xp.sqrt(xp.concatenate([ccx, ccy]) + 1e-30)
    D = _sparse.diags(d)
    HH = (D @ H @ D).tocsr()
    HH = (HH + HH.T) * 0.5

    try:
        y = _splinalg.spsolve(HH, D @ rhs)
    except Exception:
        y, _ = _splinalg.cg(HH, D @ rhs, maxiter=200)
    sol = d * y

    dxi_x = sol[:nframes]
    dxi_y = sol[nframes:]

    # trust region
    r = xp.sqrt(dxi_x ** 2 + dxi_y ** 2)
    scale = xp.where(r > 0, xp.minimum(r, max_step) / xp.where(r > 0, r, 1.0), 0.0)
    return xi_x + dxi_x * scale, xi_y + dxi_y * scale


# ---------------------------------------------------------------------------
# Multiplex-comb COARSE pre-localizer (Section III.A of arXiv:1209.4924).
# The Section IV Taylor/exact solver is the local linearization of the
# incoherent multiplex model  a_n^2 = Omega(|F z|^2 . B|omega|^2)  (Eq.24),
# which is LINEAR in the per-component weights.  So a frame's coarse position
# can be found by a convex non-negative fit over a COMB of candidate shifted
# probes -- capture = comb span, not the Taylor radius ~1/k_max, and no basin.
# This is the basin-free coarse stage of a coarse-to-fine pipeline
# (multiplex comb -> exact re-linearization); see position_multiplex_capture_test.py.
# ---------------------------------------------------------------------------
def multiplex_prelocalize(frames_data, img, dp, mapid, radius=4):
    """Coarse integer per-frame shift from a multiplex comb on the intensity data.

    For each frame, score a grid of candidate integer probe shifts c against the
    measured intensity by the matched filter (= the 1-sparse / single-occupancy
    limit of the Eq.24 non-negative multiplex solve): pick the c maximizing
    <I_n, D_{n,c}>^2 / ||D_{n,c}||^2, with D_{n,c} = |F(O_n . P(c))|^2 the
    candidate detector intensity built from the current consensus object frames
    O_n = Split(img).  Convex / basin-free, so the capture range is the comb span.

    Needs object CONTRAST within each frame (a flat object gives a shift-
    invariant |F(O P(c))|^2 and the comb degenerates).  Returns coarse integer
    shifts (cx, cy) in the probe-shift (xi) convention; migrate into the map as
    translations_y -= cx, translations_x -= cy (the probe<->map transpose+invert).

    Parameters
    ----------
    frames_data : (nframes, nx, ny) real   measured intensities |F z|^2.
    img : (Nx, Ny) complex                  current consensus image estimate.
    dp : dict                               probe derivatives; uses dp["O"].
    mapid : (nframes, nx, ny) int           current frame<->image map.
    radius : int                            comb half-width (candidates in +-radius).
    """
    O = Splitc(img, mapid)                                   # (nf, nx, ny)
    nf = O.shape[0]
    npix = O.shape[1] * O.shape[2]
    I = frames_data.reshape(nf, npix)
    best = xp.full(nf, -1.0)
    cx = xp.zeros(nf)
    cy = xp.zeros(nf)
    for cxj in range(-radius, radius + 1):
        for cyj in range(-radius, radius + 1):
            Pj = shift_probe_fourier(dp["O"],
                                     xp.asarray([float(cxj)]),
                                     xp.asarray([float(cyj)]))[0]
            Dj = (xp.abs(xp.fft.fft2(O * Pj)) ** 2).reshape(nf, npix)
            num = xp.sum(Dj * I, axis=1)                     # <I_n, D_n,c>
            den = xp.sum(Dj * Dj, axis=1) + 1e-30            # ||D_n,c||^2
            score = xp.maximum(num, 0.0) ** 2 / den
            upd = score > best
            best = xp.where(upd, score, best)
            cx = xp.where(upd, float(cxj), cx)
            cy = xp.where(upd, float(cyj), cy)
    return cx, cy
