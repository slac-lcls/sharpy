"""
Regression tests for the optional per-pixel detector weights (mask) in the
Fourier-magnitude data projection (Operators.ProxD / ProxD_noise /
Project_data / _proxd_resid_apply):

  (a) weights=None == weights=ones EXACTLY (bitwise) at the operator level --
      the default path must stay bit-identical to the historical behaviour;
  (b) frames_data values at W=0 pixels are never used: perturbing them
      changes neither the projected frames nor the residual;
  (c) on a padded-detector synthetic (data zero outside the mask), the
      masked-region |F| does NOT collapse over AP iterations when weights are
      passed, while the unweighted projection clamps it toward zero;
  (d) the SOLVER entry points (Solvers.Alternating_projections /
      Alternating_projections_position) pass weights= through to every
      Project_data call -- weights=ones stays bit-identical to weights=None,
      the masked region survives at solver level too (with and without the
      line_search acceleration, whose data misfit is masked by the same W),
      and ap_accel.line_search never reads masked amplitude values.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Operators as Op
import Solvers
from ap_accel import line_search

xp = Op.xp
eps = Op.eps


def _frames_and_data(nframes=4, nx=16, seed=1):
    rng = np.random.default_rng(seed)
    f = (rng.standard_normal((nframes, nx, nx))
         + 1j * rng.standard_normal((nframes, nx, nx))).astype(np.complex64)
    d = (rng.random((nframes, nx, nx)) * 10.0).astype(np.float32)
    return xp.asarray(f), xp.asarray(d)


def _mask(nx=16, seed=2):
    # binary (nx, nx) detector mask, ~half measured, broadcast over frames
    rng = np.random.default_rng(seed)
    return xp.asarray((rng.random((nx, nx)) < 0.5).astype(np.float32))


def _eq(a, b):
    return bool(xp.all(a == b))


# ------------------------- (a) None == ones, exactly -------------------------

def test_proxd_weights_ones_is_bit_identical():
    f, d = _frames_and_data()
    ones = xp.ones(f.shape[-2:], dtype=xp.float32)
    assert _eq(Op.ProxD(f, d, eps), Op.ProxD(f, d, eps, w=ones))


def test_proxd_noise_weights_ones_is_bit_identical():
    f, d = _frames_and_data(seed=3)
    ones = xp.ones(f.shape[-2:], dtype=xp.float32)
    for kw in ({"tau": None}, {"tau": 0.5, "metric": "amplitude"},
               {"tau": 0.5, "metric": "poisson"}):
        assert _eq(Op.ProxD_noise(f, d, **kw), Op.ProxD_noise(f, d, w=ones, **kw))


def test_project_data_weights_ones_is_bit_identical():
    f, d = _frames_and_data(seed=4)
    ones = xp.ones(f.shape[-2:], dtype=xp.float32)
    out0, mse0 = Op.Project_data(f.copy(), d, compute_residuals=True)
    out1, mse1 = Op.Project_data(f.copy(), d, compute_residuals=True, weights=ones)
    assert _eq(out0, out1)
    assert float(mse0) == float(mse1)


def test_proxd_resid_apply_weights_ones_matches():
    # exercises the fused-path entry (falls back to the plain path on CPU)
    f, d = _frames_and_data(seed=5)
    ones = xp.ones(f.shape[-2:], dtype=xp.float32)
    out0, mse0 = Op._proxd_resid_apply(f.copy(), d, True)
    out1, mse1 = Op._proxd_resid_apply(f.copy(), d, True, weights=ones)
    assert bool(xp.allclose(out0, out1, rtol=1e-6, atol=0))
    assert np.isclose(float(mse0), float(mse1), rtol=1e-6)


# ---------------- (b) masked-pixel data can never leak through ----------------

def test_masked_data_values_are_ignored():
    f, d = _frames_and_data(seed=6)
    W = _mask()
    # positive garbage at W=0 pixels only (must stay finite, like real padded data)
    d_perturbed = d + 37.5 * (1.0 - W)
    assert not _eq(d, d_perturbed)

    assert _eq(Op.ProxD(f, d, eps, w=W), Op.ProxD(f, d_perturbed, eps, w=W))
    for kw in ({"tau": None}, {"tau": 0.5, "metric": "amplitude"},
               {"tau": 0.5, "metric": "poisson"}):
        assert _eq(Op.ProxD_noise(f, d, w=W, **kw),
                   Op.ProxD_noise(f, d_perturbed, w=W, **kw))

    out0, mse0 = Op.Project_data(f.copy(), d, compute_residuals=True, weights=W)
    out1, mse1 = Op.Project_data(f.copy(), d_perturbed, compute_residuals=True, weights=W)
    assert _eq(out0, out1)
    assert float(mse0) == float(mse1)


def test_masked_pixels_keep_fourier_value():
    # W=0 pixels of ProxD's output are the INPUT values, untouched (not shrunk),
    # while the unweighted prox at y=0 clamps them toward |x|=0.
    f, d = _frames_and_data(seed=7)
    W = _mask()
    d = d * W                                       # padded detector: y=0 where W=0
    out = Op.ProxD(f, d, eps, w=W)
    m0 = (W == 0)
    assert _eq(out[:, m0], f[:, m0])
    clamped = Op.ProxD(f, d, eps)
    assert float(xp.abs(clamped[:, m0]).max()) < 1e-3 * float(xp.abs(f[:, m0]).min())


def test_bool_mask_accepted_by_project_data():
    f, d = _frames_and_data(seed=8)
    W = _mask()
    out_f, mse_f = Op.Project_data(f.copy(), d, compute_residuals=True, weights=W)
    out_b, mse_b = Op.Project_data(f.copy(), d, compute_residuals=True, weights=(W > 0))
    assert _eq(out_f, out_b)
    assert float(mse_f) == float(mse_b)


# ------- (c) masked |F| does not collapse on a padded-detector synthetic -------

def _padded_synthetic():
    """Padded-detector synthetic shared by the operator- and solver-level tests:
    the physical detector is a central low-frequency disk W; outside is
    zero-PADDING, not a measurement (~27% of the diffraction energy sits in
    the padding). The ground truth is a fixed point of the MASKED iteration."""
    nx, Nx = 32, 64
    tx, ty = Op.make_translations(8, 8, 8, 8, Nx, Nx)
    mapid = Op.map_frames(tx, ty, nx, nx, Nx, Nx)
    probe, _ = Op.make_probe(nx, nx)
    nframes = int(mapid.shape[0])

    rng = np.random.default_rng(0)
    img = (rng.standard_normal((Nx, Nx))
           + 1j * rng.standard_normal((Nx, Nx))).astype(np.complex64) + 2.0
    img = xp.asarray(img)

    frames_true = Op.Illuminate_frames(Op.Splitc(img, mapid), probe).astype(xp.complex64)
    F_true = xp.fft.fft2(frames_true)
    data_full = (xp.abs(F_true) ** 2).astype(xp.float32)

    xi = xp.fft.ifftshift(xp.arange(nx) - nx / 2).reshape(nx, 1)
    rr = xp.sqrt(xi ** 2 + (xi.T) ** 2)
    W = (rr <= 8).astype(xp.float32)
    data = data_full * W

    normalization = Op.Overlapc(
        Op.Replicate_frame(xp.abs(probe) ** 2, nframes), Nx, Nx, mapid
    ).real.astype(xp.float32) + 1e-8

    return dict(nx=nx, Nx=Nx, tx=tx, ty=ty, mapid=mapid, probe=probe,
                nframes=nframes, img=img, frames_true=frames_true,
                F_true=F_true, W=W, data=data, normalization=normalization)


def _masked_ratio(syn, frames):
    """Fraction of the true masked-region (padding) |F| retained by `frames`."""
    F = xp.fft.fft2(frames)
    M = 1.0 - syn["W"]
    return float((xp.abs(F) * M).sum() / (xp.abs(syn["F_true"]) * M).sum())


def test_masked_region_does_not_collapse_in_ap():
    syn = _padded_synthetic()

    def ap_masked_ratio(weights, niter=8):
        frames = syn["frames_true"].copy()
        for _ in range(niter):
            frames, _ = Op.Project_data(frames, syn["data"], weights=weights)
            im = Op.Overlapc(
                Op.Illuminate_frames(frames, xp.conj(syn["probe"])),
                syn["Nx"], syn["Nx"], syn["mapid"],
            ) / syn["normalization"]
            frames = Op.Illuminate_frames(Op.Splitc(im, syn["mapid"]), syn["probe"])
        return _masked_ratio(syn, frames)

    r_weighted = ap_masked_ratio(syn["W"])
    r_unweighted = ap_masked_ratio(None)
    # ground truth is a fixed point of the MASKED iteration: padded |F| preserved
    assert r_weighted > 0.99
    # the unweighted prox reads the padding as |F|=0 and clamps it (measured ~0.07)
    assert r_unweighted < 0.3


# --------------- (d) solver-level pass-through (Solvers.py) ---------------

def _run_ap_solver(syn, weights, niter, line_search=False, residuals_interval=np.inf):
    def Split(im):
        return Op.Splitc(im, syn["mapid"])

    def Overlap(fr):
        return Op.Overlapc(fr, syn["Nx"], syn["Nx"], syn["mapid"])

    return Solvers.Alternating_projections(
        False,                  # sync
        syn["img"] + 0.0,       # img (start at truth: fixed point of masked AP)
        None,                   # Gramiam
        syn["probe"],           # illumination
        Overlap,
        Split,
        syn["data"],
        False,                  # refine_illumination
        niter,
        syn["normalization"],
        None,                   # img_truth
        residuals_interval,
        line_search=line_search,
        weights=weights,
    )


def _rel(a, b):
    return float(xp.linalg.norm(a - b) / (1.0 + xp.linalg.norm(a)))


def test_ap_solver_weights_ones_matches_none():
    # ones vs None through the FULL solver loop. The iterate path without
    # line_search is elementwise + FFT + bincount -> bitwise identical.
    # Anything downstream of a float32 SUM/NORM reduction (the residual rows,
    # the line-search alpha) is only compared to a tolerance: on some numpy
    # builds (observed: anaconda arm64) those reductions are allocation-
    # alignment dependent and wobble ~1e-7 relative BETWEEN IDENTICAL CALLS,
    # while a genuine weights leak would show up orders of magnitude larger.
    syn = _padded_synthetic()
    ones = xp.ones((syn["nx"], syn["nx"]), dtype=xp.float32)

    img0, frames0, _, res0 = _run_ap_solver(syn, None, 3, residuals_interval=1)
    img1, frames1, _, res1 = _run_ap_solver(syn, ones, 3, residuals_interval=1)
    assert _eq(img0, img1)
    assert _eq(frames0, frames1)
    assert _rel(res0, res1) < 1e-5

    img0, frames0, _, res0 = _run_ap_solver(syn, None, 3, line_search=True,
                                            residuals_interval=1)
    img1, frames1, _, res1 = _run_ap_solver(syn, ones, 3, line_search=True,
                                            residuals_interval=1)
    assert _rel(img0, img1) < 1e-5
    assert _rel(frames0, frames1) < 1e-5
    assert _rel(res0, res1) < 1e-5


def test_ap_solver_masked_region_does_not_collapse():
    # same criterion as the operator-level test, run through the solver loop
    syn = _padded_synthetic()
    _, frames_w, _, _ = _run_ap_solver(syn, syn["W"], 8)
    _, frames_u, _, _ = _run_ap_solver(syn, None, 8)
    assert _masked_ratio(syn, frames_w) > 0.99
    assert _masked_ratio(syn, frames_u) < 0.3


def test_ap_solver_line_search_is_weights_aware():
    # with line_search=True the misfit driving alpha is masked by W: the
    # padded region must still survive (an unweighted misfit would read the
    # padding as sqrt(I)=0 and steer alpha toward collapsing it)
    syn = _padded_synthetic()
    _, frames_w, _, _ = _run_ap_solver(syn, syn["W"], 8, line_search=True)
    assert _masked_ratio(syn, frames_w) > 0.99


def test_line_search_masked_amp_values_are_ignored():
    # ap_accel.line_search with weights: amplitudes at W=0 pixels are padding,
    # not data -- perturbing them must not change alpha; weights=ones == None.
    # isclose, not ==: alpha itself wobbles ~1e-7 relative between IDENTICAL
    # calls on alignment-dependent-reduction numpy builds (see above); a leak
    # of the +11.0 perturbation would move alpha at the percent level.
    rng = np.random.default_rng(9)
    shape = (4, 16, 16)
    Z = xp.asarray((rng.standard_normal(shape)
                    + 1j * rng.standard_normal(shape)).astype(np.complex64))
    D = xp.asarray((rng.standard_normal(shape)
                    + 1j * rng.standard_normal(shape)).astype(np.complex64))
    a = xp.asarray((rng.random(shape) * 3.0).astype(np.float32))
    W = _mask(seed=10)
    a_perturbed = a + 11.0 * (1.0 - W)
    assert np.isclose(line_search(Z, D, a, weights=W),
                      line_search(Z, D, a_perturbed, weights=W), rtol=1e-5)
    ones = xp.ones(shape[-2:], dtype=xp.float32)
    assert np.isclose(line_search(Z, D, a),
                      line_search(Z, D, a, weights=ones), rtol=1e-5)


def test_position_solver_weights_ones_is_bit_identical():
    # smoke the weights wiring of Alternating_projections_position (no
    # position updates in 2 iters -- position_start defaults past them)
    syn = _padded_synthetic()
    ones = xp.ones((syn["nx"], syn["nx"]), dtype=xp.float32)

    def run(weights):
        return Solvers.Alternating_projections_position(
            syn["img"] + 0.0, syn["probe"], syn["data"],
            syn["tx"], syn["ty"], syn["nx"], syn["nx"], syn["Nx"], syn["Nx"],
            maxiter=2, weights=weights,
        )

    img0, frames0, _, _, res0 = run(None)
    img1, frames1, _, _, res1 = run(ones)
    assert _eq(img0, img1)
    assert _eq(frames0, frames1)
    assert _rel(res0, res1) < 1e-5  # norm-reduction rows: see _rel note above
