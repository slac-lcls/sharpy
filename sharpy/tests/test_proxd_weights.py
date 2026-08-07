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
      passed, while the unweighted projection clamps it toward zero.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Operators as Op

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

def test_masked_region_does_not_collapse_in_ap():
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

    # physical detector = central low-frequency disk; outside is zero-PADDING,
    # not a measurement (~27% of the diffraction energy sits in the padding)
    xi = xp.fft.ifftshift(xp.arange(nx) - nx / 2).reshape(nx, 1)
    rr = xp.sqrt(xi ** 2 + (xi.T) ** 2)
    W = (rr <= 8).astype(xp.float32)
    data = data_full * W

    normalization = Op.Overlapc(
        Op.Replicate_frame(xp.abs(probe) ** 2, nframes), Nx, Nx, mapid
    ).real.astype(xp.float32) + 1e-8

    def ap_masked_ratio(weights, niter=8):
        frames = frames_true.copy()
        for _ in range(niter):
            frames, _ = Op.Project_data(frames, data, weights=weights)
            im = Op.Overlapc(
                Op.Illuminate_frames(frames, xp.conj(probe)), Nx, Nx, mapid
            ) / normalization
            frames = Op.Illuminate_frames(Op.Splitc(im, mapid), probe)
        F = xp.fft.fft2(frames)
        M = 1.0 - W
        return float((xp.abs(F) * M).sum() / (xp.abs(F_true) * M).sum())

    r_weighted = ap_masked_ratio(W)
    r_unweighted = ap_masked_ratio(None)
    # ground truth is a fixed point of the MASKED iteration: padded |F| preserved
    assert r_weighted > 0.99
    # the unweighted prox reads the padding as |F|=0 and clamps it (measured ~0.07)
    assert r_unweighted < 0.3
