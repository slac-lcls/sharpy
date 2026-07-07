"""pytest suite for gauge_align -- the universal gauge aligner (MGS08 subpixel registration in both
Fourier domains + Marchesini-2005 polynomial phase + global phase/scale).

  pytest -q gauge_align_test.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gauge_align as ga


def _field(H=256, W=256, seed=0, band=0.15):
    """apodized band-limited complex field (edges tapered so circular shifts don't wrap content)."""
    rng = np.random.default_rng(seed)
    fy, fx = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing="ij")
    A = np.fft.ifft2(np.fft.fft2(rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W))) * (np.hypot(fy, fx) < band))
    return np.outer(np.hanning(H), np.hanning(W)) * A / np.abs(A).mean()


def _resid(A, B, m=20):
    return np.linalg.norm((A - B)[m:-m, m:-m]) / np.linalg.norm(A[m:-m, m:-m])


def test_translation_subpixel():
    A = _field(); sy, sx = 3.7, -2.3
    B = ga._apply_shift(A, -sy, -sx)                       # B = A shifted by (-sy,-sx); recover (sy,sx)
    ry, rx = ga.register_translation(A, B, upsample=20)
    assert abs(ry - sy) < 0.05 and abs(rx - sx) < 0.05


def test_ramp_removed():
    A = _field(); H, W = A.shape; yy, xx = np.mgrid[0:H, 0:W]
    B = A * np.exp(-2j * np.pi * (-6.0 * xx / W + 4.0 * yy / H))
    Ba, _ = ga.align_gauge(A, B, poly_order=1)
    assert _resid(A, Ba) < 1e-3


def test_defocus_removed_and_reported():
    A = _field(); H, W = A.shape; yy, xx = np.mgrid[0:H, 0:W]
    xb = (xx - W / 2) / (W / 2); yb = (yy - H / 2) / (H / 2)
    B = A * np.exp(-1j * 8.0 * (xb ** 2 + yb ** 2))        # pure defocus
    Ba, p = ga.align_gauge(A, B, poly_order=2)
    assert _resid(A, Ba) < 1e-3
    assert abs((p["poly"][(2, 0)] + p["poly"][(0, 2)]) - 16.0) < 1.0   # removal = +16 (undo -8/-8)


def test_full_gauge_order2():
    A = _field(); H, W = A.shape; yy, xx = np.mgrid[0:H, 0:W]
    xb = (xx - W / 2) / (W / 2); yb = (yy - H / 2) / (H / 2)
    B = ga._apply_shift(A, -5.1, 3.3)
    B = B * np.exp(-2j * np.pi * (-1.5 * xx / W + 2.0 * yy / H))
    B = B * np.exp(-1j * 6.0 * (xb ** 2 + yb ** 2)) * (1.7 * np.exp(-1j * 0.9))
    Ba, _ = ga.align_gauge(A, B, poly_order=2)
    assert _resid(A, Ba) < 1e-3


def test_gradient_wrapfree_large_ramp():
    """the gradient fit recovers a MANY-cycle ramp with NO coarse step -- a direct arg-fit would wrap."""
    A = _field(); H, W = A.shape; yy, xx = np.mgrid[0:H, 0:W]
    B = A * np.exp(-2j * np.pi * (-20.0 * xx / W + 15.0 * yy / H))   # 20/15 cycles: ~0.5 rad/pixel, no wrap
    phase, _ = ga.fit_poly_phase(A, B, order=1)
    Bc = B * np.exp(1j * phase)
    c = np.vdot(Bc, A) / np.vdot(Bc, Bc)                   # gradients omit the constant -> fix global phase
    assert _resid(A, Bc * c) < 1e-2


def test_defocus_dz_formula():
    dz = ga.defocus_dz({(2, 0): 2.0, (0, 2): 2.0}, NA=0.084, lam=1.65e-9, N=256)
    assert abs(dz - 1.65e-9 / (4 * np.pi * 256 * 0.084 ** 2) * 4.0) < 1e-30


def test_identity_is_noop():
    A = _field()
    Ba, p = ga.align_gauge(A, A.copy(), poly_order=2)
    assert _resid(A, Ba) < 1e-6 and abs(p["scale"] - 1.0) < 1e-3
