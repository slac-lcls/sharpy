"""pytest suite for patched_frc -- resolution WITH an error bar via subregion (patched) FRC.

  pytest -q patched_frc_test.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import patched_frc as pf


def test_erf_window_flat_interior_zero_edge():
    w = pf.erf_window(200, 40)
    assert w[100] > 0.99           # flat = 1 across the interior
    assert w[0] < 0.05             # rolls off to ~0 at the border
    assert w[100] >= w[25]         # non-decreasing toward the center


def test_frc_self_is_one():
    A, _ = pf._band_limited_pair(0.45, noise=0.0)
    freq, f, thr = pf._frc1(A, A, taper=24)
    assert f[1:len(f) // 2].min() > 0.99          # a field correlated with itself -> FRC ~ 1


def test_frc_uncorrelated_is_low():
    rng = np.random.default_rng(3); H = W = 128
    A = rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W))
    B = rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W))
    freq, f, thr = pf._frc1(A, B, taper=24)
    assert np.abs(f[2:]).mean() < 0.2             # independent fields -> FRC ~ 0


def test_resolution_tracks_bandlimit():
    rhi = pf.patched_frc(*pf._band_limited_pair(0.35), patch=96, stride=48)
    rlo = pf.patched_frc(*pf._band_limited_pair(0.15), patch=96, stride=48)
    assert rhi["median"] > rlo["median"] + 0.05   # higher band-limit -> higher resolution


def test_stats_shape_and_positive_errorbar():
    r = pf.patched_frc(*pf._band_limited_pair(0.3), patch=96, stride=48)
    assert r["n"] > 1
    assert r["std"] >= 0.0
    assert len(r["res"]) == r["n"]
    assert r["p16"] <= r["median"] <= r["p84"]


def test_lowband_frc_drops_when_lowq_degraded():
    """lowband_frc = long-range fidelity: high when low-q agrees, low when the low-q is decorrelated."""
    rng = np.random.default_rng(5); H = W = 256
    fy, fx = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing="ij"); q = np.hypot(fy, fx)
    F = np.fft.fft2(rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W)))
    A = np.fft.ifft2(F * (q < 0.45)); FA = np.fft.fft2(A)
    Nz = np.fft.fft2(rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W)))
    good = A + 0.2 * np.std(A) * (rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W)))
    lowbad = np.fft.ifft2(np.where(q < 0.05, Nz * np.std(FA), FA))   # low-q replaced by noise
    assert pf.lowband_frc(A, good) > pf.lowband_frc(A, lowbad) + 0.1


def test_erratic_field_has_larger_errorbar():
    """a field degraded NON-uniformly (half the subregions band-limited low) has a bigger error bar
    than a uniformly-degraded one -- the error bar is a spatial-uniformity diagnostic."""
    A, B = pf._band_limited_pair(0.35)            # uniform
    Alo, Blo = pf._band_limited_pair(0.12)
    Ah, Bh = A.copy(), B.copy()
    Ah[:Ah.shape[0] // 2] = Alo[:Ah.shape[0] // 2]     # top half strongly band-limited -> erratic
    Bh[:Bh.shape[0] // 2] = Blo[:Bh.shape[0] // 2]
    r_uniform = pf.patched_frc(A, B, 96, 48)
    r_erratic = pf.patched_frc(Ah, Bh, 96, 48)
    assert r_erratic["std"] > r_uniform["std"]
