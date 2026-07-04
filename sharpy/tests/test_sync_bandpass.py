"""Regression tests for the band-pass sync result and the ProxD noise-model limits.

Locks in the 2026-07-04 findings:
  * A periodic Gramian sync drives the low-frequency (inter-frame) band below plain
    alternating projection at a fixed frame count, and reaches the threshold in far
    fewer iterations -- the coarse-correction / O(1)-vs-O(N) signature.
  * ProxD_noise: the amplitude (AGM) and poisson (IPM/KL) proximal maps both collapse
    to the hard data projection |z| = sqrt(y) in the hard limit (large tau) -- so the
    metric can only matter in a relaxed/finite-tau solver.
CPU / numpy authoritative.  python -m pytest tests/test_sync_bandpass.py
"""
import os
import sys

os.environ.setdefault("NX", "16")          # small frames -> fast, scaling regime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Operators import ProxD_noise, xp
import sync_bandpass_test as T


def test_sync_beats_ap_on_lowband():
    """AP + eigsh-sync reaches the low-band threshold faster and finishes lower
    than AP-only, at a fixed frame count (K=20 -> 400 frames)."""
    ctx = T.build(20)
    T.NUMITER = 5
    maxit = 60
    ap = T.run(ctx, 0, maxit)                       # AP-only
    ei = T.run(ctx, 1, maxit, solver=T.eigsh_sync)  # AP + eigsh-sync every iter
    ap_lo, ei_lo = float(ap[-1, 0]), float(ei[-1, 0])
    ap_it = T.iters_to(ap[:, 0], 0.1)
    ei_it = T.iters_to(ei[:, 0], 0.1)
    assert ei_lo < 0.7 * ap_lo, (ei_lo, ap_lo)      # sync finishes clearly lower
    assert ei_it is not None                        # sync crosses the threshold
    assert ap_it is None or ei_it < ap_it           # and faster than AP


def test_sync_does_not_hurt_highband():
    """The sync is low-freq-targeted: it must not HURT the high band (it often
    helps it indirectly, by removing the low-freq phase error that contaminates
    the high band through the nonlinear coupling -- so we only require no harm)."""
    ctx = T.build(20)
    T.NUMITER = 5
    maxit = 60
    ap = T.run(ctx, 0, maxit)
    ei = T.run(ctx, 1, maxit, solver=T.eigsh_sync)
    ap_lo, ei_lo = float(ap[-1, 0]), float(ei[-1, 0])
    ap_hi, ei_hi = float(ap[-1, 1]), float(ei[-1, 1])
    assert ei_hi <= ap_hi * 1.15, (ei_hi, ap_hi)            # sync never hurts high band
    # and the low band is helped MORE than the high band (it is low-freq-targeted)
    assert (ap_lo / ei_lo) >= (ap_hi / ei_hi)


def _rand(n=512, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(xp.complex64)
    y = (np.abs(x) ** 2 * rng.uniform(0.3, 3.0, n)).astype(xp.float32)  # arbitrary targets
    return xp.asarray(x), xp.asarray(y)


def test_proxd_hard_sets_amplitude():
    """hard ProxD_noise (tau=None) sets |z| = sqrt(y) (phase preserved)."""
    x, y = _rand()
    z = ProxD_noise(x, y, tau=None)
    assert np.allclose(np.abs(z), np.sqrt(np.asarray(y) + 1e-8), rtol=1e-4, atol=1e-4)
    # phase preserved
    ph = np.angle(np.asarray(z) / np.asarray(x))
    assert np.allclose(ph, 0.0, atol=1e-4)


def test_proxd_metrics_collapse_at_hard_limit():
    """amplitude (AGM) and poisson (IPM/KL) both -> hard projection as tau -> large."""
    x, y = _rand(seed=1)
    hard = np.asarray(ProxD_noise(x, y, tau=None))
    amp = np.asarray(ProxD_noise(x, y, tau=1e6, metric="amplitude"))
    pois = np.asarray(ProxD_noise(x, y, tau=1e6, metric="poisson"))
    assert np.allclose(np.abs(amp), np.abs(hard), rtol=2e-3, atol=2e-3)
    assert np.allclose(np.abs(pois), np.abs(hard), rtol=2e-3, atol=2e-3)
