"""CPU tests for the fused power-iteration path (_eig_power_fused).

The GPU kernels (eig_step / eig_scale) cannot run here; what IS testable on
CPU, and is where the correctness risk actually lives, is the ALGEBRA and the
CONTROL FLOW:

  (a) the step identity  ||y/||y|| - v||^2 = 2 - 2 Re<y,v>/||y||  that lets
      one fused pass replace two norms + a materialized difference;
  (b) a numpy mirror of _eig_power_fused's loop (same formulas, same stopping
      logic) reproduces the stock power loop's step sequence exactly;
  (c) the mirrored fused path recovers the dominant eigenvector;
  (d) Chebyshev momentum converges in fewer matvecs than plain power on a
      small-gap (large-N-like) connection Gramian;
  (e) the fused path stays OFF on CPU regardless of the env flag.

GPU kernel parity is covered by eig_fused_gpu_test.py on an A100.
"""
import math
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Operators
from tests.test_sync_eigensolver import _degenerate_gramian

RNG = np.random.RandomState(7)


def _shifted_gramian(g=16, shift=5.0):
    """Grid connection Gramian + shift*I: bipartite adjacency alone has
    lambda_min = -lambda_max (power iteration oscillates); the diagonal
    self-overlap term of the real Gramian breaks that, modeled by the shift."""
    H = _degenerate_gramian(g=g)
    n = H.shape[0]
    return (H + shift * sp.identity(n, dtype=H.dtype, format="csr")).tocsr()


def _mirror_fused(H, v, num_iter, tol, momentum=False, warmup=8, K=4, R=16):
    """CPU mirror of Operators._eig_power_fused: numpy replaces the kernels,
    but the DECISIONS come from the same _EigAdapt instance the GPU loop uses
    (fed in the same batches of K), so control parity holds by construction.
    Returns (v, matvecs_used)."""
    v = np.asarray(v, dtype=np.complex128).ravel().copy()
    v /= np.linalg.norm(v)
    vp = np.zeros_like(v)
    ctl = Operators._EigAdapt(tol, momentum, warmup=warmup, R=R)
    rows = []
    used = 0
    for it in range(num_iter):
        y = H @ v
        used += 1
        ssq_y = float(np.vdot(y, y).real)
        dot = float(np.vdot(y, v).real)
        ssq_v = float(np.vdot(v, v).real)
        t = y - ctl.beta * vp if ctl.beta != 0.0 else y
        ssq_t = float(np.vdot(t, t).real)
        rows.append((ssq_t, ssq_y, dot, ssq_v))
        s = 1.0 / math.sqrt(ssq_t) if ssq_t > 0 else 0.0
        vp = v * s
        v = t * s
        if (it + 1) % K == 0 or it == num_iter - 1:
            ctl.feed(rows)
            rows = []
            if ctl.stop:
                break
    return v, used


def _stock_steps(H, v0, num_iter):
    """The stock Eigensolver inner loop, recording each step value."""
    v = np.asarray(v0, dtype=np.complex128).ravel().copy()
    v /= np.linalg.norm(v)
    steps = []
    for _ in range(num_iter):
        vn = H @ v
        vn /= np.linalg.norm(vn)
        steps.append(float(np.linalg.norm(vn - v)))
        v = vn
    return v, steps


def _dominant(H):
    w, V = np.linalg.eigh(H.toarray())
    return V[:, -1], w


def test_step_identity():
    # (a) the identity that removes one norm, one materialize and the sync
    for n in (33, 256, 1000):
        y = RNG.randn(n) + 1j * RNG.randn(n)
        v = RNG.randn(n) + 1j * RNG.randn(n)
        v /= np.linalg.norm(v)
        explicit = np.linalg.norm(y / np.linalg.norm(y) - v) ** 2
        fused = 2.0 - 2.0 * np.vdot(y, v).real / np.linalg.norm(y)
        assert abs(explicit - fused) <= 1e-10 * max(1.0, explicit)


def test_mirror_matches_stock_step_sequence():
    # (b) beta==0 mirror IS the stock loop (same iterates => same steps)
    H = _shifted_gramian(g=12)
    v0 = np.ones(H.shape[0], dtype=np.complex128)
    _, stock = _stock_steps(H, v0, 40)
    v = v0 / np.linalg.norm(v0)
    mirror_steps = []
    for _ in range(40):
        y = H @ v
        ssq_y = float(np.vdot(y, y).real)
        dot = float(np.vdot(y, v).real)
        mirror_steps.append(math.sqrt(max(0.0, 2.0 - 2.0 * dot / math.sqrt(ssq_y))))
        v = y / math.sqrt(ssq_y)
    assert np.allclose(stock, mirror_steps, rtol=1e-9, atol=1e-12)


def test_mirror_recovers_dominant_eigenvector():
    # (c)
    H = _shifted_gramian(g=16)
    ref, _ = _dominant(H)
    v0 = np.ones(H.shape[0])
    v, _ = _mirror_fused(H, v0, 3000, tol=1e-9)
    align = abs(np.vdot(ref, v))
    assert align > 1.0 - 1e-6, f"alignment {align}"


def test_momentum_fewer_matvecs_at_small_gap():
    # (d) g=24 -> n=576, Fiedler-limited gap; momentum should cut matvecs
    # substantially (theory: sqrt(gap) vs gap contraction).
    H = _shifted_gramian(g=24)
    v0 = np.ones(H.shape[0])
    tol = 1e-8
    _, used_plain = _mirror_fused(H, v0, 20000, tol, momentum=False)
    v_m, used_mom = _mirror_fused(H, v0, 20000, tol, momentum=True)
    ref, _ = _dominant(H)
    align = abs(np.vdot(ref, v_m / np.linalg.norm(v_m)))
    assert align > 1.0 - 1e-5, f"momentum alignment {align}"
    # adaptive-Chebyshev regime: measured 6.6x at n=576 (mirror); 3x is the
    # loose CI floor -- a regression to frozen-beta (~2x) must fail this.
    assert used_mom < 0.34 * used_plain, (used_mom, used_plain)


def test_fused_off_on_cpu():
    # (e) GPU-gated: CPU never takes the fused path even with the env set
    assert Operators._FUSED_EIG is False
    H = _shifted_gramian(g=8).astype(np.complex128)
    Operators.eig_reset()
    omega = Operators.Eigensolver(H, 50, tol=1e-7)
    assert omega.shape == (H.shape[0], 1, 1)
    assert np.allclose(np.abs(omega), 1.0, atol=1e-6)


def test_momentum_scaling_with_n():
    # The adaptive-Chebyshev win must GROW with problem size (sqrt-gap law);
    # frozen-beta or a broken inversion is flat (~2x) and fails here.
    gains = []
    for g in (24, 48):
        H = _shifted_gramian(g=g)
        v0 = np.ones(H.shape[0])
        _, p = _mirror_fused(H, v0, 100000, 1e-8, momentum=False)
        _, m = _mirror_fused(H, v0, 100000, 1e-8, momentum=True)
        gains.append(p / m)
    assert gains[0] > 3.0, gains
    assert gains[1] > gains[0] * 1.5, gains


def test_no_spurious_stop_with_f32_iterates():
    # Regression for the A100 finding: with v stored in complex64, ~5e-8 of
    # norm drift made the OLD formulas (which assumed ||v||=1) clamp step/res
    # to exact 0 and fire the tol-independent machine-exact stops at true
    # distance ~1e-2. With the true-cosine formulas, tol=0 must never stop
    # before the budget on an unconverged f32 iterate.
    H = _shifted_gramian(g=16)
    n = H.shape[0]
    v = (np.ones(n) / math.sqrt(n)).astype(np.complex64)
    ctl = Operators._EigAdapt(0.0, momentum=False)
    budget = 600
    for it in range(budget):
        # accumulate the four sums in float64 exactly as the GPU kernel does
        # (an f32-accumulated ssq_v breaks Cauchy-Schwarz by ~1e-7 and would
        # test a bug the kernel doesn't have)
        v64 = v.astype(np.complex128)
        y = H @ v64
        ssq_y = float(np.vdot(y, y).real)
        dot = float(np.vdot(y, v64).real)
        ssq_v = float(np.vdot(v64, v64).real)
        ctl.feed([(ssq_y, ssq_y, dot, ssq_v)])
        if ctl.stop:
            # the ONLY legitimate tol=0 stop is the machine-exact one at the
            # f32 fixed point -- i.e. at genuine convergence. Stopping while
            # still far away is the norm-drift bug this test guards against.
            w, V = np.linalg.eigh(H.toarray())
            a = abs(np.vdot(V[:, -1], v64 / np.linalg.norm(v64)))
            assert a > 1.0 - 1e-6, f"spurious stop at it={it}, align={a}"
            break
        v = (y / math.sqrt(ssq_y)).astype(np.complex64)  # f32 storage drift
    else:
        raise AssertionError("no fixed point reached in budget")
