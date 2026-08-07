"""
Accelerators for alternating-projections (AP) ptychography -- a small, tested,
importable library (the benchmarks live in ap_accel_test.py / line_search_ap_test.py).

The workhorse is a Newton LINE SEARCH on the data misfit along any update
direction (the AP/frame step, the probe step, or the position step -- same
machinery). It is parameter-free, auto-discovers the optimal over-relaxation
(~1.9 for the AP step), and is essentially free when the iterate is kept in
Fourier (data) space: the direction is already there and every trial alpha is
pointwise.

Given a current model Fourier-frame Z and a step direction D (both = F of the
real-space quantities), and a = sqrt(data):

    minimize  J(alpha) = sum_q ( |Z_q + alpha D_q| - a_q )^2

via 2 Newton steps from alpha=1 using the Wirtinger (CR-calculus) directional
derivative AND curvature:

    W = Z + alpha D ,  u = Re(conj(W) D)/|W| ,  r = |W| - a
    J'  = sum 2 r u
    J'' = sum 2 [ u^2 + r (|D|^2 - u^2)/|W| ]
    alpha <- alpha - J'/J''

For the AP/frame step:   Z=F z,        D=F(P_Q P_a z) - F z.
For the probe step:      Z=F(O_n P),   D=F(O_n (P_new - P)).   (n over frames)

`AndersonAccelerator` is an optional DIIS history-extrapolator for the AP
fixed-point map; it is Tikhonov-regularized and MUST be safeguarded by the
caller (accept its candidate only if the data misfit beats the line-search
step), because raw Anderson can diverge on the nonlinear magnitude projection.

CPU/GPU agnostic (uses Operators.xp).
"""

import numpy as np

from Operators import xp


def data_misfit(W, a, weights=None):
    """Amplitude data misfit sum_q (|W_q| - a_q)^2 (W in Fourier, a=sqrt(I)).

    weights: optional per-pixel detector weights in {0,1} (or [0,1]),
    broadcastable against W (see Operators.Project_data) -- each term is
    multiplied by its weight, so W=0 (dead / zero-padded) pixels never
    contribute. weights=None reproduces the unweighted result exactly."""
    r2 = (xp.abs(W) - a) ** 2
    if weights is not None:
        r2 = weights * r2
    return float(xp.sum(r2, dtype=xp.float64))


def line_search(Z, D, a, n_newton=2, alpha0=1.0, weights=None):
    """Newton line search for the step alpha minimizing the amplitude data misfit
    along direction D (all in Fourier space). Returns alpha. See module docstring.

    weights: optional per-pixel detector weights (see Operators.Project_data),
    broadcastable against Z -- the misfit terms (and hence J', J'') are masked
    by the weights, so W=0 pixels (unmeasured: a holds padding there, not
    data) cannot steer alpha. weights=None is exactly the unweighted search.
    """
    tiny = 1e-12
    Dabs2 = xp.abs(D) ** 2
    alpha = alpha0
    for _ in range(n_newton):
        W = Z + alpha * D
        Wabs = xp.abs(W) + tiny
        r = Wabs - a
        u = xp.real(xp.conj(W) * D) / Wabs
        g = r * u
        h = u ** 2 + r * (Dabs2 - u ** 2) / Wabs
        if weights is not None:
            g = weights * g
            h = weights * h
        # accumulate in float64: on GPU these reductions run over ~1e6 float32
        # terms and float32 summation loses the curvature to cancellation,
        # making the Newton step (and the AP iteration) diverge. CPU numpy uses
        # pairwise summation and happens to survive; force float64 everywhere.
        Jp = 2.0 * xp.sum(g, dtype=xp.float64)
        Jpp = 2.0 * xp.sum(h, dtype=xp.float64)
        alpha = alpha - Jp / (Jpp + tiny)
    return float(alpha)


class AndersonAccelerator:
    """Tikhonov-regularized Anderson (DIIS) for a fixed-point map g(x) ~ x.

    Usage per iteration (caller supplies the current iterate x and g=g(x), and
    SAFEGUARDS the result against a known-good fallback, e.g. the line-search
    step, using data_misfit):

        cand = acc.step(x, g)
        x = cand if data_misfit(F(cand), a) < data_misfit(F(x_ls), a) else x_ls
    """

    def __init__(self, depth=4, reg=1e-3):
        self.m = depth
        self.reg = reg
        self._F = []   # residuals f = g(x) - x  (flattened)
        self._G = []   # maps g(x)               (flattened)

    def reset(self):
        self._F.clear(); self._G.clear()

    def step(self, x, g):
        shape = x.shape
        self._F.append((g - x).ravel())
        self._G.append(g.ravel())
        if len(self._F) > self.m + 1:
            self._F.pop(0); self._G.pop(0)
        if len(self._F) < 2:
            return g
        dF = xp.stack([self._F[i + 1] - self._F[i] for i in range(len(self._F) - 1)], 1)
        dG = xp.stack([self._G[i + 1] - self._G[i] for i in range(len(self._G) - 1)], 1)
        A = dF.conj().T @ dF
        scale = float(xp.trace(A).real)
        if scale < 1e-30:                       # converged/degenerate history -> no extrapolation
            return g
        A = A + self.reg * (scale / A.shape[0]) * xp.eye(A.shape[0], dtype=A.dtype)
        gamma = xp.linalg.solve(A, dF.conj().T @ self._F[-1])
        return (self._G[-1] - dG @ gamma).reshape(shape)
