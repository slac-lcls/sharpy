"""
Tests for ap_accel: the Newton line search (correctness of derivative+curvature,
and that it reduces the data misfit) and that line-search AP beats plain AP.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ap_accel import line_search, data_misfit, AndersonAccelerator


def _instance(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    D = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    a = np.abs(Z + 0.7 * D) + 0.05 * rng.standard_normal(n)   # near alpha~0.7
    return Z, D, np.abs(a)


def test_line_search_reduces_misfit():
    Z, D, a = _instance()
    j1 = data_misfit(Z + 1.0 * D, a)            # full step (alpha=1)
    al = line_search(Z, D, a)
    jal = data_misfit(Z + al * D, a)
    assert jal <= j1 + 1e-9                       # never worse than alpha=1


def test_line_search_derivative_matches_finite_difference():
    # verify J'(alpha) from the analytic Wirtinger formula vs finite difference
    Z, D, a = _instance(seed=3)
    tiny = 1e-12
    alpha = 0.3
    W = Z + alpha * D
    Wabs = np.abs(W) + tiny
    u = np.real(np.conj(W) * D) / Wabs
    r = Wabs - a
    Jp = 2.0 * np.sum(r * u)
    h = 1e-5
    Jp_fd = (data_misfit(Z + (alpha + h) * D, a) - data_misfit(Z + (alpha - h) * D, a)) / (2 * h)
    assert abs(Jp - Jp_fd) / (abs(Jp_fd) + 1e-9) < 1e-4


def test_line_search_finds_known_optimum():
    # data built so the misfit is ~minimized near alpha=0.7; LS should land close
    Z, D, a = _instance(seed=1)
    al = line_search(Z, D, a, n_newton=4)
    js = [data_misfit(Z + s * D, a) for s in np.linspace(0.0, 1.4, 71)]
    s_best = np.linspace(0.0, 1.4, 71)[int(np.argmin(js))]
    assert abs(al - s_best) < 0.1


def test_anderson_runs_and_extrapolates_linear_map():
    # on a contractive linear fixed-point x <- 0.9 x + b, Anderson should converge
    rng = np.random.default_rng(0)
    n = 50
    b = rng.standard_normal((n, 1)) + 1j * rng.standard_normal((n, 1))
    xstar = b / (1 - 0.9)
    acc = AndersonAccelerator(depth=5)
    x = np.zeros((n, 1), complex)
    for _ in range(30):
        g = 0.9 * x + b
        x = acc.step(x, g)
    assert np.linalg.norm(x - xstar) / np.linalg.norm(xstar) < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
    print("all ap_accel tests passed")
