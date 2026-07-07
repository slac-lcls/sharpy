"""pytest suite for group_sync -- shift/tilt/phase synchronization on the overlap graph.

  pytest -q group_sync_test.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import group_sync as gs


def _graph(N=260, seed=0, radius=9.0):
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 24 * np.pi, N); r = np.linspace(0, 40, N)
    pos = np.c_[r * np.cos(ang), r * np.sin(ang)]
    return pos, gs.build_overlap_graph(pos, radius), rng


def test_overlap_graph_is_sparse_and_local():
    pos, edges, _ = _graph()
    assert 0 < len(edges) < len(pos) ** 2 // 4                 # sparse, not O(N^2)
    for k, l in edges:
        assert np.hypot(*(pos[k] - pos[l])) < 9.0              # only nearby frames


def test_translation_sync_recovers_up_to_global():
    pos, edges, rng = _graph()
    N = len(pos); x = rng.normal(0, 1.0, (N, 2))
    d = np.array([x[k] - x[l] for k, l in edges]) + rng.normal(0, 0.05, (len(edges), 2))
    xh = gs.sync_translation(edges, d, N)
    err = (xh - xh.mean(0)) - (x - x.mean(0))                  # both up to a global const
    assert np.sqrt((err ** 2).sum(1).mean()) < 0.1


def test_phase_sync_recovers_up_to_global():
    pos, edges, rng = _graph()
    N = len(pos); th = rng.uniform(-np.pi, np.pi, N)
    dt = np.angle(np.exp(1j * (np.array([th[k] - th[l] for k, l in edges]) + rng.normal(0, 0.05, len(edges)))))
    thh = gs.sync_phase(edges, dt, N)
    d = np.angle(np.exp(1j * (thh - th))); d = d - d.mean()
    assert np.sqrt((d ** 2).mean()) < 0.1


def test_heisenberg_all_channels_and_beats_phase_only():
    pos, edges, rng = _graph()
    N = len(pos); E = len(edges)
    s = rng.normal(0, 1.0, (N, 2)); t = rng.normal(0, 0.1, (N, 2)); th = rng.uniform(-np.pi, np.pi, N)
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
    dS = np.array([s[k] - s[l] for k, l in edges]) + rng.normal(0, 0.12, (E, 2))
    dT = np.array([t[k] - t[l] for k, l in edges]) + rng.normal(0, 0.02, (E, 2))
    dTh = np.angle(np.exp(1j * (np.array([th[k] - th[l] + 0.5 * (cross(s[k], t[k]) - cross(s[l], t[l]))
                                          for k, l in edges]) + rng.normal(0, 0.12, E))))
    sh, tht, thh = gs.sync_heisenberg(edges, dS, dT, dTh, N)
    rms = lambda a, b: np.sqrt((((a - a.mean(0)) - (b - b.mean(0))) ** 2).sum(1).mean())
    # all three channels recovered well below the injected magnitude (global consistency denoises)
    assert rms(sh, s) < 0.2 and rms(tht, t) < 0.06
    # phase-only sync leaves shift completely uncorrected (structurally can't touch it)
    assert rms(np.zeros_like(s), s) > 4 * rms(sh, s)
