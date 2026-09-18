"""
Regression test for the CPU (functional Split/Overlap) probe-refinement path:
Solvers.Alternating_projections(..., refine_illumination=True, ...).

That path was dead from the Oct-2024 GPU rewrite of refine_illumination_function
until this test: the call site still used the old 6-argument signature
(TypeError), the CPU branch never computed the least-squares denominator
norm_frames (NameError), and the caller unpacked a normalization the function
no longer returns. Nothing exercised it, so it regressed silently.

The GPU production path (Alternating_projections_c / _batched_c with the
in-place split_cuda/overlap_cuda kernels) is not covered here.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

if config.GPU:
    pytest.skip(
        "CPU regression test (functional Split/Overlap); the GPU production path "
        "is Alternating_projections_c",
        allow_module_level=True,
    )

xp = np


def tonp(a):
    return np.asarray(a)


from Operators import (  # noqa: E402
    Split_Overlap_plan,
    Gramiam_plan,
    Illuminate_frames,
    refine_illumination_function,
)
import Solvers  # noqa: E402


def _gaussian_probe(nx, sigma):
    gx = np.arange(nx) - nx / 2
    X, Y = np.meshgrid(gx, gx, indexing="ij")
    return np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2)).astype(np.complex64)


def _simulation(seed=0, nx=16, K=8, step=4):
    """Tiny full-coverage periodic raster: 8x8 frames of 16x16 on a 32x32 object.

    Smooth complex object with amplitude in [0.5, 1] (no zeros, so the
    least-squares probe denominator sum_n |O_n|^2 is well conditioned) and a
    Gaussian probe. Returns everything the solver needs plus the truth.
    """
    rng = np.random.default_rng(seed)
    Nx = Ny = K * step
    g = np.arange(K) * step
    tx, ty = np.meshgrid(g, g, indexing="ij")
    tx = tx.ravel().astype(float)
    ty = ty.ravel().astype(float)

    F = np.fft.fft2(rng.standard_normal((Nx, Ny)))
    fx, fy = np.meshgrid(np.fft.fftfreq(Nx), np.fft.fftfreq(Ny), indexing="ij")
    sm = np.real(np.fft.ifft2(F * np.exp(-(fx ** 2 + fy ** 2) / (2 * 0.1 ** 2))))
    sm = (sm - sm.min()) / (sm.max() - sm.min())
    truth = ((0.5 + 0.5 * sm) * np.exp(1j * 1.5 * (sm - 0.5))).astype(np.complex64)

    probe = _gaussian_probe(nx, nx / 4.0)
    Split, Overlap = Split_Overlap_plan(tx, ty, nx, nx, Nx, Ny)
    data = (np.abs(np.fft.fft2(Split(truth) * probe[None])) ** 2).astype(np.float32)
    return dict(
        tx=tx, ty=ty, nx=nx, Nx=Nx, Ny=Ny, truth=truth, probe=probe,
        Split=Split, Overlap=Overlap, data=data,
    )


def _probe_error(illum, probe):
    """Relative probe error after removing the global complex scale (gauge)."""
    c = np.vdot(illum.ravel(), probe.ravel()) / np.vdot(illum.ravel(), illum.ravel())
    return float(np.linalg.norm(c * illum - probe) / np.linalg.norm(probe))


def _run_ap(sim, refine, probe0, maxiter=30, sync=False, gramiam=None):
    img0 = np.ones((sim["Nx"], sim["Ny"]), dtype=np.complex64)
    img, frames, illum, res = Solvers.Alternating_projections(
        sync, img0, gramiam, probe0 + 0, sim["Overlap"], sim["Split"], sim["data"],
        refine, maxiter, None, sim["truth"], 1,
    )
    return tonp(img), tonp(illum), tonp(res)


def test_refine_illumination_function_cpu_recovers_probe():
    # Consistent frames (true object x true probe) and a wrong starting probe:
    # the regularized least-squares update must return (a) ONE array of the
    # probe's shape -- the contract every caller relies on -- and (b) the true
    # probe, once the eps0 * 2**(-i) regularization is negligible.
    sim = _simulation()
    frames = Illuminate_frames(sim["Split"](sim["truth"]), sim["probe"])
    probe_bad = _gaussian_probe(sim["nx"], 1.6 * sim["nx"] / 4.0)

    out = refine_illumination_function(
        sim["truth"], probe_bad, None, frames, None, sim["Split"], sim["Overlap"],
        False, None, 20,
    )
    assert isinstance(out, np.ndarray)          # not a tuple
    assert out.shape == sim["probe"].shape
    assert np.isfinite(out).all()
    assert np.linalg.norm(out - sim["probe"]) / np.linalg.norm(sim["probe"]) < 1e-4


def test_alternating_projections_cpu_refine_illumination():
    # The reported reproduction: config.GPU=False, Split/Overlap from
    # Split_Overlap_plan, Alternating_projections(..., refine_illumination=True, ...).
    # Start from a probe 1.6x too wide; refinement must run, stay finite and
    # recover object + probe far better than the fixed wrong probe does.
    sim = _simulation()
    probe_bad = _gaussian_probe(sim["nx"], 1.6 * sim["nx"] / 4.0)
    err0 = _probe_error(probe_bad, sim["probe"])          # ~0.33

    img, illum, res = _run_ap(sim, True, probe_bad)
    assert np.isfinite(img).all() and np.isfinite(illum).all()
    assert illum.shape == sim["probe"].shape
    assert res[-1, 1] < 0.1 * res[0, 1]                   # data residual drops
    assert res[-1, 0] < 0.1                               # object NMSE (measured ~1e-2)
    assert _probe_error(illum, sim["probe"]) < 0.25 * err0  # probe fixed (measured ~1e-2)

    # control: the same loop with the wrong probe held fixed stays far worse
    _, illum_fixed, res_fixed = _run_ap(sim, False, probe_bad)
    assert res[-1, 0] < 0.25 * res_fixed[-1, 0]          # measured 1e-2 vs 4e-1
    assert np.allclose(illum_fixed, probe_bad)            # not refined


def test_alternating_projections_cpu_refine_illumination_with_sync():
    # Same, with Gramian phase synchronization on: the refreshed normalization
    # also feeds the sync weights (inormalization_split), so this path must
    # keep running after a probe update.
    sim = _simulation()
    probe_bad = _gaussian_probe(sim["nx"], 1.6 * sim["nx"] / 4.0)
    gramiam = Gramiam_plan(
        sim["tx"], sim["ty"], sim["tx"].size, sim["nx"], sim["nx"], sim["Nx"], sim["Ny"]
    )
    img, illum, res = _run_ap(sim, True, probe_bad, sync=True, gramiam=gramiam)
    assert np.isfinite(img).all() and np.isfinite(illum).all()
    assert res[-1, 1] < 0.1 * res[0, 1]
    assert res[-1, 0] < 0.1
    assert _probe_error(illum, sim["probe"]) < 0.1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("all refine_illumination CPU tests passed")
