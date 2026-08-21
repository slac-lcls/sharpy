"""Fetch-normalization validation (run on a GPU node from sharpy/sharpy): Gramiam_calc_cuda dispatched
with the OBJECT-SIZED nu canvas (fetch path, dotp_fetch kernel) must reproduce the frame-stack path
(dotp kernel) BIT-EXACTLY -- same complex64 values loaded at the same physical pixels, same reduction
order. Also times both and reports the eliminated stack memory. Origins/wrap conventions come from
Gramiam_plan (map_frames corner semantics), so a convention drift shows up here as a hard FAIL.

    config.GPU=True required. PASS criterion: H data arrays bit-identical for all three geometries.
"""
import numpy as np
import config; config.GPU = True
import cupy as cp
import Operators as ops
xp = cp


def run_case(g, nxf, step, seed=0, label=""):
    rng = np.random.default_rng(seed)
    Ny = g * step; Nx = g * step
    tyg, txg = np.meshgrid(np.arange(g) * step, np.arange(g) * step, indexing="ij")
    tx = xp.asarray(txg.ravel().astype(float)); ty = xp.asarray(tyg.ravel().astype(float))
    nframes = int(tx.size)
    mapid = ops.map_frames(tx, ty, nxf, nxf, Nx, Ny)
    img = xp.asarray((rng.standard_normal((Ny, Nx)) + 1j * rng.standard_normal((Ny, Nx))).astype(np.complex64))
    frames = ops.Splitc(img, mapid)
    illum = xp.asarray((rng.standard_normal((nxf, nxf)) + 1j * rng.standard_normal((nxf, nxf))).astype(np.complex64))
    nu_can = xp.asarray((rng.random((Ny, Nx)) + 0.5).astype(np.float32)).astype(xp.complex64)
    norm_stack = ops.Splitc(nu_can, mapid)                    # the redundant frame-sized copy
    # TWO plans, because the path is now opt-in (fetch=) rather than inferred from normalization.ndim.
    # Do NOT collapse these back into one plan: with a single fetch=False plan both calls below take
    # the stack path and this test compares a computation to itself -- it would print BITEXACT=True
    # while validating nothing.
    plan_stack = ops.Gramiam_plan(tx, ty, nframes, nxf, nxf, Nx, Ny, bw=0)
    plan_fetch = ops.Gramiam_plan(tx, ty, nframes, nxf, nxf, Nx, Ny, bw=0, fetch=True)
    assert not plan_stack["fetch"] and plan_fetch["fetch"], "plan dispatch flags not set as intended"
    frames_norm = ops.Precondition_calc(frames)

    H1 = ops.Gramiam_calc_cuda(frames, plan_stack, illum, norm_stack, frames_norm)  # stack path (dotp)
    H2 = ops.Gramiam_calc_cuda(frames, plan_fetch, illum, nu_can,     frames_norm)  # fetch path (dotp_fetch)
    d1, d2 = H1.data, H2.data
    bitexact = bool(xp.all(d1.view(xp.int64) == d2.view(xp.int64)))
    maxd = float(xp.max(xp.abs(d1 - d2)))

    import time
    def clock(fn, n=20):
        fn(); cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        cp.cuda.runtime.deviceSynchronize()
        return (time.perf_counter() - t0) / n * 1e3

    t_stack = clock(lambda: ops.Gramiam_calc_cuda(frames, plan_stack, illum, norm_stack, frames_norm))
    t_fetch = clock(lambda: ops.Gramiam_calc_cuda(frames, plan_fetch, illum, nu_can, frames_norm))
    print("CASE %-20s frames %5d x %d^2 | BITEXACT=%s max|dH| %.3g | stack-path %.2f ms -> fetch-path "
          "%.2f ms | stack %.0f MB -> canvas %.0f MB" % (
          label, nframes, nxf, bitexact, maxd, t_stack, t_fetch,
          norm_stack.nbytes / 1e6, nu_can.nbytes / 1e6), flush=True)
    return bitexact


ok  = run_case(16, 64, 16, seed=0, label="small 256f/64px")
ok &= run_case(32, 128, 32, seed=1, label="mid 1024f/128px")
ok &= run_case(64, 128, 24, seed=2, label="big 4096f/128px")
print("FETCH-NORM DISPATCH VERDICT:", "PASS" if ok else "FAIL", flush=True)
