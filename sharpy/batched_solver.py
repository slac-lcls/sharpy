"""Frame-batched drop-in for Solvers.Alternating_projections_c (GPU).

ADDITIVE module (does NOT touch Solvers.py) so it can't clobber in-progress edits there;
promote into Solvers later. Same call signature + one extra `frame_batch` (B) argument, same
return `(img, frames, illumination, residuals)`.

Why: production Alternating_projections_c materialises the whole frames stack every iter
(frames=zeros(nf,nx,ny); split all; Project all; frames_old copy; sync; overlap all) → a
~5-12× working-set multiplier on O(nf·nx²) → OOM at nx=128 above ~16k frames (memory_limit
baseline). This keeps object+data+probe resident and processes frames in batches of B:
  PASS A  split_cuda→Project_data per batch → DEVICE-resident F (+ frames_norm)
  SYNC    H = Gramiam_calc_cuda(F, ...)  (single-shot, == production) → Eigensolver → omega
  PASS B  overlap_cuda(F*omega) per batch → object
Drops the frames_old copy + the full-stack FFT temp + framesl/framesr → peak ≈ data + F +
inorm + O(B) (~5× data) vs production ~10-12× → runs where production OOMs, at ~production
speed (fast path; validated `batched_gpu_sync_harness.py`: 315 ms/iter @65536 fr nx=128,
bit-exact H vs production, recon 1e-5 vs single-batch).

Faithful to production for the RECON (img) and the truth/data residuals; residuals[:,2]
(‖frames-frames_old‖, the ePIE step size) is NOT computed here — that copy is exactly what
batching avoids — left 0. GPU-only (device-resident F); refine_illumination not supported yet.
"""
import numpy as np
import cupy as cp
from Operators import (Project_data, Precondition_calc, Gramiam_calc_cuda, Eigensolver,
                       mse_calc, eig_reset, xp, GPU)

reg0 = 1e-8


def Alternating_projections_batched_c(
    sync, img, Gramiam, illumination, translations_x, translations_y,
    overlap_cuda, split_cuda, frames_data, refine_illumination, maxiter,
    normalization=None, img_truth=None, residuals_interval=1, sync_interval=1,
    num_iter=5, frame_batch=1024,
):
    assert GPU, "batched solver is the GPU device-resident-F path"
    assert not refine_illumination, "refine_illumination not supported in the batched path yet"
    nframes, nx, ny = frames_data.shape
    B = frame_batch if (frame_batch and frame_batch > 0) else nframes
    illumination_start = illumination
    translations = (translations_x + 1j * translations_y).astype(cp.complex64)

    # ── normalization + inorm (stack-free: overlap_cuda(.,0,.) / split_cuda(.,.,.,0)) ──
    if normalization is None:
        normalization = cp.zeros(img.shape, dtype=cp.complex64)
        overlap_cuda(normalization, 0, translations, illumination_start + 0)
    reg = reg0 * float(xp.max(xp.abs(normalization)))
    if sync:
        inorm_split = cp.zeros(frames_data.shape, dtype=cp.complex64)
        split_cuda(1.0 / (normalization + reg), inorm_split, translations, 0)
        eig_reset()

    # ── residual bookkeeping (matches Alternating_projections_c) ──
    nresiduals = int(np.ceil(maxiter / residuals_interval))
    residuals = xp.zeros((nresiduals, 4), dtype=xp.float32)
    frames_norm_sum = xp.linalg.norm(xp.sqrt(frames_data))
    frames_norm_r = frames_norm_sum / xp.sqrt(xp.prod(xp.array(frames_data.shape[-2:])))
    if img_truth is not None:
        nrm_truth = xp.linalg.norm(img_truth)

    F = cp.zeros(frames_data.shape, dtype=cp.complex64)          # DEVICE-resident frames
    frames_norm = cp.zeros(nframes, dtype=cp.complex64)

    for ii in range(maxiter):
        cr = (ii % residuals_interval == 0)
        mse_acc = 0.0
        # PASS A: split + data-project per batch → F (+ per-frame norm for sync)
        for s in range(0, nframes, B):
            e = min(nframes, s + B)
            fb = cp.zeros((e - s, nx, ny), dtype=cp.complex64)
            split_cuda(img, fb, translations[s:e], illumination_start)
            fb, mse = Project_data(fb, frames_data[s:e], compute_residuals=cr)
            if cr and mse is not None:
                mse_acc += float(mse)
            F[s:e] = fb
            if sync:
                frames_norm[s:e] = Precondition_calc(fb, bw=Gramiam["bw"])
        # SYNC: single-shot Gramian (== production Gramiam_calc_cuda) → omega
        omega = None
        if sync and (ii % sync_interval == 0):
            H = Gramiam_calc_cuda(F, Gramiam, illumination_start, inorm_split, frames_norm)
            omega = Eigensolver(H, num_iter).reshape(nframes, 1, 1)
        # PASS B: (apply omega) + overlap per batch → object
        img0 = cp.zeros(img.shape, dtype=cp.complex64)
        for s in range(0, nframes, B):
            e = min(nframes, s + B)
            fb = F[s:e] if omega is None else F[s:e] * omega[s:e]
            overlap_cuda(img0, fb, translations[s:e], illumination_start)
        img = img0 / (normalization + reg)
        # residuals (recon-side exact; step-size residuals[:,2] intentionally skipped)
        if cr:
            residuals[ii // residuals_interval, 1] = mse_acc
            if img_truth is not None:
                residuals[ii // residuals_interval, 0] = mse_calc(img_truth, img)

    residuals[:, 1] /= frames_norm_sum
    if img_truth is not None:
        residuals[:, 0] /= nrm_truth
        residuals[:, 3] = 1.0 / (residuals[:, 0] + 1e-30)
    return img, F, illumination_start, residuals


# ── validation __main__: batched == production Alternating_projections_c, + past-cap ──
if __name__ == "__main__":
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import matplotlib
    matplotlib.use("Agg")                                        # headless: no plt.show() hang
    import Solvers
    from Operators import make_probe, make_translations, Gramiam_plan
    from wrap_ops import split_cuda, overlap_cuda

    NX = int(os.environ.get("NX", 128)); Dx = NX // 4
    MAXIT = int(os.environ.get("MAXIT", 5)); NUMITER = int(os.environ.get("NUMITER", 10))
    B = int(os.environ.get("B", 1024))
    EQ_NNX = int(os.environ.get("EQ_NNX", 64))
    BIG_NNX = [int(x) for x in os.environ.get("BIG_NNX", "256").split()]
    pool = cp.get_default_memory_pool()

    def simulate(nx, nnx, Dx, gen_B):
        nframes = nnx * nnx
        illum, _ = make_probe(nx, nx, r1=0.075, r2=0.255, fx=+20, fy=-20)
        tx, ty = make_translations(Dx, Dx, nnx, nnx, nnx * Dx, nnx * Dx)
        Nx = int(np.ceil(float(cp.max(tx) - cp.min(tx))))
        np.random.seed(42)
        truth = cp.array(np.exp(0.69 * (-1 + 0.5j) * np.random.rand(Nx, Nx).astype(np.float32)).astype(np.complex64))
        illum = cp.array(illum, dtype=cp.complex64)
        tr = (tx + 1j * ty).astype(cp.complex64)
        data = cp.zeros((nframes, nx, nx), dtype=cp.float32)
        for s in range(0, nframes, gen_B):
            e = min(nframes, s + gen_B)
            fb = cp.zeros((e - s, nx, nx), dtype=cp.complex64)
            split_cuda(truth, fb, tr[s:e], illum)
            data[s:e] = (cp.abs(cp.fft.fft2(fb)) ** 2).astype(cp.float32)
        del fb; pool.free_all_blocks()
        return truth, illum, tx, ty, data, Nx

    # (1) faithfulness: batched (single + batched) vs production Alternating_projections_c
    nnx = EQ_NNX; nf = nnx * nnx
    truth, illum, tx, ty, data, Nx = simulate(NX, nnx, Dx, B)
    Gplan = Gramiam_plan(tx, ty, nf, NX, NX, Nx, Nx, bw=0)
    imgP, _, _, resP = Solvers.Alternating_projections_c(
        True, cp.ones((Nx, Nx), cp.complex64), Gplan, illum, tx, ty, overlap_cuda, split_cuda,
        data, False, MAXIT, normalization=None, img_truth=truth, residuals_interval=MAXIT,
        sync_interval=1, num_iter=NUMITER)
    imgS, _, _, _ = Alternating_projections_batched_c(
        True, cp.ones((Nx, Nx), cp.complex64), Gplan, illum, tx, ty, overlap_cuda, split_cuda,
        data, False, MAXIT, None, truth, MAXIT, 1, NUMITER, frame_batch=nf)
    imgB, _, _, resB = Alternating_projections_batched_c(
        True, cp.ones((Nx, Nx), cp.complex64), Gplan, illum, tx, ty, overlap_cuda, split_cuda,
        data, False, MAXIT, None, truth, MAXIT, 1, NUMITER, frame_batch=B)
    rS = float(cp.max(cp.abs(imgS - imgP)) / (cp.max(cp.abs(imgP)) + 1e-30))
    rB = float(cp.max(cp.abs(imgB - imgP)) / (cp.max(cp.abs(imgP)) + 1e-30))
    print(f"\n== BATCHED SOLVER vs PRODUCTION Alternating_projections_c (nx={NX}, {nf} frames) ==")
    print(f"   batched single-batch vs production = {rS:.2e} | batched B={B} vs production = {rB:.2e}")
    print(f"   truth-NMSE  production {float(resP[-1,0]):.4f}  batched {float(resB[-1,0]):.4f}")
    del truth, illum, data, Gplan, imgP, imgS, imgB; pool.free_all_blocks()

    # (2) past-cap: batched runs where production OOMs
    print(f"\n== PAST-CAP: batched Alternating_projections_batched_c (production nx=128 caps 16384) ==")
    print(f"{'nframes':>9} {'data_GB':>8} {'status':>7} {'ms/iter':>9} {'truthNMSE':>10}")
    for nnx in BIG_NNX:
        nf = nnx * nnx
        try:
            pool.free_all_blocks()
            truth, illum, tx, ty, data, Nx = simulate(NX, nnx, Dx, B)
            Gplan = Gramiam_plan(tx, ty, nf, NX, NX, Nx, Nx, bw=0)
            cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
            img, _, _, res = Alternating_projections_batched_c(
                True, cp.ones((Nx, Nx), cp.complex64), Gplan, illum, tx, ty, overlap_cuda,
                split_cuda, data, False, MAXIT, None, truth, MAXIT, 1, NUMITER, frame_batch=B)
            cp.cuda.Stream.null.synchronize()
            ms = (time.perf_counter() - t0) / MAXIT * 1e3
            print(f"{nf:>9} {data.nbytes/1e9:>8.2f} {'OK':>7} {ms:>9.1f} {float(res[-1,0]):>10.4f}")
            del truth, illum, data, Gplan, img; pool.free_all_blocks()
        except (cp.cuda.memory.OutOfMemoryError, MemoryError) as ex:
            print(f"{nf:>9} {'-':>8} {'OOM':>7}  ({type(ex).__name__})"); pool.free_all_blocks()
    print("\nEXPECT: batched == production ~1e-6 (recon); runs past production's OOM cap at ~production speed.")
