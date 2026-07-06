"""Batched GPU AP v2 — SYNC ON. Completes the port: the Gramian (sharpy's contribution) is
assembled WITHOUT holding the full frames stack, by calling the raw zQQz kernel per
locality-ordered PAIR-CHUNK on a frame SUBSET (the CPU prototype frame_batch_test.py proved
this bit-exact). Post-projection frames live on HOST; each chunk streams only the ~unique
frames it references (peak sync frames = O(degree)); inorm_split is recomputed per chunk via
split_cuda (never stored at O(nf)).

The raw kernel (Operators.zQQz_raw_kernel, "dotp") takes col/row/dx/dy as ARGS, so a sub-plan
= (frame subset, local col/row, chunk dx/dy) reproduces the production Gramiam_calc_cuda
exactly. Validates in one run: (1) chunked-H == production H; (2) batched recon == single-
batch recon (sync on); (3) runs at 65536 frames (production OOMs).

Run on a GPU node: source ~/sharpy-venv/bin/activate; python -u batched_gpu_sync_harness.py
env: NX(128) MAXIT(5) NUMITER(10) B(1024) PCHUNK(4096) EQ_NNX(64) BIG_NNX(256)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cupy as cp
import Operators
from Operators import (make_probe, make_translations, Gramiam_plan, Project_data,
                       Precondition_calc, Gramiam_calc_cuda, zQQz_raw_kernel, eig_reset)
from wrap_ops import split_cuda, overlap_cuda

NX = int(os.environ.get("NX", 128))
MAXIT = int(os.environ.get("MAXIT", 5))
NUMITER = int(os.environ.get("NUMITER", 10))
B = int(os.environ.get("B", 1024))
PCHUNK = int(os.environ.get("PCHUNK", 4096))
EQ_NNX = int(os.environ.get("EQ_NNX", 64))
BIG_NNX = [int(x) for x in os.environ.get("BIG_NNX", "256").split()]
pool = cp.get_default_memory_pool()
_KERNEL = cp.RawKernel(zQQz_raw_kernel, "dotp", jitify=True, options=("--std=c++17",))


def gpu_used_gb():
    return pool.used_bytes() / 1e9


def free():
    pool.free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks()


def simulate(nx, nnx, Dx, gen_B):
    ny, nny, Dy = nx, nnx, Dx
    nframes = nnx * nny
    illum, _ = make_probe(nx, ny, r1=0.075, r2=0.255, fx=+20, fy=-20)
    tx, ty = make_translations(Dx, Dy, nnx, nny, nnx * Dx, nny * Dy)
    Nx = int(np.ceil(float(cp.max(tx) - cp.min(tx)))); Ny = Nx
    np.random.seed(42)
    density = np.random.rand(Nx, Ny).astype(np.float32)
    truth = cp.array(np.exp(0.69 * (-1 + 0.5j) * density).astype(np.complex64))
    illum = cp.array(illum, dtype=cp.complex64)
    translations = (tx + 1j * ty).astype(cp.complex64)
    data = cp.zeros((nframes, nx, ny), dtype=cp.float32)
    for s in range(0, nframes, gen_B):
        e = min(nframes, s + gen_B)
        fb = cp.zeros((e - s, nx, ny), dtype=cp.complex64)
        split_cuda(truth, fb, translations[s:e], illum)
        data[s:e] = (cp.abs(cp.fft.fft2(fb)) ** 2).astype(cp.float32)
    del fb, truth, density; free()
    return illum, translations, data, Nx, Ny


def compute_norm(illum, translations, Nx, Ny):
    norm = cp.zeros((Nx, Ny), dtype=cp.complex64)
    overlap_cuda(norm, 0, translations, illum)
    reg = 1e-8 * float(cp.max(cp.abs(norm)))
    return norm, reg


def gramian_chunked(F, frames_norm, translations, illum, norm_reg_inv, Gplan, nx, chunk):
    """Assemble H from post-projection frames F (host np or device cp) by pair-chunks. Streams
    only the frames each chunk touches to device; recomputes inorm_split[U] via split_cuda.
    Returns (H, peak_frames)."""
    col = cp.asnumpy(Gplan["col"]).astype(np.int64); row = cp.asnumpy(Gplan["row"]).astype(np.int64)
    dx = Gplan["dx"]; dy = Gplan["dy"]; bw = int(Gplan["bw"]); nnz = col.size
    val = cp.zeros((nnz, 1), dtype=cp.complex64)
    order = np.argsort(col, kind="stable")                       # frame-locality
    peak = 0
    fn = frames_norm.astype(cp.complex64)
    for s in range(0, nnz, chunk):
        pidx = order[s:min(nnz, s + chunk)]
        c = col[pidx]; r = row[pidx]
        U = np.unique(np.concatenate([c, r])); peak = max(peak, U.size)
        pos = np.empty(int(U.max()) + 1, np.int64); pos[U] = np.arange(U.size)
        Ud = cp.asarray(U)
        F_sub = cp.asarray(F[U]).astype(cp.complex64)            # host/device -> device (<=2*chunk)
        fn_sub = fn[Ud]
        inorm_sub = cp.zeros((U.size, nx, nx), dtype=cp.complex64)
        split_cuda(norm_reg_inv, inorm_sub, translations[Ud], 0)  # = inorm_split[U]
        cl = cp.asarray(pos[c]).astype(int); rl = cp.asarray(pos[r]).astype(int)
        pidxd = cp.asarray(pidx)
        vch = cp.zeros((pidx.size, 1), dtype=cp.complex64)
        _KERNEL((int(pidx.size),), (128,),
                (vch, F_sub, fn_sub, illum, inorm_sub, cl, rl, dx[pidxd], dy[pidxd],
                 bw, int(pidx.size), nx, nx))
        val[pidxd] = vch
    return Gplan["val2H"](val.ravel()), peak


def project_all(img, translations, illum, data, B):
    """One AP data-projection pass -> post-projection frames on HOST + frames_norm (device)."""
    nf, nx, ny = data.shape
    F = np.empty((nf, nx, ny), dtype=np.complex64)
    fn = cp.empty(nf, dtype=cp.complex64)
    for s in range(0, nf, B):
        e = min(nf, s + B)
        fb = cp.zeros((e - s, nx, ny), dtype=cp.complex64)
        split_cuda(img, fb, translations[s:e], illum)
        fb, _ = Project_data(fb, data[s:e])
        fn[s:e] = Precondition_calc(fb, bw=0)
        F[s:e] = cp.asnumpy(fb)
    return F, fn


def ap_sync_on_fast(translations, illum, data, norm, reg, Nx, Ny, maxit, B, Gplan):
    """FAST path for frame counts that FIT on device: keep F DEVICE-resident, batch only
    split/project/overlap (drops the frames_old copy + FFT-temp + framesl/framesr multiplier),
    and use the production single-shot Gramiam_calc_cuda (no per-chunk Python loop, no PCIe).
    Peak ~ data + F + inorm_split + O(B) = ~3*O(nf) vs production ~6*O(nf)."""
    nf, nx, ny = data.shape
    norm_reg_inv = (1.0 / (norm + reg)).astype(cp.complex64)
    inorm_full = cp.zeros((nf, nx, ny), dtype=cp.complex64)
    split_cuda(norm_reg_inv, inorm_full, translations, 0)               # O(nf), once
    eig_reset()
    img = cp.ones((Nx, Ny), dtype=cp.complex64)
    F = cp.zeros((nf, nx, ny), dtype=cp.complex64)                      # DEVICE-resident
    fn = cp.zeros(nf, dtype=cp.complex64)
    for _ in range(maxit):
        for s in range(0, nf, B):                                      # PASS A: split+project
            e = min(nf, s + B)
            fb = cp.zeros((e - s, nx, ny), dtype=cp.complex64)
            split_cuda(img, fb, translations[s:e], illum)
            fb, _ = Project_data(fb, data[s:e])
            F[s:e] = fb; fn[s:e] = Precondition_calc(fb, bw=0)
        H = Gramiam_calc_cuda(F, Gplan, illum, inorm_full, fn)         # single-shot production
        omega = Operators.Eigensolver(H, NUMITER).reshape(nf, 1, 1)
        img0 = cp.zeros((Nx, Ny), dtype=cp.complex64)
        for s in range(0, nf, B):                                      # PASS B: omega + overlap
            e = min(nf, s + B)
            overlap_cuda(img0, F[s:e] * omega[s:e], translations[s:e], illum)
        img = img0 / (norm + reg)
    return img


def ap_sync_on(translations, illum, data, norm, reg, Nx, Ny, maxit, B, chunk, Gplan):
    """Frame-batched AP with SYNC on (chunked Gramian). B=nf & chunk=nnz -> single-batch."""
    nf, nx, ny = data.shape
    norm_reg_inv = (1.0 / (norm + reg)).astype(cp.complex64)
    eig_reset()
    img = cp.ones((Nx, Ny), dtype=cp.complex64)
    peak_f = 0
    for _ in range(maxit):
        F, fn = project_all(img, translations, illum, data, B)          # host store + norms
        H, pk = gramian_chunked(F, fn, translations, illum, norm_reg_inv, Gplan, nx, chunk)
        peak_f = max(peak_f, pk)
        omega = Operators.Eigensolver(H, NUMITER)                        # (nf,1,1) power
        om = cp.asnumpy(omega).reshape(nf, 1, 1)
        img0 = cp.zeros((Nx, Ny), dtype=cp.complex64)
        for s in range(0, nf, B):                                        # apply omega + overlap
            e = min(nf, s + B)
            fb = cp.asarray(F[s:e]) * cp.asarray(om[s:e])
            overlap_cuda(img0, fb, translations[s:e], illum)
        img = img0 / (norm + reg)
    return img, peak_f


if __name__ == "__main__":
    Dx = int(os.environ.get("DX", NX // 4))
    print(f"NX={NX} Dx={Dx} MAXIT={MAXIT} NUMITER={NUMITER} B={B} PCHUNK={PCHUNK} | "
          f"GPU total {cp.cuda.Device().mem_info[1]/1e9:.0f} GB")

    # ── (1) H VALIDATION: chunked == production Gramiam_calc_cuda (fitting size) ──
    nnx = EQ_NNX; nf = nnx * nnx
    illum, tr, data, Nx, Ny = simulate(NX, nnx, Dx, B)
    norm, reg = compute_norm(illum, tr, Nx, Ny)
    Gplan = Gramiam_plan(tr.real, tr.imag, nf, NX, NX, Nx, Ny, bw=0)
    img = cp.ones((Nx, Ny), dtype=cp.complex64)
    F, fn = project_all(img, tr, illum, data, nf)                        # full frames (host)
    norm_reg_inv = (1.0 / (norm + reg)).astype(cp.complex64)
    F_dev = cp.asarray(F)
    inorm_full = cp.zeros((nf, NX, NX), dtype=cp.complex64)
    split_cuda(norm_reg_inv, inorm_full, tr, 0)
    H_prod = Gramiam_calc_cuda(F_dev, Gplan, illum, inorm_full, fn)      # production path
    H_chk, pk = gramian_chunked(F, fn, tr, illum, norm_reg_inv, Gplan, NX, PCHUNK)
    dH = float(cp.linalg.norm((H_prod - H_chk).data)) / (float(cp.linalg.norm(H_prod.data)) + 1e-30)
    print(f"\n== (1) CHUNKED-H vs PRODUCTION Gramiam_calc_cuda ({nf} frames, chunk={PCHUNK}) ==")
    print(f"   rel |H_chunked - H_prod| = {dH:.2e}   (peak sync frames/chunk = {pk} of {nf}; ~degree)")
    del F_dev, inorm_full, H_prod, H_chk; free()

    # ── (2) RECON EXACTNESS: streaming-batched & FAST both == single-batch, sync ON ──
    img_full, _ = ap_sync_on(tr, illum, data, norm, reg, Nx, Ny, MAXIT, nf, Gplan["col"].size, Gplan)
    img_str, pkf = ap_sync_on(tr, illum, data, norm, reg, Nx, Ny, MAXIT, min(B, nf), PCHUNK, Gplan)
    img_fast = ap_sync_on_fast(tr, illum, data, norm, reg, Nx, Ny, MAXIT, min(B, nf), Gplan)
    rl_s = float(cp.max(cp.abs(img_str - img_full)) / (cp.max(cp.abs(img_full)) + 1e-30))
    rl_f = float(cp.max(cp.abs(img_fast - img_full)) / (cp.max(cp.abs(img_full)) + 1e-30))
    print(f"\n== (2) SYNC-ON RECON EXACTNESS ({nf} frames) ==")
    print(f"   streaming-batched vs single = {rl_s:.2e} (peak sync frames {pkf}) | fast vs single = {rl_f:.2e}")
    del illum, tr, data, norm, Gplan, F, fn, img_full, img_str, img_fast; free()

    # ── (3) SPEED: FAST path (device-resident F, single-shot Gramian) vs streaming ──
    print(f"\n== (3) SYNC ON @ frame counts production OOMs (16384 cap): FAST vs STREAMING ms/iter ==")
    print(f"{'nframes':>9} {'data_GB':>8} {'mode':>10} {'status':>7} {'peak_GB':>8} {'ms/iter':>9}")
    for nnx in BIG_NNX:
        nf = nnx * nnx
        for mode in ("fast", "stream"):
            try:
                free()
                illum, tr, data, Nx, Ny = simulate(NX, nnx, Dx, B)
                norm, reg = compute_norm(illum, tr, Nx, Ny)
                Gplan = Gramiam_plan(tr.real, tr.imag, nf, NX, NX, Nx, Ny, bw=0)
                cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
                if mode == "fast":
                    img = ap_sync_on_fast(tr, illum, data, norm, reg, Nx, Ny, MAXIT, B, Gplan)
                else:
                    img, _ = ap_sync_on(tr, illum, data, norm, reg, Nx, Ny, MAXIT, B, PCHUNK, Gplan)
                cp.cuda.Stream.null.synchronize()
                ms = (time.perf_counter() - t0) / MAXIT * 1e3
                print(f"{nf:>9} {data.nbytes/1e9:>8.2f} {mode:>10} {'OK':>7} {gpu_used_gb():>8.2f} {ms:>9.1f}")
                del illum, tr, data, norm, Gplan, img; free()
            except (cp.cuda.memory.OutOfMemoryError, MemoryError) as ex:
                print(f"{nf:>9} {'-':>8} {mode:>10} {'OOM':>7}  ({type(ex).__name__})"); free()
    print("\nEXPECT: (1) H 0; (2) both batched paths == single ~1e-6; (3) FAST (device-F, single Gramian) "
          "~orders faster than STREAMING at frame counts that fit device; STREAMING wins only past device mem.")
