"""Work-item 1 (efficient optimizers): ONE parameterized ptycho solver over the full
{data-step} x {consensus-step} matrix -- the UNIFICATION of the data-fitting family
(hard magnitude / proxD-AGM / proxD-IPM, Chang-Enfedaque-Marchesini SIAM 2019) and the
consensus family (none / power / eigsh / invit) into a single AP/ML loop -- measured on the
THREE axes at once: MEMORY (analytic model), SPEED (ms/iter + iters-to-quality), and
DOSE-AWARE QUALITY (low/high band error at a given photon dose; per Ophus/Varnavides
"Beyond Contrast Transfer" the noise-free transfer function overstates -- the real limiter
is photons, so quality is reported vs DOSE, not noise-free).

The two stages are ORTHOGONAL stages of one iteration -- data_step (per-frame Fourier
constraint) then sync (cross-frame phase consensus) then Overlap (object update) -- so this
subsumes sync_bandpass_test.run (sync only) and pipm_prox_noise_test.run_prox (prox only).

Deliverables this script produces:
  (1) MEMORY MODEL: the frames stack O(nf*nx^2) dominates and is the batchable/streamable
      term (gpu-memory-scaling, od2p-domain-decomposition) -- report it vs the object + plan.
  (2) MATRIX: every (data_step x sync) combo on speed + quality at a representative dose ->
      the Pareto view (which combos are worth promoting to GPU).
  (3) INTERACTION (the open science Q): does the proxD data-relaxation MOVE the sync dose
      threshold, or are the two axes independent? (prior: proxD = resolution-band regularizer
      NOT a threshold lever; sync has a BBP dose threshold -- do they compose or interact?)

CPU/numpy authoritative; structured so the GPU port = frame-batch the loop + stream the plan.
Run thread-capped + low priority:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    nice -n 19 /opt/anaconda3/bin/python3 unified_solver_test.py
env: NX(48) K(16) STEPD(4) MAXIT(150) SE(1) NUMITER(5) REPS(3)
     DATASTEPS("hard pagm:2 pipm:2") SYNCS("none power eigsh invit")
     REPDOSE(1.0) DOSELIST("3 1 0.3 0.1")  LOWTHR(0.1)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NX", "48")

import time
import numpy as np
import config
import Operators
from Operators import (Illuminate_frames, Splitc, Overlapc, Project_data, Precondition_calc,
                       synchronize_frames_c, eig_reset, xp)
import sync_bandpass_test as T

nx = T.nx
K = int(os.environ.get("K", 16))
MAXIT = int(os.environ.get("MAXIT", 150))
SE = int(os.environ.get("SE", 1))
NUMITER = int(os.environ.get("NUMITER", 5))
REPS = int(os.environ.get("REPS", 3))
LOWTHR = float(os.environ.get("LOWTHR", 0.1))
REPDOSE = float(os.environ.get("REPDOSE", 1.0))
DATASTEPS = os.environ.get("DATASTEPS", "hard pagm:2 pipm:2").split()
SYNCS = os.environ.get("SYNCS", "none power eigsh invit").split()
DOSELIST = [float(x) for x in os.environ.get("DOSELIST", "3 1 0.3 0.1").split()]


def prox_data(frames, data, step):
    """Per-frame Fourier-magnitude data step. step=('hard',_) | ('pagm',beta) | ('pipm',beta).
    beta->0 recovers hard. pAGM = Gaussian-amplitude prox; pIPM = Poisson-ML prox."""
    metric, beta = step
    if metric == "hard":
        return Project_data(frames, data)[0]
    Z = xp.fft.fft2(frames)
    a = xp.abs(Z); a = xp.where(a < 1e-12, xp.asarray(1e-12, a.dtype), a)
    r = xp.sqrt(data)
    if metric == "pagm":
        x = (r + beta * a) / (1 + beta)
    elif metric == "pipm":
        x = (beta * a + xp.sqrt(beta * beta * a * a + 4 * (1 + beta) * data)) / (2 * (1 + beta))
    else:
        raise ValueError(metric)
    return xp.fft.ifft2(Z * (x / a).astype(xp.float32))


def set_sync(sync):
    """Wire the consensus backend on the Operators module (the single dispatch point)."""
    if sync == "invit":
        Operators.SYNC_METHOD = "invit"; Operators.SYNC_MODE = "cg"
    elif sync == "eigsh":
        Operators.SYNC_METHOD = "power"; Operators.Eigensolver = T.eigsh_sync
    elif sync == "power":
        Operators.SYNC_METHOD = "power"; Operators.Eigensolver = T._POWER
    elif sync == "none":
        Operators.SYNC_METHOD = "power"; Operators.Eigensolver = T._POWER
    else:
        raise ValueError(sync)


def solve(ctx, data_step, sync, sync_every, maxit, num_iter=NUMITER):
    """The UNIFIED loop: prox_data (data family) -> optional sync (consensus family) ->
    Overlap (object update). Returns the band-error curve (maxit, 2) = [low, high]."""
    set_sync(sync)
    eig_reset(); T._eigsh_v0["v"] = None
    probe, cprobe, mapid = ctx["probe"], ctx["cprobe"], ctx["mapid"]
    do_sync = sync != "none"
    img = xp.ones((ctx["Ny"], ctx["Nx"]), dtype=xp.complex64)
    frames = Illuminate_frames(Splitc(img, mapid), probe)
    curve = []
    for it in range(maxit):
        frames = prox_data(frames, ctx["data"], data_step)
        if do_sync and sync_every and it % sync_every == 0:
            omega = synchronize_frames_c(frames, probe, ctx["frames_norm"], ctx["inorm_split"],
                                         ctx["Gramiam"], ctx["Gramiam"]["bw"], num_iter)
            frames = frames * omega
        img = Overlapc(Illuminate_frames(frames, cprobe), ctx["Nx"], ctx["Ny"], mapid) / ctx["normalization"]
        frames = Illuminate_frames(Splitc(img, mapid), probe)
        curve.append(T.band_err(ctx, img))
    Operators.SYNC_METHOD = "power"; Operators.Eigensolver = T._POWER
    return np.array(curve)


def parse_step(s):
    if ":" in s:
        m, b = s.split(":"); return (m, float(b))
    return (s, 0.0)


def label(step, sync):
    m, b = step
    d = m if m == "hard" else f"{m}{b:g}"
    return f"{d}+{sync}"


def mem_model(ctx):
    """Analytic peak-resident model (complex64 = 8 B). The frames stack is O(nf*nx^2) and is
    the batchable/streamable term; object is fixed; the Gramian plan is O(nnz)=O(nf*degree)."""
    nf = ctx["nframes"]
    frames_b = nf * nx * nx * 8                                  # THE dominant, batchable term
    # working copies in the loop (frames, framesl, framesr, fft temporaries) ~ a few x frames
    frames_working = 4 * frames_b
    obj_b = ctx["Nx"] * ctx["Ny"] * 8
    g = ctx["Gramiam"]
    nnz = int(np.asarray(g["col"]).size) if "col" in g else -1   # overlapping frame-pairs
    plan_b = (nnz * (8 + 16)) if nnz > 0 else -1                 # index + complex-ish per pair (rough)
    return dict(nf=nf, frames_b=frames_b, frames_working=frames_working, obj_b=obj_b,
                nnz=nnz, plan_b=plan_b)


def mb(b):
    return f"{b/1e6:8.2f}" if b and b > 0 else f"{'--':>8}"


def iters_to(col, thr):
    w = np.where(col < thr)[0]
    return int(w[0]) + 1 if w.size else None


def fmt(n):
    return f"{n:>5d}" if n is not None else f"{'-':>5}"


def set_dose(ctx, clean, dose, rng):
    scale = dose / (float(clean.sum()) / clean.size)
    noisy = xp.asarray(rng.poisson(clean * scale).astype(np.float32) / scale)
    ctx["data"] = noisy
    fn = Precondition_calc(noisy, bw=ctx["Gramiam"]["bw"])
    ctx["frames_norm"] = xp.where(xp.abs(fn) < 1e-6, xp.asarray(1e-6, fn.dtype), fn)


if __name__ == "__main__":
    T.STEPD = int(os.environ.get("STEPD", 4))
    T.NUMITER = NUMITER
    ctx = T.build(K)
    clean = np.asarray(ctx["data"]) + 0.0
    steps = [parse_step(s) for s in DATASTEPS]
    ov = 100 * (1 - ctx["step"] / nx)
    print(f"NX={nx} K={K} ({ctx['nframes']} frames), img {ctx['Nx']}x{ctx['Ny']}, overlap {ov:.0f}%, "
          f"MAXIT={MAXIT} SE={SE} NUMITER={NUMITER} REPS={REPS}")

    m = mem_model(ctx)
    print(f"\n== (1) MEMORY MODEL (complex64) ==  nf={m['nf']}  nnz(overlap pairs)={m['nnz']}")
    print(f"  frames stack (batchable) {mb(m['frames_b'])} MB | loop working ~{mb(m['frames_working'])} MB "
          f"| object {mb(m['obj_b'])} MB | Gramian plan ~{mb(m['plan_b'])} MB")
    print(f"  frames-stack fraction of (frames+obj+plan) resident = "
          f"{100*m['frames_b']/max(1,(m['frames_b']+m['obj_b']+max(0,m['plan_b']))):.0f}%  "
          f"-> frame-batching / streaming is the memory lever (gpu-memory-scaling, od2p)")

    print(f"\n== (2) MATRIX: speed + dose-aware quality @ {REPDOSE} ph/px (REPS={REPS} avg) ==")
    print(f"{'combo':>14} | {'ms/it':>6} {'lo->thr':>7} | {'low':>7} {'high':>7}   (low-thr={LOWTHR})")
    set_dose(ctx, clean, REPDOSE, np.random.default_rng(0))          # warm caches/JIT (1st-call
    for _y in ("power", "eigsh", "invit"):                          # setup skews ms/it otherwise)
        solve(ctx, ("hard", 0.0), _y, SE, 3)
    rng = np.random.default_rng(7)
    for step in steps:
        for sync in SYNCS:
            los, his, its, tms = [], [], [], []
            for r in range(REPS):
                set_dose(ctx, clean, REPDOSE, rng)
                if T.GPU:
                    import cupy; cupy.cuda.Stream.null.synchronize()
                t0 = time.perf_counter()
                curve = solve(ctx, step, sync, SE, MAXIT)
                tms.append((time.perf_counter() - t0) / MAXIT * 1e3)
                los.append(curve[-1, 0]); his.append(curve[-1, 1]); its.append(iters_to(curve[:, 0], LOWTHR))
            it_med = sorted([x for x in its if x is not None])
            it_rep = it_med[len(it_med) // 2] if it_med else None
            print(f"{label(step, sync):>14} | {np.mean(tms):>6.1f} {fmt(it_rep):>7} | "
                  f"{np.mean(los):>7.3f} {np.mean(his):>7.3f}")

    print(f"\n== (3) INTERACTION: does proxD move the SYNC dose threshold? (final low-band vs dose) ==")
    idata = [parse_step(s) for s in os.environ.get("IDATASTEPS", "hard pipm:2").split()]
    isync = os.environ.get("ISYNCS", "none eigsh").split()
    cols = [f"{label(s, y)}" for s in idata for y in isync]
    print(f"{'ph/px':>6} | " + " ".join(f"{c:>13}" for c in cols))
    rng = np.random.default_rng(11)
    for dose in DOSELIST:
        vals = []
        for s in idata:
            for y in isync:
                acc = []
                for r in range(REPS):
                    set_dose(ctx, clean, dose, rng)
                    acc.append(solve(ctx, s, y, SE, MAXIT)[-1, 0])
                vals.append(np.mean(acc))
        print(f"{dose:>6g} | " + " ".join(f"{v:>13.3f}" for v in vals))
    print("\nEXPECT: (2) eigsh/invit reach the low-band in fewest iters (invit if the Fiedler gap "
          "is tight) but cost more ms/it; hard vs proxD ~equal on quality at moderate dose. "
          "(3) if the proxD column tracks the hard column across dose (same collapse point), the "
          "data-step and consensus-step axes are INDEPENDENT -> factorize the solver (pick data "
          "step for resolution, sync for long-range, separately). If proxD lowers the collapse "
          "dose, they interact -> a joint win worth a combined GPU kernel.")
