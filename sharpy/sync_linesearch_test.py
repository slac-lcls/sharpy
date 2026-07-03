"""MG/OPT-style LINE-SEARCH SAFEGUARD on the Gramian sync (Fung & Di 2018 lesson).

The sync correction omega is a per-frame-constant gauge; when the frame is a large
fraction of the image (model mismatch: within-frame phase ramps), applying it
outright every iteration can HURT the endgame (measured: nx=64 K=10, eigsh-sync
final low-band 0.044 vs AP-only 0.018). MG/OPT's fix: prolonged coarse corrections
are search DIRECTIONS, accepted via a line search on the fine objective. Here:
candidates omega^alpha = exp(i*alpha*arg omega), alpha in {0, 1/2, 1}, scored by
the DATA misfit ||  |F frames| - sqrt(d) || after consensus; keep the best.

Expect: alpha=1 chosen early (keeps the O(1) low-band collapse), alpha -> 0 in the
mismatched endgame (recovers AP quality); at scale (frame << image) alpha=1 always
-> no regression. Cost: extra P_a evaluations per sync (amortized by cadence).

CPU:  /opt/anaconda3/bin/python3 sync_linesearch_test.py
env: NX K STEPD MAXIT SYNCEVERY SOLVER(eigsh|invit) + sync_bandpass_test's envs
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import sync_bandpass_test as T
import Operators
from Operators import xp

ALPHAS = (0.0, 0.5, 1.0)


def data_misfit(ctx, frames):
    """Relative amplitude misfit ||  |F z| - sqrt(d) || / ||sqrt(d)|| (the fine objective)."""
    a = xp.abs(xp.fft.fft2(frames))
    sq = xp.sqrt(ctx["data"])
    return float(xp.linalg.norm(a - sq) / xp.linalg.norm(sq))


def run_ls(ctx, sync_every, maxit, solver=None, linesearch=True):
    """AP + periodic sync with the omega^alpha line-search safeguard."""
    Operators.Eigensolver = solver if solver is not None else T._POWER
    T.eig_reset(); T._eigsh_v0["v"] = None
    probe, cprobe, mapid = ctx["probe"], ctx["cprobe"], ctx["mapid"]
    img = xp.ones((ctx["Ny"], ctx["Nx"]), dtype=xp.complex64)
    frames = T.Illuminate_frames(T.Splitc(img, mapid), probe)
    curve, alog = [], []
    for it in range(maxit):
        frames, _ = T.Project_data(frames, ctx["data"])
        if sync_every and it % sync_every == 0:
            om = T.synchronize_frames_c(frames, probe, ctx["frames_norm"], ctx["inorm_split"],
                                        ctx["Gramiam"], ctx["Gramiam"]["bw"], T.NUMITER)
            if linesearch:
                th = xp.angle(om)
                best, bscore = None, np.inf
                for a in ALPHAS:
                    fc = frames * xp.exp(1j * a * th).astype(xp.complex64) if a else frames
                    ic = T.Overlapc(T.Illuminate_frames(fc, cprobe), ctx["Nx"], ctx["Ny"],
                                    mapid) / ctx["normalization"]
                    fc2 = T.Illuminate_frames(T.Splitc(ic, mapid), probe)
                    sc = data_misfit(ctx, fc2)
                    if sc < bscore:
                        best, bscore, bimg, bfr = a, sc, ic, fc2
                alog.append(best)
                img, frames = bimg, bfr
            else:
                frames = frames * om
                img = T.Overlapc(T.Illuminate_frames(frames, cprobe), ctx["Nx"], ctx["Ny"],
                                 mapid) / ctx["normalization"]
                frames = T.Illuminate_frames(T.Splitc(img, mapid), probe)
        else:
            img = T.Overlapc(T.Illuminate_frames(frames, cprobe), ctx["Nx"], ctx["Ny"],
                             mapid) / ctx["normalization"]
            frames = T.Illuminate_frames(T.Splitc(img, mapid), probe)
        curve.append(T.band_err(ctx, img))
    Operators.Eigensolver = T._POWER
    return np.array(curve), alog


if __name__ == "__main__":
    MAXIT = int(os.environ.get("MAXIT", 300))
    SE = int(os.environ.get("SYNCEVERY", 1))
    T.NUMITER = int(os.environ.get("NUMITER", 5))
    solver_name = os.environ.get("SOLVER", "eigsh")
    if solver_name == "invit":
        Operators.SYNC_METHOD = "invit"
        solver = None
    else:
        Operators.SYNC_METHOD = "power"   # dispatch through Eigensolver -> patched eigsh
        solver = T.eigsh_sync
    K = int(os.environ.get("K", 10))
    ctx = T.build(K)
    print(f"K={K} nx={T.nx}: {ctx['nframes']} frames, img {ctx['Nx']}^2, solver={solver_name}, "
          f"sync every {SE}, MAXIT={MAXIT}")
    ap = T.run(ctx, 0, MAXIT)
    sy, _ = run_ls(ctx, SE, MAXIT, solver=solver, linesearch=False)
    ls, alog = run_ls(ctx, SE, MAXIT, solver=solver, linesearch=True)
    for lbl, c in (("AP-only", ap), ("sync (raw)", sy), ("sync+linesearch", ls)):
        it = T.iters_to(c[:, 0], 0.1)
        print(f"  {lbl:>16}: it1 {c[0,0]:.4f} it20 {c[min(19,MAXIT-1),0]:.4f} "
              f"final low {c[-1,0]:.4f} high {c[-1,1]:.4f} | to<0.1 {it}")
    a = np.array(alog)
    n10 = max(1, len(a) // 10)
    print(f"  alpha chosen: first-10% mean {a[:n10].mean():.2f}, last-10% mean {a[-n10:].mean():.2f}, "
          f"overall {a.mean():.2f} (fraction alpha=0: {np.mean(a==0):.2f})")
