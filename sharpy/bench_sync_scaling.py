"""Framewise strong-scaling + wall-time benchmark of the sync eigensolver map.

Sweeps frame count K x solver arm x cadence on one GPU, measuring iterations to a
fixed low-band error AND wall-time, to produce the definitive scaling numbers for
the paper's "Computational cost and single-GPU scaling" section:

  arms: AP-only | power-sync | eigsh (host ARPACK) | invit-cg | invit-si
  metric: iters-to-low<LOWTHR, final low-band, wall ms/it, sync ms/call

Writes each row incrementally (flush) to BENCHOUT so partial results survive a kill,
plus a final .npz. eigsh is capped to K<=EIGSHKMAX (it degrades past there; that is
part of the story, but we don't let it eat the walltime).

GPU:  python -u bench_sync_scaling.py
env: NX(16) KLIST("20 40 60 80 100 128") MAXIT(200) SYNCEVERY_LIST("1 5")
     LOWTHR(0.1) NUMITER(5) EIGSHKMAX(100) BENCHOUT($SCRATCH/bench_sync_scaling.out)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NX", "16")

import numpy as np
import config
import Operators
import sync_bandpass_test as T

MAXIT = int(os.environ.get("MAXIT", 200))
KLIST = [int(k) for k in os.environ.get("KLIST", "20 40 60 80 100 128").split()]
SE_LIST = [int(s) for s in os.environ.get("SYNCEVERY_LIST", "1 5").split()]
LOWTHR = float(os.environ.get("LOWTHR", 0.1))
T.NUMITER = int(os.environ.get("NUMITER", 5))
EIGSHKMAX = int(os.environ.get("EIGSHKMAX", 100))
BENCHOUT = os.environ.get("BENCHOUT", "bench_sync_scaling.out")
Operators.SYNC_EPS = float(os.environ.get("SHARPY_SYNC_EPS", "1e-3"))
Operators.SYNC_TOL = float(os.environ.get("SHARPY_SYNC_TOL", "1e-4"))
Operators.SYNC_STEPS = int(os.environ.get("SHARPY_SYNC_STEPS", "2"))


def _sync():
    if config.GPU:
        import cupy
        cupy.cuda.Stream.null.synchronize()


def timed(ctx, se, method, mode, solver):
    """Run one arm; return (iters_to_thr, final_low, final_high, ms_per_it)."""
    Operators.SYNC_METHOD = method
    if mode:
        Operators.SYNC_MODE = mode
    _sync(); t0 = time.perf_counter()
    c = T.run(ctx, se, MAXIT, solver=solver)
    _sync(); dt = time.perf_counter() - t0
    it = T.iters_to(c[:, 0], LOWTHR)
    return it, float(c[-1, 0]), float(c[-1, 1]), 1e3 * dt / MAXIT


ARMS = [  # (label, sync_every_uses_SE, method, mode, solver_factory)
    ("AP-only",  False, "power", None, lambda: None),
    ("power",    True,  "power", None, lambda: None),
    ("eigsh",    True,  "power", None, lambda: T.eigsh_sync),
    ("invit-cg", True,  "invit", "cg", lambda: None),
    ("invit-si", True,  "invit", "si", lambda: None),
]

rows = []
hdr = (f"{'K':>4} {'frames':>6} {'cad':>3} {'arm':>9} | {'iters':>5} "
       f"{'final_lo':>8} {'final_hi':>8} {'ms/it':>8}")
with open(BENCHOUT, "w") as f:
    f.write(f"# bench_sync_scaling nx={T.nx} MAXIT={MAXIT} LOWTHR={LOWTHR} "
            f"eps={Operators.SYNC_EPS} tol={Operators.SYNC_TOL}\n{hdr}\n")
print(hdr, flush=True)

for K in KLIST:
    ctx = T.build(K)
    nf = ctx["nframes"]
    # AP-only once per K (cadence-independent); reuse its time as the baseline.
    for se in SE_LIST:
        for label, uses_se, method, mode, sfac in ARMS:
            if label == "AP-only" and se != SE_LIST[0]:
                continue                       # AP has no cadence; run once
            if label == "eigsh" and K > EIGSHKMAX:
                row = (K, nf, se, label, None, np.nan, np.nan, np.nan)
                rows.append(row)
                line = (f"{K:>4} {nf:>6} {se:>3} {label:>9} | {'skip':>5} "
                        f"{'-':>8} {'-':>8} {'-':>8}")
                print(line, flush=True)
                with open(BENCHOUT, "a") as f:
                    f.write(line + "\n")
                continue
            eff_se = se if uses_se else 0
            try:
                it, lo, hi, mspit = timed(ctx, eff_se, method, mode, sfac())
            except Exception as e:               # never let one arm kill the sweep
                it, lo, hi, mspit = None, float("nan"), float("nan"), float("nan")
                print(f"  !! {label} K={K} se={se} FAILED: {type(e).__name__}: {e}",
                      flush=True)
            rows.append((K, nf, se, label, it, lo, hi, mspit))
            its = f"{it:>5d}" if it is not None else f"{'--':>5}"
            line = (f"{K:>4} {nf:>6} {se:>3} {label:>9} | {its} "
                    f"{lo:>8.4f} {hi:>8.4f} {mspit:>8.2f}")
            print(line, flush=True)
            with open(BENCHOUT, "a") as f:
                f.write(line + "\n")

# structured save for plotting
import numpy as _np
K_a  = _np.array([r[0] for r in rows]); nf_a = _np.array([r[1] for r in rows])
se_a = _np.array([r[2] for r in rows]); arm_a = _np.array([r[3] for r in rows])
it_a = _np.array([(-1 if r[4] is None else r[4]) for r in rows])
lo_a = _np.array([r[5] for r in rows]); hi_a = _np.array([r[6] for r in rows])
ms_a = _np.array([r[7] for r in rows])
_np.savez(os.path.splitext(BENCHOUT)[0] + ".npz", K=K_a, frames=nf_a, cadence=se_a,
          arm=arm_a, iters=it_a, final_lo=lo_a, final_hi=hi_a, ms_per_it=ms_a)
print(f"\nWROTE {BENCHOUT} + .npz  ({len(rows)} rows)", flush=True)
