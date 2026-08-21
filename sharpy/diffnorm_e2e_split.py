"""End-to-end attribution: how much of the SHARPY_FUSED_PROXD win is _diffnorm?

proxd_e2e_test.py toggles ONE flag that turns on THREE things at once -- the
fused ProxD+residual kernel, the frames_old reference trick (copy dropped), and
the fused eps_S residual _diffnorm -- so its end-to-end speedup cannot tell you
what the eps_S residual contributed. This script separates them with a third
arm, monkeypatching Operators._diffnorm back to the historical
xp.linalg.norm(a - b) while leaving the ProxD kernel and the ref trick on:

  A  plain        _FUSED_PROXD=0 : two-pass ProxD, frames_old copy, linalg.norm
  B  proxd-only   _FUSED_PROXD=1 : fused ProxD + ref trick, linalg.norm eps_S
  C  fused        _FUSED_PROXD=1 : all three (production fused path)

  B/A = the ProxD kernel + ref trick;  C/B = _diffnorm alone;  C/A = both.

Arms ALTERNATE within each rep (proxd_e2e_test.py style) so GPU clock-ramp and
memory-pool state bias all three equally; the median over reps is reported.

  sbatch -p <gpu-partition> -A <account> -N1 -n1 --gpus 1 -t 20 \
    --wrap 'source <venv>/bin/activate; cd <sharpy>/sharpy; \
            python -u diffnorm_e2e_split.py'
  env: NX (frame size, 128), KG (scan grid KxK, 24), MAXITER (100), REPS (4)
"""
import os
import sys
import time
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import numpy as np
from Operators import Split_Overlap_plan, xp
import Operators
import Solvers

GPU = config.GPU
if GPU:
    import cupy as cp
    print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())

_REAL_DIFFNORM = Operators._diffnorm


def _plain_diffnorm(a, b):
    """The historical eps_S path, forced regardless of _FUSED_PROXD."""
    return xp.linalg.norm(a - b)


def phantom(Nx, Ny, contrast=1.5, seed=0):
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.standard_normal((Nx, Ny)))
    k = np.fft.fftfreq(Nx)
    KX, KY = np.meshgrid(k, k)
    sm = np.real(np.fft.ifft2(F * np.exp(-(KX ** 2 + KY ** 2) / (2 * 0.03 ** 2))))
    sm = (sm - sm.min()) / (sm.max() - sm.min())
    return ((0.5 + 0.5 * sm) * np.exp(1j * contrast * (sm - 0.5))).astype(np.complex64)


nx = ny = int(os.environ.get("NX", 128))
K = int(os.environ.get("KG", 24))
step = nx // 4                              # 75% overlap
Nx = Ny = (K - 1) * step + nx
g = xp.arange(K) * step
tx, ty = xp.meshgrid(g, g, indexing="ij")
tx = tx.ravel().astype(float); ty = ty.ravel().astype(float)
nframes = tx.size

truth = xp.asarray(phantom(Nx, Ny, 1.5)).astype(xp.complex64)
c = nx // 2
gx = xp.arange(nx) - c
X, Y = xp.meshgrid(gx, gx)
probe = xp.exp(-(X ** 2 + Y ** 2) / (2.0 * (0.18 * nx) ** 2)).astype(xp.complex64)
probe = (probe / xp.abs(probe).max()).astype(xp.complex64)

Split, Overlap = Split_Overlap_plan(tx, ty, nx, ny, Nx, Ny)
data = (xp.abs(xp.fft.fft2(Split(truth) * probe[None])) ** 2).astype(xp.float32)
print(f"img {Nx}x{Ny}, frames {nframes} x {nx}, "
      f"eps_S reduction size = {nframes*nx*ny/1e6:.1f}M complex")


def sync():
    if GPU:
        cp.cuda.Stream.null.synchronize()


ARMS = ("plain", "proxd-only", "fused")


def run(arm, maxiter):
    Operators._FUSED_PROXD = (arm != "plain")
    # arm B keeps the fused ProxD kernel + ref trick but the OLD eps_S residual
    Operators._diffnorm = _plain_diffnorm if arm == "proxd-only" else _REAL_DIFFNORM
    for k in list(Solvers.timers):
        Solvers.timers[k] = 0
    img0 = xp.ones((Nx, Ny), dtype=xp.complex64)
    sync(); t0 = time.time()
    img, frames, illum, res = Solvers.Alternating_projections(
        False, img0, None, probe + 0, Overlap, Split, data,
        False, maxiter, None, truth, 1)
    sync(); dt = time.time() - t0
    Operators._diffnorm = _REAL_DIFFNORM
    r = res.get() if GPU else res
    return r[-1, 0], r[-1, 2], dt


MAXITER = int(os.environ.get("MAXITER", 100))
REPS = int(os.environ.get("REPS", 4))
for a in ARMS:                                   # warmup: cuFFT plans + JIT
    run(a, 5)

dts = {a: [] for a in ARMS}
nmses, epss = {}, {}
for rep in range(REPS):
    for arm in ARMS:                             # ALTERNATE arms within a rep
        nmse, eps_s, dt = run(arm, MAXITER)
        dts[arm].append(dt); nmses[arm] = nmse; epss[arm] = eps_s

print(f"\nmedian over {REPS} alternating reps, {MAXITER} iters each")
print(f"{'arm':12} {'NMSE':>11} {'eps_S':>11} {'ms/iter':>9}")
for arm in ARMS:
    print(f"{arm:12} {nmses[arm]:>11.4e} {epss[arm]:>11.4e} "
          f"{1e3*median(dts[arm])/MAXITER:>9.3f}")

A, B, C = (median(dts[a]) for a in ARMS)
print(f"\nProxD kernel + ref trick (B/A): {A/B:.3f}x")
print(f"_diffnorm alone          (C/B): {B/C:.3f}x   <-- the eps_S residual's own share")
print(f"both                     (C/A): {A/C:.3f}x")
ok = all(abs(epss[a] - epss["plain"]) <= 1e-3 * abs(epss["plain"]) for a in ARMS)
print(f"eps_S agrees across arms: {ok}  " + " / ".join(f"{a}={epss[a]:.6e}" for a in ARMS))
sys.exit(0 if ok else 1)
