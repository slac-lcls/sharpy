"""Where does sync HELP? Hypothesis: low-frequency phase recovery.

Non-blind (known probe), step=5, SMALLER probe (larger r2 -> less overlap -> small
Fiedler gap, where sync should matter). Object = smooth long-range quadratic phase
exp(i*alpha*(x^2+y^2)), centered, alpha set so the corner-to-center swing is
~CYCLES*2pi across the image (periodic BC noted). Unit amplitude = isolate the phase.

Compares object-NMSE convergence (residuals[:,0]) for no-sync vs power-sync vs
eigsh-sync, over a few phase levels. If sync helps low-freq phase, no-sync lags.

  CYCLES "0 1 2"  R2 0.40
  srun -A lcls_g -C gpu -q interactive -N1 -n1 --gpus 1 -t 00:25:00 python phase_sync_test.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
assert config.GPU
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
from cupyx.scipy.sparse.linalg import eigsh
import Operators
from Operators import (make_probe, make_translations, Gramiam_plan, eig_reset, xp, cropmat)
from wrap_ops import overlap_cuda, split_cuda
import Solvers

nx = ny = 16; Dx = 5; nnx = 64
r2 = float(os.environ.get("R2", 0.40))                      # smaller probe = larger r2
maxiter = int(os.environ.get("SHARPY_MAXITER", 300))
cycles_list = [float(c) for c in os.environ.get("CYCLES", "0 1 2").split()]
OBJECT = os.environ.get("OBJECT", "quad")   # "quad": pure quadratic phase; "gold": gold-balls transmission x quad phase
print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())

probe = make_probe(nx, ny, r1=0.025 * 3, r2=r2, fx=0.0, fy=0.0)
probe = xp.asarray(probe[0] if isinstance(probe, tuple) else probe).astype(xp.complex64)
tx, ty = make_translations(Dx, Dx, nnx, nnx, nnx * Dx, nnx * Dx)
tx = xp.asarray(tx).astype(xp.int64); ty = xp.asarray(ty).astype(xp.int64)
Nx = int(xp.ceil(xp.max(tx) - xp.min(tx))); nf = tx.size
tr = (tx + 1j * ty).astype(xp.complex64)
G = Gramiam_plan(tx, ty, nf, nx, ny, Nx, Nx, bw=0)
xx = (xp.arange(Nx) - Nx / 2).astype(xp.float32)
R2 = (xx[:, None] ** 2 + xx[None, :] ** 2)                 # centered r^2
print("frames %d, image %d^2, probe r2=%.3f, maxiter %d  object=%s" % (nf, Nx, r2, maxiter, OBJECT))
_power = Operators.Eigensolver

gold = None
if OBJECT == "gold":
    from PIL import Image
    _here = os.path.dirname(os.path.abspath(__file__))
    _img = np.array(Image.open(os.path.join(_here, "..", "data", "gold_balls.png")), np.float32) / 63.0
    _t = np.exp(0.69 * (-1.0 + 0.5j) * _img).astype(np.complex64)   # gold-balls transmission (amp+phase)
    gold = cropmat(xp.asarray(_t), [Nx, Nx]).astype(xp.complex64)   # crop/pad to the image grid


def gauge(w, n):
    w = w.ravel() / (cp.abs(w.ravel()) + 1e-30)
    s = cp.sum(w); w = w * cp.conj(s) / (cp.abs(s) + 1e-30)
    return cp.reshape(w, (n, 1, 1))


def eigsh_cupy(H, num_iter, v0=None, tol=1e-6):
    lam, V = eigsh(H, k=1, ncv=3, maxiter=20, which="LM")
    return gauge(V[:, int(cp.argmax(lam))], H.shape[0])


def invit_cupy(H, num_iter, v0=None, tol=1e-6):
    # A4 inverse iteration on the connection Laplacian (arXiv:1209.4924 App. A):
    # solve the BOTTOM of L=D-H instead of power-iterating the near-degenerate TOP.
    return Operators.Eigensolver_invit(H, eps=1e-4, steps=2)


_seed_state = {"n": 0}
def invit_seed_cupy(H, num_iter, v0=None, tol=1e-6):
    # invit ONCE for the first (cold) sync -> seeds Operators._eig_v0; warm power thereafter.
    if _seed_state["n"] == 0:
        v = Operators.Eigensolver_invit(H, eps=1e-4, steps=2)   # seeds _eig_v0 internally
    else:
        v = _power(H, num_iter)                                  # warm power from _eig_v0 (drift tracking)
    _seed_state["n"] += 1
    return v


def run(truth, sync, solver):
    Operators.Eigensolver = solver if solver else _power
    eig_reset(); _seed_state["n"] = 0
    cp.cuda.Stream.null.synchronize(); _t0 = time.perf_counter()
    img, frames, illum, res = Solvers.Alternating_projections_c(
        sync, xp.ones((Nx, Nx), xp.complex64), G, probe + 0, tx, ty, overlap_cuda,
        split_cuda, data, False, maxiter, None, truth, 1, sync_interval=1, num_iter=5)
    cp.cuda.Stream.null.synchronize(); dt = time.perf_counter() - _t0
    nmse = cp.asnumpy(res[:, 0])
    s = xp.vdot(img, truth) / (xp.vdot(img, img) + 1e-30)
    fin = float(xp.linalg.norm(s * img - truth) / xp.linalg.norm(truth))
    return nmse, fin, dt


fig, axes = plt.subplots(1, len(cycles_list), figsize=(5 * len(cycles_list), 4), squeeze=False)
all_vals = []
for ci, cyc in enumerate(cycles_list):
    alpha = cyc * 2 * np.pi / float(R2.max())               # corner swing ~ cyc*2pi
    truth = xp.exp(1j * alpha * R2).astype(xp.complex64)    # smooth low-frequency quadratic phase
    if gold is not None:
        truth = (gold * truth).astype(xp.complex64)         # gold-balls transmission x quad phase
    # data from the known probe + truth
    frd = xp.zeros((nf, nx, ny), xp.complex64); split_cuda(truth, frd, tr, probe)
    global data; data = (xp.abs(xp.fft.fft2(frd)) ** 2).astype(xp.float32)
    print("\n=== CYCLES=%.1f (corner phase %.2f rad) ===" % (cyc, float(alpha * R2.max())))
    res = {}
    for label, sync, solver in [("no sync", False, _power),
                                 ("power-sync", True, _power),
                                 ("eigsh-sync", True, eigsh_cupy),
                                 ("invit-sync", True, invit_cupy),
                                 ("invit-seed", True, invit_seed_cupy)]:
        nmse, fin, dt = run(truth, sync, solver)
        res[label] = nmse; all_vals.append(nmse)
        print("  %-12s final-NMSE %.3e  in-loop last %.3e  min %.3e  %5.1fs"
              % (label, fin, nmse[-1], np.nanmin(nmse), dt))
    ax = axes[0, ci]
    for k, v in res.items():
        ax.semilogy(v, label=k)
    ax.set_title("%s, %.0f cycles (corner %.1f rad)" % (OBJECT, cyc, float(alpha * R2.max())))
    ax.set_xlabel("AP iter"); ax.set_ylabel("object NMSE"); ax.grid(True, which="both", alpha=0.3); ax.legend()
# common vertical axis across panels, so the flat-object panel is not a magnified-noise zoom
_allc = np.concatenate([np.asarray(v).ravel() for v in all_vals])
_ylo = max(1e-8, float(np.nanmin(_allc))); _yhi = float(np.nanmax(_allc))
for _ax in axes.ravel():
    _ax.set_ylim(_ylo * 0.7, _yhi * 1.5)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase_sync.png")
fig.tight_layout(); fig.savefig(out, dpi=120)
print("\nsaved", out, "\nDONE")
