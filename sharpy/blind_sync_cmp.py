"""Blind-mode + sync: the regime where the poster catastrophe lives.

Same geometry as ppm_sync_cmp.py (gold_balls, step=5, non-astig probe, 4096 frames)
but with refine_illumination=True so the probe is co-estimated from a WRONG start
(the _c blind init: tilted round probe fx=+10,fy=-10). The probe-refinement feedback
is the hypothesised amplifier: an eigsh phase-flip in the sync vector corrupts the
overlap, which corrupts the refined probe, which feeds back -> divergence; while the
warm-start power iteration stays on the smooth consensus and the loop converges.

Compares the in-loop object-NMSE trajectory (residuals[:,0]) for:
  - no sync
  - power + warm/ones (committed Eigensolver)
  - PPM (projected power)
  - eigsh (cupy, k=1, ncv=3, under-converged) = the suspect

  srun -A lcls_g -C gpu -q interactive -N1 -n1 --gpus 1 -t 00:30:00 python blind_sync_cmp.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
assert config.GPU
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
from PIL import Image
from cupyx.scipy.sparse.linalg import eigsh
import Operators
from Operators import (make_probe, make_translations, cropmat, Gramiam_plan, eig_reset, xp)
from wrap_ops import overlap_cuda, split_cuda
import Solvers

nx = ny = 16; nnx = 64
Dx = int(os.environ.get("SHARPY_STEP", 5))                 # scan step -> controls overlap/Fiedler gap
maxiter = int(os.environ.get("SHARPY_MAXITER", 300))
CONTRAST = float(os.environ.get("CONTRAST", 0.69))         # object absorption; >0.69 -> non-uniform frame energy -> eigsh flip
print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
print("CONTRAST=%.2f  STEP=%d" % (CONTRAST, Dx))
img_full = np.exp(CONTRAST * (-1 + 0.5j) * (np.array(Image.open("../data/gold_balls.png"), np.float32) / 63.0))
probe = make_probe(nx, ny, r1=0.025 * 3, r2=0.085 * 3, fx=0.0, fy=0.0)        # non-astig TRUTH probe
probe = xp.asarray(probe[0] if isinstance(probe, tuple) else probe).astype(xp.complex64)
tx, ty = make_translations(Dx, Dx, nnx, nnx, nnx * Dx, nnx * Dx)
tx = xp.asarray(tx).astype(xp.int64); ty = xp.asarray(ty).astype(xp.int64)
Nx = int(xp.ceil(xp.max(tx) - xp.min(tx))); nf = tx.size
tr = (tx + 1j * ty).astype(xp.complex64)
truth = cp.ascontiguousarray(cropmat(xp.asarray(img_full).astype(xp.complex64), [Nx, Nx]))
frd = xp.zeros((nf, nx, ny), xp.complex64); split_cuda(truth, frd, tr, probe)
data = (xp.abs(xp.fft.fft2(frd)) ** 2).astype(xp.float32)
G = Gramiam_plan(tx, ty, nf, nx, ny, Nx, Nx, bw=0)
overlap_lin = max(0.0, (nx - Dx) / nx)
print("frames %d, image %d^2, step %d (~%.0f%% linear overlap), maxiter %d (BLIND)"
      % (nf, Nx, Dx, 100 * overlap_lin, maxiter))

# report the Gramian conditioning (top eigenvalue cluster / Fiedler gap) on truth frames
from Operators import Gramiam_calc_cuda, Precondition_calc
_norm = xp.zeros((Nx, Nx), xp.complex64); overlap_cuda(_norm, 0, tr, probe)
_inorm = xp.zeros(frd.shape, xp.complex64); split_cuda(1.0 / (_norm + 1e-8), _inorm, tr, 0)
_H = Gramiam_calc_cuda(frd, G, probe, _inorm, Precondition_calc(frd, bw=0))
_Hd = _H.toarray() if hasattr(_H, "toarray") else _H
_lam = cp.linalg.eigvalsh(_Hd.astype(cp.complex128))[::-1].real
_lam = _lam / _lam[0]
print("Gramian top-5 eig/eig1 =", np.round(cp.asnumpy(_lam[:5]), 4),
      " gap(1-lam2) = %.4f" % float(1 - _lam[1]))
del _H, _Hd, _norm, _inorm
_power = Operators.Eigensolver


def gauge(w, n):
    w = w.ravel() / (cp.abs(w.ravel()) + 1e-30)
    s = cp.sum(w); w = w * cp.conj(s) / (cp.abs(s) + 1e-30)
    return cp.reshape(w, (n, 1, 1))


def ppm(H, num_iter, v0=None, tol=1e-6):
    n = H.shape[0]; v = cp.ones((n, 1), cp.complex64)
    for _ in range(num_iter):
        v = H @ v; v = v / (cp.abs(v) + 1e-30)            # per-entry unit-modulus projection
    return gauge(v[:, 0], n)


def eigsh_cupy(H, num_iter, v0=None, tol=1e-6):
    n = H.shape[0]
    lam, V = eigsh(H, k=1, ncv=3, maxiter=20, which="LM")
    return gauge(V[:, int(cp.argmax(lam))], n)


def run(label, sync, solver):
    Operators.Eigensolver = solver if solver else _power
    eig_reset()
    img, frames, illum, res = Solvers.Alternating_projections_c(
        sync, xp.ones((Nx, Nx), xp.complex64), G, probe + 0, tx, ty, overlap_cuda,
        split_cuda, data, True, maxiter, None, truth, 1, sync_interval=1, num_iter=5)
    nmse = cp.asnumpy(res[:, 0])
    s = xp.vdot(img, truth) / (xp.vdot(img, img) + 1e-30)
    fin = float(xp.linalg.norm(s * img - truth) / xp.linalg.norm(truth))
    print("  %-30s final-NMSE = %.3e  (in-loop last %.3e, min %.3e)"
          % (label, fin, nmse[-1], np.nanmin(nmse)))
    return nmse


curves = {}
curves["no sync"] = run("no sync", False, _power)
curves["power (committed)"] = run("power + warm/ones (committed)", True, _power)
curves["PPM"] = run("PPM (projected power)", True, ppm)
curves["eigsh (k=1,ncv=3)"] = run("eigsh cupy (k=1,ncv=3)", True, eigsh_cupy)

plt.figure(figsize=(7, 5))
for k, v in curves.items():
    plt.semilogy(v, label=k)
plt.xlabel("AP iteration"); plt.ylabel("object NMSE"); plt.legend()
plt.title("Blind mode, step=%d (~%.0f%% overlap, gap %.3f): sync eigensolver cmp"
          % (Dx, 100 * overlap_lin, float(1 - _lam[1])))
plt.grid(True, which="both", alpha=0.3)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blind_sync_step%d.png" % Dx)
plt.tight_layout(); plt.savefig(out, dpi=120)
print("saved", out)
print("DONE")
