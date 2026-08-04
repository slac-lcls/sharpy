#!/usr/bin/env python3
"""How does SYNC EIGENSOLVER ACCURACY affect the AP reconstruction, and what is each
solver's MEMORY footprint? Varying-low-freq-phase regime (where sync matters).

Sweeps power-sync num_iter (= sync accuracy knob) and compares to eigsh / invit / no-sync,
recording AP final object-NMSE, cupy memory-pool high-water (MB), and wall-time. Answers:
(a) is there an accuracy THRESHOLD the AP loop needs (does power@large converge, or only
accurate eigsh/invit?), (b) do invit/eigsh cost materially more memory than power?

  CYCLES="1 2" R2=0.40 SHARPY_MAXITER=300 srun ... python sync_accuracy_mem.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config; assert config.GPU
import cupy as cp, numpy as np
from cupyx.scipy.sparse.linalg import eigsh
import Operators
from Operators import make_probe, make_translations, Gramiam_plan, eig_reset, xp
from wrap_ops import overlap_cuda, split_cuda
import Solvers

nx = ny = 16; Dx = 5; nnx = int(os.environ.get("NNX", 64))
r2 = float(os.environ.get("R2", 0.40)); maxiter = int(os.environ.get("SHARPY_MAXITER", 300))
cycles_list = [float(c) for c in os.environ.get("CYCLES", "1 2").split()]
probe = make_probe(nx, ny, r1=0.025 * 3, r2=r2, fx=0.0, fy=0.0)
probe = xp.asarray(probe[0] if isinstance(probe, tuple) else probe).astype(xp.complex64)
tx, ty = make_translations(Dx, Dx, nnx, nnx, nnx * Dx, nnx * Dx)
tx = xp.asarray(tx).astype(xp.int64); ty = xp.asarray(ty).astype(xp.int64)
Nx = int(xp.ceil(xp.max(tx) - xp.min(tx))); nf = tx.size
tr = (tx + 1j * ty).astype(xp.complex64)
G = Gramiam_plan(tx, ty, nf, nx, ny, Nx, Nx, bw=0)
xx = (xp.arange(Nx) - Nx / 2).astype(xp.float32); R2 = xx[:, None] ** 2 + xx[None, :] ** 2
print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
print("frames %d  image %d^2  r2=%.3f  maxiter %d" % (nf, Nx, r2, maxiter))
_power = Operators.Eigensolver
mp = cp.get_default_memory_pool()


def gauge(w, n):
    w = w.ravel() / (cp.abs(w.ravel()) + 1e-30); s = cp.sum(w)
    return cp.reshape(w * cp.conj(s) / (cp.abs(s) + 1e-30), (n, 1, 1))

def eigsh_cupy(H, num_iter, v0=None, tol=1e-6):
    lam, V = eigsh(H, k=1, ncv=3, maxiter=20, which="LM"); return gauge(V[:, int(cp.argmax(lam))], H.shape[0])

def invit_cupy(H, num_iter, v0=None, tol=1e-6):
    return Operators.Eigensolver_invit(H, eps=1e-4, steps=2)


def invit_direct_cupy(H, num_iter, v0=None, tol=1e-6):
    # inverse-power iteration via cupyx splu factorization = the shift-invert route
    # (smallest eigenvalue of L = I - Hn / Gramian eigenvalue closest to 1).
    return Operators.Eigensolver_invit(H, eps=1e-4, steps=3, mode="direct")


def run(truth, sync, solver, num_iter):
    Operators.Eigensolver = solver if solver else _power
    eig_reset(); mp.free_all_blocks(); cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
    img, frames, illum, res = Solvers.Alternating_projections_c(
        sync, xp.ones((Nx, Nx), xp.complex64), G, probe + 0, tx, ty, overlap_cuda,
        split_cuda, data, False, maxiter, None, truth, 1, sync_interval=1, num_iter=num_iter)
    cp.cuda.Stream.null.synchronize(); dt = time.perf_counter() - t0
    s = xp.vdot(img, truth) / (xp.vdot(img, img) + 1e-30)
    fin = float(xp.linalg.norm(s * img - truth) / xp.linalg.norm(truth))
    return fin, dt, mp.total_bytes() / 1e6


for cyc in cycles_list:
    alpha = cyc * 2 * np.pi / float(R2.max())
    truth = xp.exp(1j * alpha * R2).astype(xp.complex64)
    frd = xp.zeros((nf, nx, ny), xp.complex64); split_cuda(truth, frd, tr, probe)
    global data; data = (xp.abs(xp.fft.fft2(frd)) ** 2).astype(xp.float32)
    print("\n=== CYCLES=%.1f ===  %-22s %12s %8s %8s" % (cyc, "method", "final-NMSE", "mem-MB", "s"))
    configs = [("no-sync", False, _power, 5)]
    configs += [("power num_iter=%d" % k, True, _power, k) for k in (5, 50)]
    configs += [("eigsh (Lanczos LM)", True, eigsh_cupy, 5),
                ("invit (CG)", True, invit_cupy, 5),
                ("invit-direct (inv-power/splu)", True, invit_direct_cupy, 5)]
    for label, sync, solver, k in configs:
        fin, dt, mem = run(truth, sync, solver, k)
        print("  %-22s %12.3e %8.0f %8.1f" % (label, fin, mem, dt))

print("\nKey: does AP final-NMSE keep IMPROVING with power num_iter (threshold?), or PLATEAU")
print("above eigsh/invit (=accuracy-limited -> accurate solver needed)? And is invit/eigsh mem >> power?")
