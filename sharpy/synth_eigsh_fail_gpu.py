#!/usr/bin/env python3
"""GPU twin of synth_eigsh_fail.py -- confirm the sign-flip on the REAL cupy eigsh.

Builds the same synthetic overlap-graph Gramian H = diag(s) K diag(s) (see
synth_eigsh_fail.py for the model and the two knobs: near-degenerate cluster +
non-uniform per-frame energy), then runs the PRODUCTION eigensolver
(cupyx.scipy.sparse.linalg.eigsh, the poster's tool, k=1, ncv=3) under-converged,
vs. GPU power iteration from ones. Also tests the real poster Gramian saveH.npz
if present.

Expected: with STRUCTURED weights (and on saveH.npz), under-converged cupy eigsh
returns a sign-FLIPPED top eigenvector (omega=v/|v| minority-sign ~0.1-0.5);
power iteration from ones stays clean (~0). With UNIFORM weights eigsh is clean.

Run on Perlmutter (sharpy-venv has cupy 12.3.0):
  source ~/sharpy-venv/bin/activate
  srun -A lcls_g -C gpu -q interactive -N1 -n1 --gpus 1 -t 00:10:00 \
       python -u $SCRATCH/sharpy/sharpy/synth_eigsh_fail_gpu.py
  ENV: G=64 SIGMA=1.6 R=4 CONTRAST=0.85   (same as the CPU script)
"""
import os
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.linalg import eigsh as cu_eigsh

# reuse the CPU model builders (scipy build is fine; we move H to the GPU)
from synth_eigsh_fail import build_H, frame_weights, minority_sign, N

MAXITERS = [int(x) for x in os.environ.get("MAXITERS", "5 10 20 50 100").split()]
NCV = int(os.environ.get("NCV", 3))


def to_gpu(H_csr):
    return csp.csr_matrix(
        (cp.asarray(H_csr.data), cp.asarray(H_csr.indices), cp.asarray(H_csr.indptr)),
        shape=H_csr.shape)


def gpu_power(Hg, it):
    v = cp.ones(Hg.shape[0], dtype=cp.complex128)
    v /= cp.linalg.norm(v)
    for _ in range(it):
        v = Hg @ v
        v /= cp.linalg.norm(v)
    return cp.asnumpy(v)


def eigsh_flip(Hg, maxiter):
    """Top eigenvector via the production cupy eigsh, under-converged; return flip."""
    try:
        w, V = cu_eigsh(Hg, k=1, which="LA", ncv=NCV, maxiter=maxiter, tol=0)
        return minority_sign(cp.asnumpy(V[:, 0]))
    except Exception as e:
        return f"err({type(e).__name__})"


def _fmt(f):
    return f"{f:.3f}" if isinstance(f, float) else str(f)


def report(name, Hg):
    print(f"\n[{name}]  #frames={Hg.shape[0]}  nnz/row={Hg.nnz / Hg.shape[0]:.0f}")
    print("   cupy eigsh(ncv=%d) flip:   " % NCV +
          "  ".join(f"mi{m}={_fmt(eigsh_flip(Hg, m))}" for m in MAXITERS))
    print("   power(ones) flip:         " +
          "  ".join(f"it{it}={minority_sign(gpu_power(Hg, it)):.3f}" for it in (2, 5, 20)))


if __name__ == "__main__":
    print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    print(f"GPU eigsh-flip reproducer  (ncv={NCV}, maxiters={MAXITERS})")

    report("UNIFORM weights",   to_gpu(build_H(frame_weights(0.0))))
    report("STRUCTURED weights", to_gpu(build_H(frame_weights(float(os.environ.get("CONTRAST", 0.85))))))

    # the real poster Gramian, if available
    import glob
    for cand in ("saveH.npz", os.path.join(os.path.dirname(__file__), "saveH.npz")):
        if os.path.exists(cand):
            from scipy import sparse as ssp
            Hreal = ssp.load_npz(cand).tocsr().astype(np.complex128)
            report(f"REAL saveH.npz ({cand})", to_gpu(Hreal))
            break
    else:
        print("\n(no saveH.npz found -- skipping the real-matrix check)")
