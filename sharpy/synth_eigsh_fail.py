#!/usr/bin/env python3
"""Synthetic reproducer for the sync-eigensolver (eigsh) sign-flip failure.

Builds a model frame-overlap Gramian H = diag(s) K diag(s) on a G x G scan grid,
where K_ij = exp(-d_ij^2/sigma^2) (d<=R) is the overlap kernel and s_i is the
per-frame energy (|z_i|, i.e. the object transmission seen by frame i). This
reproduces -- with no ptychography simulation and no GPU -- the eigsh
catastrophe studied on the real poster Gramian (sharpy/saveH.npz):

  near-degenerate overlap-graph cluster (tiny eig2/eig1 gap)         [knob: G, sigma, R]
  + NON-UNIFORM per-frame energy s_i (contrasty object, dark spots)  [knob: --contrast]
  => the consensus (top eigenvector) is NON-FLAT, so v0 = ones is a poor start
  => an under-converged Krylov solver (cupy eigsh / short Lanczos) returns a
     consensus + Fiedler mixture -> pi phase (sign) flip in omega = v/|v|.
  Power iteration AMPLIFIES toward the smooth consensus and never flips.

With UNIFORM s the consensus IS ~flat (ones ~ mode1) and nothing flips, even at
the same tiny gap -- which is why weak-contrast gold_balls sims never reproduced
the failure. The structured case matches saveH.npz: |<v1,ones>|^2 ~ 0.16,
Rayleigh(ones)/eig1 ~ 0.80, short-Lanczos flip ~ 0.2.

scipy/ARPACK does NOT expose the under-converged flip (it returns nothing or a
converged clean vector), so here we emulate the under-converged thick-restart
Lanczos (top Ritz from a short Krylov basis) -- the GPU twin synth_eigsh_fail_gpu.py
confirms on the real cupy eigsh.

  ENV: G=64 SIGMA=1.6 R=4 CONTRAST=0.85 SEED=0 python synth_eigsh_fail.py
"""
import os
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

G        = int(os.environ.get("G", 64))          # scan grid side; #frames = G^2
SIGMA    = float(os.environ.get("SIGMA", 1.6))   # overlap-kernel width (bigger = more overlap = larger gap)
R        = int(os.environ.get("R", 4))           # kernel cutoff radius in frames (~pi R^2 neighbors)
CONTRAST = float(os.environ.get("CONTRAST", 0.85))  # depth of the dark (absorbing) blobs in 0..1; 0 = uniform
SEED     = int(os.environ.get("SEED", 0))
N = G * G


def minority_sign(v):
    """Per-frame gauge flip metric: fraction of omega=v/|v| on the minority side
    after removing the global phase. ~0 = smooth/clean, ~0.5 = sign-flipping mode."""
    ph = v / (np.abs(v) + 1e-30)
    g = ph.sum()
    ph = ph * np.conj(g) / (np.abs(g) + 1e-30)
    return float(np.mean(ph.real < 0))


def frame_weights(contrast):
    """Per-frame energy s_i: bright background with a few absorbing (dark) blobs.
    contrast=0 -> uniform; ->1 -> deep dark spots (strongly non-uniform energy)."""
    if contrast <= 0:
        return np.ones(N)
    xs, ys = np.meshgrid(np.linspace(0, 1, G), np.linspace(0, 1, G), indexing="ij")
    blob = np.zeros((G, G))
    for cx, cy, a in [(0.5, 0.5, 0.9), (0.3, 0.7, 0.7), (0.75, 0.35, 0.6)]:
        blob += a * np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / 0.02))
    s = 1.0 - contrast * np.clip(blob, 0, 1)
    return np.clip(s, 0.1, 1.0).ravel()


def build_H(s):
    """Sparse Hermitian PSD overlap Gramian H = diag(s) K diag(s)."""
    xs, ys = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    xs = xs.ravel(); ys = ys.ravel()
    rows, cols, vals = [], [], []
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            d2 = dx * dx + dy * dy
            if d2 > R * R:
                continue
            xn, yn = xs + dx, ys + dy
            ok = (xn >= 0) & (xn < G) & (yn >= 0) & (yn < G)
            i = np.where(ok)[0]
            j = xn[ok] + yn[ok] * G
            rows.append(i); cols.append(j)
            vals.append(np.full(i.size, np.exp(-d2 / (SIGMA * SIGMA))))
    K = sparse.csr_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(N, N))
    D = sparse.diags(s)
    return (D @ K @ D).tocsr().astype(np.complex128)


def lanczos_topritz(A, v0, m):
    """Top Ritz vector from an m-step (full-reorth) Lanczos started at v0 --
    emulates an UNDER-CONVERGED thick-restart eigsh on a near-degenerate cluster."""
    n = A.shape[0]
    Q = np.zeros((n, m), complex); a = np.zeros(m); b = np.zeros(m)
    Q[:, 0] = v0 / np.linalg.norm(v0); bprev = 0.0; mm = m
    for j in range(m):
        z = A @ Q[:, j]
        if j > 0:
            z = z - bprev * Q[:, j - 1]
        a[j] = (Q[:, j].conj() @ z).real
        z = z - a[j] * Q[:, j]
        z = z - Q[:, :j + 1] @ (Q[:, :j + 1].conj().T @ z)   # full reorthogonalization
        bj = np.linalg.norm(z)
        if j + 1 < m:
            b[j] = bj
            if bj < 1e-12:
                mm = j + 1; break
            Q[:, j + 1] = z / bj; bprev = bj
    T = np.diag(a[:mm]) + np.diag(b[:mm - 1], 1) + np.diag(b[:mm - 1], -1)
    tw, tv = np.linalg.eigh(T)
    return Q[:, :mm] @ tv[:, np.argmax(tw)]


def power(A, v0, it):
    v = v0 / np.linalg.norm(v0)
    for _ in range(it):
        v = A @ v; v /= np.linalg.norm(v)
    return v


def report(name, s):
    H = build_H(s)
    w = np.sort(eigsh(H, k=6, which="LA", maxiter=20000, tol=0,
                      return_eigenvectors=False))[::-1]
    v1 = eigsh(H, k=1, which="LA", maxiter=20000, tol=0)[1][:, 0]
    ones = np.ones(N) + 0j
    ov1 = abs(v1.conj() @ (ones / np.sqrt(N))) ** 2
    rq = (ones.conj() @ (H @ ones)).real / N
    print(f"\n[{name}]  #frames={N}  nnz/row={H.nnz / N:.0f}")
    print(f"   eig2/eig1 = {w[1] / w[0]:.5f}   (near-degenerate cluster)")
    print(f"   |<v1,ones>|^2 = {ov1:.3f}   Rayleigh(ones)/eig1 = {rq / w[0]:.3f}   "
          f"({'ones is a GOOD start' if ov1 > 0.5 else 'ones is a POOR start'})")
    print("   short-Lanczos(ones) flip:  " +
          "  ".join(f"m{m}={minority_sign(lanczos_topritz(H, ones, m)):.3f}" for m in (3, 5, 10, 20)))
    print("   power(ones) flip:          " +
          "  ".join(f"it{it}={minority_sign(power(H, ones, it)):.3f}" for it in (2, 5, 20)))


if __name__ == "__main__":
    np.random.seed(SEED)
    print(f"Synthetic eigsh-flip reproducer  (G={G}, sigma={SIGMA}, R={R}, contrast={CONTRAST})")
    report("UNIFORM weights (contrast=0)", frame_weights(0.0))
    report(f"STRUCTURED weights (contrast={CONTRAST})", frame_weights(CONTRAST))
    print("\nExpected: UNIFORM stays clean (flip~0); STRUCTURED makes ones a poor start and")
    print("the short-Lanczos top Ritz FLIPS (flip~0.2) while power iteration stays clean (0.0).")
