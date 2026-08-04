"""RANDOM-interleaved frame batches (vs spatial tiles) + Gramian sync.

Each batch = a random 1/M subset of ALL frames -> covers the WHOLE FOV sparsely
(not a spatial tile). Consequences:
  - SYNC easier: every batch pair overlaps over the whole image -> dense, high-SNR
    batch-Gramian (vs a thin band).
  - RECON harder: sparse scan coverage per batch -> overlap ratio ~ /sqrt(M) ->
    each local recon degrades as M grows (ptycho conditioning), independent of sync.

Sweep M; report floor (global AP), naive avg, Gramian-synced avg, and the mean
SINGLE-batch recon NMSE (the conditioning limit). Prediction: gramian << naive at
all M (sync works, even better than spatial), but gramian tracks the per-batch
conditioning, which collapses past some M.

  python od2p_random_batch_test.py      (CPU; anaconda python has numpy)
  env: NX(48) KG(16) NIN(60) APIT(150) SEED(0)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import od2p_admm_scaffold as S
import od2p_batchsync_test as T

xp = S.xp
nframes, Nx, Ny = S.nframes, S.Nx, S.Ny


def build_random(M, seed):
    """M batches, each a round-robin slice of a random permutation -> whole-FOV, sparse."""
    perm = np.random.default_rng(seed).permutation(nframes)
    grp = [xp.asarray(np.sort(perm[k::M])) for k in range(M)]
    mapk = [S.mapid[grp[k]] for k in range(M)]
    supp, cover = [], xp.zeros((Ny, Nx), int)
    for k in range(M):
        s = xp.zeros(Nx * Ny, bool); s[xp.unique(mapk[k])] = True
        s = s.reshape(Ny, Nx); supp.append(s); cover += s.astype(int)
    band = cover >= 2
    norm_k = [S._norm(grp[k], mapk[k]) for k in range(M)]
    return grp, mapk, supp, cover, band, norm_k


if __name__ == "__main__":
    NIN = int(os.environ.get("NIN", 60))
    APIT = int(os.environ.get("APIT", 150))
    SEED = int(os.environ.get("SEED", 0))
    print(f"img {Nx}x{Ny}, {nframes} frames x {S.nx}  (RANDOM-interleaved batches)")

    for dose in [1e5, 10]:
        data_n = T.make_data(dose, SEED)
        floor = T.nmse(T.global_ap(data_n, APIT), S.truth)
        tag = "clean" if dose >= 1e5 else f"{dose:g} ph/px"
        print(f"\n=== dose {tag}   (global-AP floor {floor:.4f}) ===")
        print(f"{'M':>3} {'fr/batch':>9} {'band%':>7} {'1-batch':>9} {'naive':>8} {'gramian':>8}")
        for M in [2, 4, 8, 16]:
            grp, mapk, supp, cover, band, norm_k = build_random(M, SEED)
            S.grp = grp
            rng = np.random.default_rng(SEED + 1)
            ph = rng.uniform(0, 2 * np.pi, M)
            uk = [T.batch_solve(k, data_n, mapk, norm_k, NIN, ph[k]) for k in range(M)]
            # single-batch conditioning limit (avg over batches, phase absorbed by nmse scale)
            one = np.mean([T.nmse(uk[k], S.truth) for k in range(M)])
            naive = T.nmse(T.combine(uk, supp, cover, band, align=False), S.truth)
            gram = T.nmse(T.combine(uk, supp, cover, band, align=True), S.truth)
            bandpct = 100 * float(band.sum()) / (Nx * Ny)
            print(f"{M:>3} {nframes//M:>9} {bandpct:>6.0f}% {one:>9.4f} {naive:>8.4f} {gram:>8.4f}")
    print("\n=> gramian<<naive at all M = sync works (whole-FOV overlap); gramian ~ 1-batch"
          "\n   conditioning, which degrades with M (sparse coverage). Sync is NOT the limit.")
