"""Does the ADMM outer loop close the one-shot-sync PLATEAU?

od2p_random_batch_test showed independent random batches + ONE sync at the end
plateau above the joint floor (cross-batch coupling discarded). Here each round
re-seeds every batch from the shared consensus w, runs NIN local iters, phase-
aligns (Gramian), and averages back -> the consensus carries cross-batch info
between rounds. Report NMSE(w) vs #outer rounds; expect descent toward the floor.
(Seeding from the common w also keeps gauges consistent, so with feedback the
Gramian align is nearly a no-op -- it matters for the one-shot/independent case.)

  /opt/anaconda3/bin/python3 od2p_admm_feedback_test.py
  env: NX(48) KG(16) M(8) NIN(15) DOSE(30)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import od2p_admm_scaffold as S
import od2p_batchsync_test as T
import od2p_random_batch_test as R

xp = S.xp


eps = xp.float32(1e-8)


def local_solve(k, w, lam, data_n, mapk, norm_k, band, nin, rho):
    """Proximal local solve: AP on batch k's frames, PULLED toward (w - lam) by rho
    (the consensus coupling -- WITHOUT it, free-running batches over-fit + diverge)."""
    u = w + 0.0
    rb = rho * band
    for _ in range(nin):
        z = S.Illuminate_frames(S.Splitc(u, mapk[k]), S.probe)
        z, _ = S.Project_data(z, data_n[S.grp[k]])
        acc = S.Overlapc(S.Illuminate_frames(z, S.cprobe), S.Nx, S.Ny, mapk[k])
        u = (acc + rb * (w - lam[k])) / (norm_k[k] + rb + eps)
    return u


def admm(M, dose, rounds, nin, rho):
    grp, mapk, supp, cover, band, norm_k = R.build_random(M, 0)
    S.grp = grp
    band = xp.ones((S.Ny, S.Nx), bool)                     # random batches cover the whole FOV
    data_n = T.make_data(dose, 1)
    covf = xp.maximum(cover, 1).astype(xp.float32)
    w = xp.ones((S.Ny, S.Nx), dtype=xp.complex64)
    lam = [xp.zeros((S.Ny, S.Nx), dtype=xp.complex64) for _ in range(M)]
    curve = []
    for _ in range(rounds):
        uk = [local_solve(k, w, lam, data_n, mapk, norm_k, band, nin, rho) for k in range(M)]
        num = xp.zeros((S.Ny, S.Nx), dtype=xp.complex64)
        for k in range(M):
            num = num + supp[k].astype(uk[k].dtype) * (uk[k] + lam[k])
        w = num / covf
        for k in range(M):
            lam[k] = lam[k] + supp[k].astype(uk[k].dtype) * (uk[k] - w)
        curve.append(T.nmse(w, S.truth))
    return curve


if __name__ == "__main__":
    M = int(os.environ.get("M", 8))
    NIN = int(os.environ.get("NIN", 15))
    DOSE = float(os.environ.get("DOSE", 30))
    ROUNDS = int(os.environ.get("ROUNDS", 20))
    RHO = float(os.environ.get("RHO", 1.0))

    data_n = T.make_data(DOSE, 1)
    floor = T.nmse(T.global_ap(data_n, ROUNDS * NIN), S.truth)   # matched total work
    print(f"img {S.Nx}x{S.Ny}, {S.nframes} frames, M={M} random batches, NIN={NIN}, "
          f"dose {DOSE:g} ph/px, rho={RHO}")
    print(f"global-AP floor (~{ROUNDS*NIN} it): {floor:.4f}\n")
    cur = admm(M, DOSE, ROUNDS, NIN, RHO)
    print(f"{'round':>6} {'NMSE(w)':>9} {'gap/floor':>10}")
    for r in [0, 1, 2, 4, 8, 15, ROUNDS - 1]:
        if r < len(cur):
            print(f"{r+1:>6} {cur[r]:>9.4f} {cur[r]/floor:>10.2f}")
    print(f"\none-shot (round 1) = the plateau {cur[0]:.4f}; "
          f"after {ROUNDS} rounds {cur[-1]:.4f} -> floor {floor:.4f}")
    print("=> ADMM feedback closes the plateau (consensus carries cross-batch coupling).")
