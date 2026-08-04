"""Stage D: consensus-ADMM duals on the coarse-only OD2P -- does dual MEMORY close the
block-Jacobi gap (lf 0.02) toward the global full-sync gold (0.0006)?

Chang-Enfedaque-Marchesini (SIAM J. Imaging Sci. 12(1):153, 2019, DOI 10.1137/18M1188446)
solves ptychography by generalized ADMM whose Step-4 multiplier update ACCUMULATES the
constraint residual across outers. Our Stage-A block-Jacobi (od2p_coarse_only_test.py) has
no such memory: each outer re-blends tile objects from scratch, so it stalls at a fixed
point where local data-fit and the blend disagree. Here we add the consensus-ADMM tether:

    min sum_t f_t(o_t)  s.t.  o_t = R_t x   (R_t = tile footprint restriction)
    o_t-step: K_loc AP iters on tile data, each followed by a proximal pull toward
              (R_t x - lam_t) with weight RHO*cov_t (the augmented-Lagrangian tether);
    x-step:   coverage-weighted average of (o_t + lam_t);
    dual:     lam_t += o_t - R_t x        (per-tile dual field on the tile footprint).

Arms: bj (Stage A block-Jacobi + gauge), admm (duals, no gauge), admm+g (duals + inter-tile
gauge -- gauge fixes the fast rotation, duals grind the slow residual). Noise-free
(convergence question). CPU/numpy.  /opt/anaconda3/bin/python3 od2p_admm_test.py
env: NX(16) K(24) STEPD(4) TILEK(4) HALO(1) NOUT(16) KLOC(10) RHO(0.5) MAXIT(160)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NX", "16")

import numpy as np
import sync_bandpass_test as T
from Operators import (Illuminate_frames, Splitc, Overlapc, Project_data,
                       Precondition_calc, xp)
from od2p_coarse_only_test import make_tiles, coarse_gauge

nx = T.nx
K = int(os.environ.get("K", 24))
TILEK = int(os.environ.get("TILEK", 4))
HALO = int(os.environ.get("HALO", 1))
NOUT = int(os.environ.get("NOUT", 16))
KLOC = int(os.environ.get("KLOC", 10))
RHO = float(os.environ.get("RHO", 0.5))
MAXIT = int(os.environ.get("MAXIT", 160))


def run_admm(ctx, tiles, data, n_out, k_loc, rho, use_dual, use_gauge,
             warmup=None, eta=0.5):
    """Scaled-form consensus ADMM. Scalar rho (tether weight per KLOC iter, O(1) units);
    duals start after `warmup` outers (dual ascent on an unconverged local solve locks in
    garbage); gauge rotates o_t AND lam_t jointly (the constraint o_t = R_t x is
    gauge-covariant -- rotating only o_t makes gauge and dual fight)."""
    probe, cprobe = ctx["probe"], ctx["cprobe"]
    Nx, Ny = ctx["Nx"], ctx["Ny"]
    x = xp.ones((Ny, Nx), dtype=xp.complex64)                 # global consensus object
    covs = [t[3] for t in tiles]
    den = sum(covs); den = xp.where(xp.abs(den) < 1e-30, xp.complex64(1), den)
    masks = [(xp.abs(c) > 0).astype(xp.complex64) for c in covs]
    lams = [xp.zeros((Ny, Nx), dtype=xp.complex64) for _ in tiles]   # scaled duals u_t
    if warmup is None:
        warmup = max(2, n_out // 4)
    for it in range(n_out):
        dual_on = use_dual and it >= warmup
        objs = []
        for k, (idx, mapid_t, norm_safe, cov_t) in enumerate(tiles):
            data_t = data[idx]
            target = (x - lams[k]) if dual_on else x          # tether center (x - u_t)
            fr = Illuminate_frames(Splitc(target, mapid_t), probe)
            for _ in range(k_loc):
                fr, _ = Project_data(fr, data_t)
                o = Overlapc(Illuminate_frames(fr, cprobe), Nx, Ny, mapid_t) / norm_safe
                fr = Illuminate_frames(Splitc(o, mapid_t), probe)
            if dual_on and rho > 0:
                # aug-Lagrangian o_t prox applied ONCE after the (inexact) local solve:
                # blending every inner iter compounds to (1/(1+rho))^k_loc and freezes the
                # local solve onto the stale consensus -- tether at the end, not throughout
                o = (o + rho * target * masks[k]) / (1 + rho * masks[k])
            objs.append(o)
        if use_gauge:
            g = coarse_gauge(objs, covs)
            objs = [o * xp.conj(g[k]) for k, o in enumerate(objs)]
            if dual_on:
                lams = [l * xp.conj(g[k]) for k, l in enumerate(lams)]  # rotate duals too
        x = sum(c * (o + (lams[k] if dual_on else 0)) for k, (c, o) in
                enumerate(zip(covs, objs))) / den
        if dual_on:
            for k in range(len(tiles)):
                lams[k] = lams[k] + eta * masks[k] * (objs[k] - x)   # damped dual ascent
    return T.band_err(ctx, x)


if __name__ == "__main__":
    T.STEPD = int(os.environ.get("STEPD", 4))
    T.NUMITER = 5
    ctx = T.build(K)
    data = ctx["data"]
    ctx["frames_norm"] = Precondition_calc(data, bw=ctx["Gramiam"]["bw"])
    gfull = T.run(ctx, 1, MAXIT, solver=T.eigsh_sync)[-1]
    tiles = make_tiles(ctx, TILEK, HALO)
    print(f"K={K} ({ctx['nframes']} fr), TILEK={TILEK} ({len(tiles)} tiles) HALO={HALO} "
          f"NOUT={NOUT} KLOC={KLOC} RHO={RHO}")
    print(f"GLOBAL full-sync gold: lo={gfull[0]:.5f} hi={gfull[1]:.5f}")
    print(f"{'arm':>8} | {'lo':>8} {'hi':>8}")
    for name, (ud, ug) in (("bj+g", (False, True)), ("admm", (True, False)),
                           ("admm+g", (True, True))):
        be = run_admm(ctx, tiles, data, NOUT, KLOC, RHO, ud, ug)
        print(f"{name:>8} | {be[0]:>8.5f} {be[1]:>8.5f}")
    print("\nEXPECT: admm+g < bj+g (dual memory accumulates the inter-tile disagreement -> "
          "consensus grinds toward the global gold instead of stalling at the blend fixed "
          "point); admm without gauge shows how much the gauge still buys (fast rotation).")
