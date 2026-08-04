"""Dynamic border migration (thread #4, Stage C): shift the tile-grid origin between OUTER
steps (red-black / half-shifted tilings) instead of a fixed partition. Answers SM's earlier
"alternate tile grouping shifting half way between tiles / sliding windows" question, and
models the cluster-OD2P dynamic-load-balance move (boundary frames migrate to a neighbor
tile under spare bandwidth). Two questions:
  (1) does shifting the borders IMPROVE the coarse-only recon (averaging seams + letting the
      inter-tile gauge see overlapping groupings -> closer to the global full-sync gold)?
  (2) migration COST = fraction of frames that change core-tile ownership per shift (small =
      "spare bandwidth" boundary migration; large = a full regroup).
Compares FIXED tiling vs a small shift (cheap, s=1) vs a half-tile shift (regroup, s=tilek/2).

CPU / numpy authoritative.  /opt/anaconda3/bin/python3 od2p_migrate_test.py
env: NX(16) K(24) STEPD(4) TILEK(6) HALO(1) NOUT(16) KLOC(10) MAXIT(160)
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
TILEK = int(os.environ.get("TILEK", 6))
HALO = int(os.environ.get("HALO", 1))
NOUT = int(os.environ.get("NOUT", 16))
KLOC = int(os.environ.get("KLOC", 10))
MAXIT = int(os.environ.get("MAXIT", 160))


def cell_index(off, tilek):
    """Core-cell index (ai, aj) each frame maps to for a given grid offset. A frame's
    co-tiled GROUP changes iff this index changes = a partition boundary swept over it."""
    oi, oj = off
    fi = np.arange(K * K) // K
    fj = np.arange(K * K) % K
    return (np.floor((fi - oi) / tilek).astype(int),
            np.floor((fj - oj) / tilek).astype(int))


def migration_frac(offA, offB, tilek):
    aiA, ajA = cell_index(offA, tilek)
    aiB, ajB = cell_index(offB, tilek)
    return float(np.mean((aiA != aiB) | (ajA != ajB)))   # frames swept by a moved boundary


def run_schedule(ctx, data, offsets, tilek, halo, n_out, k_loc):
    """OD2P block-Jacobi with a per-outer tile-grid OFFSET schedule (borders migrate)."""
    probe, cprobe = ctx["probe"], ctx["cprobe"]
    Nx, Ny = ctx["Nx"], ctx["Ny"]
    gobj = xp.ones((Ny, Nx), dtype=xp.complex64)
    tile_cache = {}
    for it in range(n_out):
        off = offsets[it % len(offsets)]
        if off not in tile_cache:
            tile_cache[off] = make_tiles(ctx, tilek, halo, off)
        tiles = tile_cache[off]
        covs = [t[3] for t in tiles]
        den = sum(covs); den = xp.where(xp.abs(den) < 1e-30, xp.complex64(1), den)
        objs = []
        for (idx, mapid_t, norm_safe, cov_t) in tiles:
            data_t = data[idx]
            fr = Illuminate_frames(Splitc(gobj, mapid_t), probe)
            for _ in range(k_loc):
                fr, _ = Project_data(fr, data_t)
                obj_t = Overlapc(Illuminate_frames(fr, cprobe), Nx, Ny, mapid_t) / norm_safe
                fr = Illuminate_frames(Splitc(obj_t, mapid_t), probe)
            objs.append(obj_t)
        g = coarse_gauge(objs, covs)
        objs = [o * xp.conj(g[k]) for k, o in enumerate(objs)]
        gobj = sum(c * o for c, o in zip(covs, objs)) / den
    return T.band_err(ctx, gobj)


if __name__ == "__main__":
    T.STEPD = int(os.environ.get("STEPD", 4))
    T.NUMITER = 5
    ctx = T.build(K)
    data = ctx["data"]
    ctx["frames_norm"] = Precondition_calc(data, bw=ctx["Gramiam"]["bw"])
    gfull = T.run(ctx, 1, MAXIT, solver=T.eigsh_sync)[-1]

    h = TILEK // 2
    scheds = {
        "fixed":       [(0, 0)],
        f"shift s=1":  [(0, 0), (1, 0), (0, 1), (1, 1)],
        f"shift s={h}": [(0, 0), (h, 0), (0, h), (h, h)],
    }
    print(f"K={K} ({ctx['nframes']} fr), TILEK={TILEK} HALO={HALO} NOUT={NOUT} KLOC={KLOC}")
    print(f"GLOBAL full-sync gold: lo={gfull[0]:.5f} hi={gfull[1]:.5f}")
    print(f"{'schedule':>10} | {'lo':>8} {'hi':>8} | {'migrate/outer':>13}")
    for name, offs in scheds.items():
        be = run_schedule(ctx, data, offs, TILEK, HALO, NOUT, KLOC)
        if len(offs) > 1:
            mig = np.mean([migration_frac(offs[i], offs[(i + 1) % len(offs)], TILEK)
                           for i in range(len(offs))])
        else:
            mig = 0.0
        print(f"{name:>10} | {be[0]:>8.5f} {be[1]:>8.5f} | {100 * mig:>11.0f}%")
    print("\nEXPECT: shifting the tile borders between outers should LOWER the coarse-only "
          "residual toward the global gold (seams averaged, gauge sees overlapping groupings). "
          "A small s=1 shift migrates only boundary frames (cheap, 'spare bandwidth'); a "
          "half-tile s shift regroups more (higher migration) -- see which buys the accuracy.")
