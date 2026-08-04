"""Coarse-only OD2P (thread #4, Stage A): is INTER-tile sync alone enough -- no intra-tile
sync -- when tiles are small? And how small must a tile be? (SM's insight: a small tile has
few frames -> big spectral gap -> AP converges its internal low-freq phase on its own; the
slow mode is the GLOBAL one across tiles, so one gauge per tile should suffice.)

Faithful block-Jacobi OD2P on the shared image grid (Splitc/Overlapc subset by mapid[idx]):
partition the KxK scan into overlapping tiles; each OUTER step, every tile runs K_local AP
iters LOCALLY on its own frames + object patch (the streaming model: tile resident, no
global coupling during local iters), then the tiles' patches are reconciled by an INTER-tile
coarse Gramian gauge (one phase per tile, degree-normalized ones-anchored -- the invit
lesson at the coarse level) and blended (partition-of-unity by coverage) into the global
object. Sweeps tile size. Compares to the GLOBAL references (full per-frame eigsh-sync =
gold; no sync). If od2p-coarse == global-full for small tiles, intra-tile sync is provably
unnecessary; the tile size where coarse starts to lag = the answer to "how small".

CPU / numpy authoritative.  /opt/anaconda3/bin/python3 od2p_coarse_only_test.py
env: NX(16) K(24) STEPD(4) TILEKS("3 4 6 8 12") HALO(1) NOUT(14) KLOC(10) MAXIT(140) SEED(0)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NX", "16")

import numpy as np
import sync_bandpass_test as T
from Operators import (Illuminate_frames, Splitc, Overlapc, Project_data,
                       Precondition_calc, xp)

nx = T.nx
K = int(os.environ.get("K", 24))
TILEKS = [int(x) for x in os.environ.get("TILEKS", "3 4 6 8 12").split()]
HALO = int(os.environ.get("HALO", 1))
NOUT = int(os.environ.get("NOUT", 14))
KLOC = int(os.environ.get("KLOC", 10))
MAXIT = int(os.environ.get("MAXIT", 140))


def make_tiles(ctx, tilek, halo, off=(0, 0)):
    """Overlapping tiles over the KxK scan (frame f -> grid (f//K, f%K)). Returns per-tile
    (idx, mapid_t, norm_t_safe, cov_t) on the global image grid. `off` shifts the tile-grid
    origin (for shifted/red-black tilings + dynamic border migration, Stage C)."""
    mapid, probe = ctx["mapid"], ctx["probe"]
    Nx, Ny = ctx["Nx"], ctx["Ny"]
    absP2 = (xp.abs(probe) ** 2).astype(xp.complex64)
    fi = np.arange(K * K) // K
    fj = np.arange(K * K) % K
    oi, oj = off
    base = int(np.ceil(K / tilek))
    rng = range(-1, base + 1) if (oi or oj) else range(base)   # extend only when shifted
    tiles = []
    for a in rng:
        for b in rng:
            lo_i, hi_i = oi + a * tilek - halo, oi + (a + 1) * tilek + halo
            lo_j, hi_j = oj + b * tilek - halo, oj + (b + 1) * tilek + halo
            sel = (fi >= lo_i) & (fi < hi_i) & (fj >= lo_j) & (fj < hi_j)
            idx = np.where(sel)[0]
            if idx.size == 0:
                continue
            mapid_t = mapid[idx]
            nt = idx.size
            norm_t = Overlapc(xp.broadcast_to(absP2, (nt, nx, nx)), Nx, Ny, mapid_t)
            mx = float(xp.max(xp.abs(norm_t))) + 1e-30
            cov_t = xp.abs(norm_t)
            norm_safe = xp.where(xp.abs(norm_t) < 1e-6 * mx, xp.complex64(1), norm_t)
            tiles.append((idx, mapid_t, norm_safe, cov_t))
    return tiles


def coarse_gauge(objs, covs):
    """Per-tile phase gauge from the coverage-weighted inter-tile overlap Gramian
    (degree-normalized, ones-anchored). objs/covs: lists of (Ny,Nx)."""
    Tn = len(objs)
    WO = xp.stack([(c * o).ravel() for c, o in zip(covs, objs)])   # (Tn, Npix), weight=coverage
    G = np.asarray(WO @ xp.conj(WO).T)
    A = G - np.diag(np.diag(G))
    deg = np.maximum(np.abs(A).sum(1), 1e-30)
    s = 1.0 / np.sqrt(deg)
    Hn = s[:, None] * A * s[None, :]
    g = np.linalg.solve(np.eye(Tn) - Hn + 1e-3 * np.eye(Tn), np.ones(Tn, dtype=Hn.dtype))
    g = g / (np.abs(g) + 1e-30)
    g = g * np.conj(g.sum()) / (abs(g.sum()) + 1e-30)
    return xp.asarray(g).astype(xp.complex64)


def od2p_run(ctx, tiles, data, mode, n_out, k_loc):
    probe, cprobe = ctx["probe"], ctx["cprobe"]
    Nx, Ny = ctx["Nx"], ctx["Ny"]
    gobj = xp.ones((Ny, Nx), dtype=xp.complex64)
    covs = [t[3] for t in tiles]
    den = sum(covs); den = xp.where(xp.abs(den) < 1e-30, xp.complex64(1), den)
    for _ in range(n_out):
        objs = []
        for (idx, mapid_t, norm_safe, cov_t) in tiles:
            data_t = data[idx]
            fr = Illuminate_frames(Splitc(gobj, mapid_t), probe)
            for _ in range(k_loc):
                fr, _ = Project_data(fr, data_t)
                obj_t = Overlapc(Illuminate_frames(fr, cprobe), Nx, Ny, mapid_t) / norm_safe
                fr = Illuminate_frames(Splitc(obj_t, mapid_t), probe)
            objs.append(obj_t)
        if mode == "coarse":
            g = coarse_gauge(objs, covs)
            objs = [o * xp.conj(g[k]) for k, o in enumerate(objs)]   # conj: REMOVE the tile phase
        gobj = sum(c * o for c, o in zip(covs, objs)) / den
    return T.band_err(ctx, gobj)


if __name__ == "__main__":
    T.STEPD = int(os.environ.get("STEPD", 4))
    T.NUMITER = 5
    ctx = T.build(K)
    data = ctx["data"]
    ctx["frames_norm"] = Precondition_calc(data, bw=ctx["Gramiam"]["bw"])
    # global references (noise-free convergence): full per-frame eigsh-sync vs no sync
    gfull = T.run(ctx, 1, MAXIT, solver=T.eigsh_sync)[-1]
    gnone = T.run(ctx, 0, MAXIT)[-1]
    ov = 100 * (1 - ctx["step"] / nx)
    print(f"K={K} ({ctx['nframes']} fr x {nx}), img {ctx['Nx']}^2, overlap {ov:.0f}%, "
          f"NOUT={NOUT} KLOC={KLOC} (={NOUT*KLOC} local iters) vs MAXIT={MAXIT}, HALO={HALO}")
    print(f"GLOBAL refs:  full-sync lo={gfull[0]:.4f} hi={gfull[1]:.4f}   |   "
          f"no-sync lo={gnone[0]:.4f} hi={gnone[1]:.4f}")
    print(f"{'tilek':>6} {'tiles':>6} {'fr/tile':>8} | {'od2p-coarse (lo/hi)':>20} | "
          f"{'od2p-none (lo/hi)':>18}")
    for tilek in TILEKS:
        tiles = make_tiles(ctx, tilek, HALO)
        frpt = int(np.mean([t[0].size for t in tiles]))
        cl = od2p_run(ctx, tiles, data, "coarse", NOUT, KLOC)
        nn = od2p_run(ctx, tiles, data, "none", NOUT, KLOC)
        print(f"{tilek:>6} {len(tiles):>6} {frpt:>8} | {cl[0]:>9.4f} {cl[1]:>9.4f} | "
              f"{nn[0]:>8.4f} {nn[1]:>8.4f}")
    print("\nEXPECT: od2p-coarse (inter-tile gauge only, NO intra sync) should MATCH global "
          "full-sync for SMALL tiles -> intra-tile sync unnecessary; as tiles grow, coarse "
          "starts to lag (residual intra-tile slow phase) = the tile-size threshold. od2p-none "
          "(no inter-tile sync) should be worse -> inter-tile sync is the piece you cannot drop.")
