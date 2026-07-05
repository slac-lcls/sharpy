"""Does HIERARCHICAL (tiled) sync lower the photon-noise threshold for long-range phase?

Claim (Stefano): a single frame-frame Gramian edge integrates only one overlap band, so
its SNR is photon-starved; if instead you SYNC OBJECT PATCHES (each tile reconstructed
from its frames via Overlapc, aggregating all that tile's photons), the coarse tile-to-
tile edges have far higher SNR -> the longest-wavelength consensus (the mode that fails
FIRST as dose drops) should be recoverable at LOWER dose. AP is the fine smoother
(intra-tile); the coarse patch-sync carries the inter-tile long-range phase.

Compares, across a photons/pixel sweep, the low-freq-band NMSE for:
  none | single (full frame-frame sync) | tile (fixed tiles) | tile-shift (half-shifted
  tiling, alternating each sync -- the red-black multigrid trick that kills tile seams).

CPU / numpy authoritative for the threshold.  /opt/anaconda3/bin/python3 tile_sync_noise_test.py
env: NX(16) K(24) STEPD(4) TK(6 frames/tile side) MAXIT(120) SE(3)
     PHLIST("3 1 0.3 0.1 0.03") REPS(2) QCUT(0.12 lf band)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NX", "16")

import numpy as np
import sync_bandpass_test as T
from Operators import (Illuminate_frames, Splitc, Overlapc, Project_data,
                       synchronize_frames_c, Precondition_calc, eig_reset, xp)

nx = T.nx
K = int(os.environ.get("K", 24))
TK = int(os.environ.get("TK", 6))                 # frames per tile side
MAXIT = int(os.environ.get("MAXIT", 120))
SE = int(os.environ.get("SE", 3))
PHLIST = [float(p) for p in os.environ.get("PHLIST", "3 1 0.3 0.1 0.03").split()]
REPS = int(os.environ.get("REPS", 2))
QCUT = float(os.environ.get("QCUT", 0.12))
WARMUP = int(os.environ.get("WARMUP", 8))         # AP iters before any sync (image must form)


def tile_ids(K, tk, shift):
    """Assign each frame (row-major on the KxK grid) to a tile, optionally shifted
    by half a tile (wrapping) -- the alternating red-black grid."""
    ii, jj = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
    off = tk // 2 if shift else 0
    ti = ((ii + off) % K) // tk
    tj = ((jj + off) % K) // tk
    ntx = (K + tk - 1) // tk
    return (ti * ntx + tj).ravel(), ntx * ((K + tk - 1) // tk)


def coarse_tile_sync(frames, ctx, tid, ntiles):
    """Reconstruct each tile's object patch (Overlapc over its frames = photon
    aggregation), build the dense tile-to-tile patch Gramian, take its consensus
    eigenvector, and apply the per-tile phase to every frame in the tile."""
    cprobe, mapid = ctx["cprobe"], ctx["mapid"]
    Nx, Ny = ctx["Nx"], ctx["Ny"]
    patches = []
    for t in range(ntiles):
        m = tid == t
        if not m.any():
            patches.append(None); continue
        illum = Illuminate_frames(frames[m], cprobe)
        num = Overlapc(illum, Nx, Ny, mapid[m])
        den = Overlapc(ctx["absP2"][m], Nx, Ny, mapid[m])
        den = xp.where(xp.abs(den) < 1e-6 * float(xp.max(xp.abs(den)) + 1e-30), 1, den)
        patches.append((num / den).astype(xp.complex64))
    idx = [t for t in range(ntiles) if patches[t] is not None]
    m2 = len(idx)
    P = xp.stack([patches[t].ravel() for t in idx])              # (m2, Npix)
    Hc = np.asarray(P @ xp.conj(P).T)                            # dense tile Gramian
    d = np.sqrt(np.abs(np.diag(Hc)) + 1e-30)
    Hn = Hc / (d[:, None] * d[None, :])
    Hn = Hn - np.diag(np.diag(Hn))                               # normalized adjacency
    # ONES-ANCHORED inverse iteration on the coarse Laplacian L = I - Hn: the same
    # cluster-ambiguity that breaks a plain top-eigenvector at the frame level recurs
    # at the tile level (tiles at high overlap are near-degenerate), so anchor at ones.
    L = np.eye(m2) - Hn + 1e-3 * np.eye(m2)
    om = np.linalg.solve(L, np.ones(m2, dtype=Hn.dtype))
    om = om / (np.abs(om) + 1e-30)
    om = om * np.conj(np.sum(om)) / (abs(np.sum(om)) + 1e-30)    # fix global phase
    out = frames + 0.0
    for k, t in enumerate(idx):
        out[tid == t] = frames[tid == t] * xp.complex64(om[k])
    return out


def run(ctx, data, mode, maxit):
    probe, cprobe, mapid = ctx["probe"], ctx["cprobe"], ctx["mapid"]
    Nx, Ny = ctx["Nx"], ctx["Ny"]
    eig_reset()
    img = xp.ones((Ny, Nx), dtype=xp.complex64)
    frames = Illuminate_frames(Splitc(img, mapid), probe)
    tid0, nt0 = tile_ids(K, TK, False)
    tid1, nt1 = tile_ids(K, TK, True)
    for it in range(maxit):
        frames, _ = Project_data(frames, data)
        if mode != "none" and it >= WARMUP and it % SE == 0:
            if mode == "single":
                om = synchronize_frames_c(frames, probe, ctx["frames_norm"],
                                          ctx["inorm_split"], ctx["Gramiam"],
                                          ctx["Gramiam"]["bw"], 5)
                frames = frames * om
            elif mode == "tile":
                frames = coarse_tile_sync(frames, ctx, tid0, nt0)
            elif mode == "tile-shift":
                use0 = (it // SE) % 2 == 0
                frames = coarse_tile_sync(frames, ctx, tid0 if use0 else tid1,
                                          nt0 if use0 else nt1)
        img = Overlapc(Illuminate_frames(frames, cprobe), Nx, Ny, mapid) / ctx["normalization"]
        frames = Illuminate_frames(Splitc(img, mapid), probe)
    return _coarse_err(ctx, img)                                 # TILE-SCALE band NMSE


def _coarse_err(ctx, img):
    """NMSE in the very-low-freq band coarser than a tile (q < 1/tile_footprint) --
    the tile-to-tile relative phase the coarse sync targets, isolated from the
    within-tile phase a per-tile constant cannot represent."""
    tile_px = TK * ctx["step"]                                  # object footprint of a tile (px)
    qy, qx = xp.meshgrid(xp.fft.fftfreq(ctx["Ny"]), xp.fft.fftfreq(ctx["Nx"]), indexing="ij")
    mask = xp.sqrt(qx ** 2 + qy ** 2) < (1.0 / tile_px)
    s = xp.vdot(img, ctx["truth"]) / (xp.vdot(img, img) + 1e-30)
    D = xp.fft.fft2(s * img - ctx["truth"])
    return float(xp.linalg.norm(D[mask]) / (xp.linalg.norm(xp.fft.fft2(ctx["truth"])[mask]) + 1e-30))


if __name__ == "__main__":
    T.STEPD = int(os.environ.get("STEPD", 4))
    ctx = T.build(K)
    ctx["absP2"] = (xp.abs(ctx["probe"]) ** 2)[None].astype(xp.complex64) * xp.ones(
        (ctx["nframes"], 1, 1), xp.complex64)
    clean = ctx["clean"] if "clean" in ctx else ctx["data"]
    clean = np.asarray(clean)
    rng = np.random.default_rng(0)
    ov = 100 * (1 - ctx["step"] / nx)
    print(f"K={K} ({ctx['nframes']} frames x {nx}), img {ctx['Nx']}^2, overlap {ov:.0f}%, "
          f"tile {TK}x{TK} frames -> {(K + TK - 1) // TK}^2 tiles, low band q<{QCUT}, "
          f"MAXIT={MAXIT} SE={SE} REPS={REPS}")
    print(f"{'ph/px':>7} | {'none':>8} {'single':>8} {'tile':>8} {'tile-shift':>10}")
    for PH in PHLIST:
        scale = PH / (float(clean.sum()) / clean.size)          # avg photons/pixel = PH
        acc = {m: [] for m in ("none", "single", "tile", "tile-shift")}
        for r in range(REPS):
            noisy = xp.asarray(rng.poisson(clean * scale).astype(np.float32) / scale)
            ctx["data"] = noisy
            fn = Precondition_calc(noisy, bw=ctx["Gramiam"]["bw"])
            fn = xp.where(xp.abs(fn) < 1e-6, xp.asarray(1e-6, fn.dtype), fn)   # floor zero norms
            ctx["frames_norm"] = fn
            for m in acc:
                try:
                    acc[m].append(run(ctx, noisy, m, MAXIT))
                except Exception as e:
                    acc[m].append(float("nan"))
                    print(f"  !! {m} PH={PH} rep{r} failed: {type(e).__name__}")
        mean = {m: float(np.mean(acc[m])) for m in acc}
        print(f"{PH:>7.2f} | {mean['none']:>8.3f} {mean['single']:>8.3f} "
              f"{mean['tile']:>8.3f} {mean['tile-shift']:>10.3f}")
    print("\nEXPECT: if aggregation lowers the threshold, tile/tile-shift recover the "
          "low-band (< none) at doses where single-level sync has already collapsed to ~none.")
