"""Does OD2P/HALO sync (overlapping subdomains, blended gauge) lower the photon
threshold for the long-range phase -- where naive HARD tiles failed (seams)?

Fix over tile_sync_noise_test.py: instead of one constant phase per hard tile (which
seams adjacent frames across boundaries), place coarse tile centres on an nc x nc grid,
give each frame BILINEAR weights to its 4 surrounding centres (a partition of unity =
multigrid prolongation), reconstruct each tile's object patch from its weighted frames
(photon aggregation), sync the tile patches (ones-anchored coarse Gramian -- the invit
lesson recurses to every level), and apply to each frame the BLENDED gauge
omega_frame = normalize(sum_t w_t omega_tile[t]). The blend is continuous across the
scan -> no seams. Compares none / single(full frame-sync) / halo, over a dose sweep.

CPU / numpy authoritative.  /opt/anaconda3/bin/python3 od2p_halosync_noise_test.py
env: NX(16) K(24) STEPD(4) NC(4 coarse grid side) MAXIT(120) SE(3) WARMUP(8)
     PHLIST("3 1 0.3 0.1 0.03") REPS(2)
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
NC = int(os.environ.get("NC", 4))                 # coarse grid: NC x NC tile centres
MAXIT = int(os.environ.get("MAXIT", 120))
SE = int(os.environ.get("SE", 3))
WARMUP = int(os.environ.get("WARMUP", 8))
PHLIST = [float(p) for p in os.environ.get("PHLIST", "3 1 0.3 0.1 0.03").split()]
REPS = int(os.environ.get("REPS", 2))


def bilinear_weights(K, nc):
    """(nf, nc*nc) partition-of-unity weights: each frame on the KxK grid gets
    bilinear weights to its 4 surrounding coarse centres. Rows sum to 1."""
    ii, jj = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
    ii = ii.ravel().astype(float); jj = jj.ravel().astype(float)
    u = ii / max(K - 1, 1) * (nc - 1)              # coarse coords in [0, nc-1]
    v = jj / max(K - 1, 1) * (nc - 1)
    a = np.clip(np.floor(u).astype(int), 0, nc - 2); fu = u - a
    b = np.clip(np.floor(v).astype(int), 0, nc - 2); fv = v - b
    W = np.zeros((ii.size, nc * nc))
    for (da, db, w) in ((0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)),
                        (0, 1, (1 - fu) * fv), (1, 1, fu * fv)):
        t = (a + da) * nc + (b + db)
        W[np.arange(ii.size), t] += w
    return W                                        # (nf, nc*nc)


def halo_sync(frames, ctx, W):
    """Overlapping-subdomain (halo) coarse sync with a bilinear-blended gauge."""
    cprobe, mapid, Nx, Ny = ctx["cprobe"], ctx["mapid"], ctx["Nx"], ctx["Ny"]
    nt = W.shape[1]
    illum = Illuminate_frames(frames, cprobe)                   # object contribution per frame
    patches = []
    for t in range(nt):
        wt = xp.asarray(W[:, t]).astype(xp.complex64)[:, None, None]
        num = Overlapc(illum * wt, Nx, Ny, mapid)
        den = Overlapc(ctx["absP2"] * wt, Nx, Ny, mapid)
        den = xp.where(xp.abs(den) < 1e-6 * float(xp.max(xp.abs(den)) + 1e-30), 1, den)
        patches.append((num / den).ravel())
    P = xp.stack(patches)                                       # (nt, Npix)
    Hc = np.asarray(P @ xp.conj(P).T)
    Hc = Hc - np.diag(np.diag(Hc))                              # adjacency (zero self)
    deg = np.abs(Hc).sum(axis=1)                                # DEGREE = total overlap
    deg = np.maximum(deg, 1e-30)
    s = 1.0 / np.sqrt(deg)
    Hn = s[:, None] * Hc * s[None, :]                           # D^-1/2 H D^-1/2
    om_t = np.linalg.solve(np.eye(nt) - Hn + 1e-3 * np.eye(nt),  # PSD Laplacian, ones-anchored
                           np.ones(nt, dtype=Hn.dtype))
    om_t = om_t / (np.abs(om_t) + 1e-30)
    om_t = om_t * np.conj(np.sum(om_t)) / (abs(np.sum(om_t)) + 1e-30)
    om_f = W @ om_t                                             # bilinear blend -> per frame
    om_f = xp.asarray(om_f / (np.abs(om_f) + 1e-30)).astype(xp.complex64)
    return frames * om_f[:, None, None]


def run(ctx, data, mode, W, maxit):
    probe, cprobe, mapid = ctx["probe"], ctx["cprobe"], ctx["mapid"]
    Nx, Ny = ctx["Nx"], ctx["Ny"]
    eig_reset()
    img = xp.ones((Ny, Nx), dtype=xp.complex64)
    frames = Illuminate_frames(Splitc(img, mapid), probe)
    for it in range(maxit):
        frames, _ = Project_data(frames, data)
        if mode != "none" and it >= WARMUP and it % SE == 0:
            if mode in ("single", "vcycle"):
                om = synchronize_frames_c(frames, probe, ctx["frames_norm"],
                                          ctx["inorm_split"], ctx["Gramiam"],
                                          ctx["Gramiam"]["bw"], 5)
                frames = frames * om                            # fine sync = the SMOOTHER
            if mode == "halo":
                frames = halo_sync(frames, ctx, W)
            if mode == "vcycle":
                # coarse-grid RESIDUAL correction ON TOP of the fine sync: after the
                # smoother the patches are aligned up to a slow residual, so the coarse
                # gauge is small (auto-vanishes at high dose) and only corrects the
                # noise-limited long modes at low dose. This is the multigrid V-cycle.
                frames = halo_sync(frames, ctx, W)
        img = Overlapc(Illuminate_frames(frames, cprobe), Nx, Ny, mapid) / ctx["normalization"]
        frames = Illuminate_frames(Splitc(img, mapid), probe)
    return T.band_err(ctx, img)[0]


if __name__ == "__main__":
    T.STEPD = int(os.environ.get("STEPD", 4))
    ctx = T.build(K)
    ctx["absP2"] = (xp.abs(ctx["probe"]) ** 2)[None].astype(xp.complex64) * xp.ones(
        (ctx["nframes"], 1, 1), xp.complex64)
    W = bilinear_weights(K, NC)
    clean = np.asarray(ctx["data"])
    rng = np.random.default_rng(0)
    ov = 100 * (1 - ctx["step"] / nx)
    print(f"K={K} ({ctx['nframes']} frames x {nx}), img {ctx['Nx']}^2, overlap {ov:.0f}%, "
          f"coarse {NC}x{NC} centres (bilinear blend), MAXIT={MAXIT} SE={SE} REPS={REPS}")
    print(f"{'ph/px':>7} | {'none':>8} {'single':>8} {'halo':>8} {'vcycle':>8}")
    for PH in PHLIST:
        scale = PH / (float(clean.sum()) / clean.size)
        acc = {m: [] for m in ("none", "single", "halo", "vcycle")}
        for r in range(REPS):
            noisy = xp.asarray(rng.poisson(clean * scale).astype(np.float32) / scale)
            ctx["data"] = noisy
            fn = Precondition_calc(noisy, bw=ctx["Gramiam"]["bw"])
            ctx["frames_norm"] = xp.where(xp.abs(fn) < 1e-6, xp.asarray(1e-6, fn.dtype), fn)
            for m in acc:
                try:
                    acc[m].append(run(ctx, noisy, m, W, MAXIT))
                except Exception as e:
                    acc[m].append(float("nan")); print(f"  !! {m} PH={PH}: {type(e).__name__}")
        mn = {m: float(np.nanmean(acc[m])) for m in acc}
        print(f"{PH:>7.2f} | {mn['none']:>8.3f} {mn['single']:>8.3f} {mn['halo']:>8.3f} "
              f"{mn['vcycle']:>8.3f}")
    print("\nEXPECT: halo continuous-gauge sync should NOT hurt at high dose (unlike hard tiles) "
          "and may beat single-level at low dose (aggregated coarse Gramian survives lower photons).")
