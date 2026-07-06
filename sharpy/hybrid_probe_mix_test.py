"""HYBRID probe-mix dataset: fraction phi of frames use the SMOOTH probe, 1-phi use the
SPECKLE/BLR probe -- at FIXED total dose -- reconstructed BAND-SPLIT (smooth subset + sync
-> low band; speckle subset, no sync -> high band; gauge-align + Fourier-merge). Original
hypothesis: a mix Pareto-beats both pure datasets by feeding each band from the probe that
concentrates photons there.

*** RESULT: NEGATIVE -- the apparent "win" was a metric ARTIFACT (do not cite as a recipe). ***
The single-global-gauge band-error used here (band_err: one complex gauge s=<img,truth>/
<img,img> for the whole image) COUPLES the two bands: a recon with a poor high band gets its
global phase pulled off, INFLATING the reported low-band error. Band-split supplies a good
high band that pins the gauge, so the SAME low-band coefficients score better -- a bookkeeping
effect, not a reconstruction gain (the merged low-band coeffs are bit-identical to the
smooth-subset's). Under a PER-BAND gauge (each band its own optimal phase) the win vanishes:
a speckle probe gives ZERO high-band gain at any dose (per-band high ~0.071 for smooth AND
speckle at 1/0.3/0.1/0.03 ph/px), and a pure smooth probe is strictly better on BOTH bands.
So: speckle probes are strictly worse (kill the low/long-range band, no resolution gain);
there is NO photon-allocation trade. The STILL-VALID probe lever is the EXPANDED smooth probe
(focus_probe_noise_test.py), which lowers the long-range threshold (a low-band result under
one consistent gauge). Kept as the worked example of the gauge pitfall: for any band-split /
cross-probe / two-quality-band comparison, use a PER-BAND (or fixed-convention) gauge, not one
global gauge. See also blr_sweep_test.py (its high-band column is the same artifact).

CPU / numpy authoritative.  /opt/anaconda3/bin/python3 hybrid_probe_mix_test.py
env: NX(16) K(24) STEPD(4) MAXIT(120) SE(1) REPS(3) LCORR(3)
     PHILIST("1 0.75 0.5 0.25 0")  PHLIST("1 0.3")
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NX", "16")

import numpy as np
import sync_bandpass_test as T
from Operators import (Splitc, Overlapc, Project_data, Gramiam_calc, Gramiam_plan,
                       Precondition_calc, eig_reset, xp)
from blr_probe_noise_test import blr_from

nx = T.nx
K = int(os.environ.get("K", 24))
MAXIT = int(os.environ.get("MAXIT", 120))
SE = int(os.environ.get("SE", 1))
REPS = int(os.environ.get("REPS", 3))
LCORR = float(os.environ.get("LCORR", 3.0))
PHILIST = [float(x) for x in os.environ.get("PHILIST", "1 0.75 0.5 0.25 0").split()]
PHLIST = [float(p) for p in os.environ.get("PHLIST", "1 0.3").split()]
# SYNCMODE: "all" = Gramian over every frame pair (speckle-touching edges are phase-
# decorrelated -> noise edges); "smooth" = sync the SMOOTH-SMOOTH subgraph only (coherent
# edges), speckle frames keep omega=1 and inherit the gauge through the image blend;
# "split" = BAND-SPLIT reconstruction: recon the two subsets SEPARATELY (smooth+sync ->
# low band; speckle, no sync -> high band), gauge-align, merge in Fourier at the low_mask
# edge. Rationale: in a joint blend the speckle frames inject scrambled low-q with full
# |P|^2 weight -- separating the recons keeps each band fed only by the probe that
# concentrates photons there (the allocator law, done right).
SYNCMODE = os.environ.get("SYNCMODE", "all")


def build_mixed(ctx, smooth, blr, mask_smooth):
    """Per-frame probe stack P (nf,nx,nx) + clean data + normalization for the mix."""
    nf = ctx["nframes"]
    P = np.empty((nf, nx, nx), dtype=np.complex64)
    P[mask_smooth] = np.asarray(smooth)
    P[~mask_smooth] = np.asarray(blr)
    P = xp.asarray(P)
    clean = (xp.abs(xp.fft.fft2(Splitc(ctx["truth"], ctx["mapid"]) * P)) ** 2).astype(xp.float32)
    nrm = Overlapc((xp.abs(P) ** 2).astype(xp.complex64), ctx["Nx"], ctx["Ny"], ctx["mapid"])
    nrm = xp.where(xp.abs(nrm) < 1e-6 * float(xp.max(xp.abs(nrm))), xp.complex64(1), nrm)
    return P, clean, nrm, Splitc(1.0 / nrm, ctx["mapid"])


def run_mixed(ctx, P, data, nrm, inorm_split, frames_norm, se, maxit, sub=None):
    """AP + eigsh-sync with a PER-FRAME probe stack (sync inlined: the Gramian only needs
    the conj-illuminated frame products, so a stack works where the single-probe
    synchronize_frames_c signature doesn't). sub=(mask, subplan, sub_frames_norm) syncs
    only that coherent subgraph; the rest keep omega=1 (gauge flows via the image blend)."""
    cP = xp.conj(P)
    mapid, Nx, Ny = ctx["mapid"], ctx["Nx"], ctx["Ny"]
    eig_reset(); T._eigsh_v0["v"] = None
    img = xp.ones((Ny, Nx), dtype=xp.complex64)
    frames = Splitc(img, mapid) * P
    for it in range(maxit):
        frames, _ = Project_data(frames, data)
        if se and it % se == 0:
            if sub is None:
                framesl = frames * cP
                framesr = framesl * inorm_split
                H = Gramiam_calc(framesl, framesr, ctx["Gramiam"], frames_norm)
                om = T.eigsh_sync(H, 5)
            else:
                mask, splan, sfn = sub
                framesl = frames[mask] * cP[mask]
                framesr = framesl * inorm_split[mask]
                Hs = Gramiam_calc(framesl, framesr, splan, sfn)
                om_s = T.eigsh_sync(Hs, 5)
                om = xp.ones((frames.shape[0], 1, 1), dtype=xp.complex64)
                om[xp.asarray(mask)] = om_s
            frames = frames * om
        img = Overlapc(frames * cP, Nx, Ny, mapid) / nrm
        frames = Splitc(img, mapid) * P
    return T.band_err(ctx, img)


def run_subset(ctx, P, data, mask, plan, fn, se, maxit):
    """AP recon from ONE frame subset only (own normalization on the global grid);
    optional sync via a subset plan. Returns the image."""
    m = xp.asarray(mask)
    mapid_s = ctx["mapid"][m]
    Ps = P[m]; cPs = xp.conj(Ps)
    ds = data[m]
    nrm = Overlapc((xp.abs(Ps) ** 2).astype(xp.complex64), ctx["Nx"], ctx["Ny"], mapid_s)
    nrm = xp.where(xp.abs(nrm) < 1e-6 * float(xp.max(xp.abs(nrm))), xp.complex64(1), nrm)
    inorm_s = Splitc(1.0 / nrm, mapid_s)
    eig_reset(); T._eigsh_v0["v"] = None
    img = xp.ones((ctx["Ny"], ctx["Nx"]), dtype=xp.complex64)
    fr = Splitc(img, mapid_s) * Ps
    for it in range(maxit):
        fr, _ = Project_data(fr, ds)
        if se and plan is not None and it % se == 0:
            framesl = fr * cPs
            framesr = framesl * inorm_s
            H = Gramiam_calc(framesl, framesr, plan, fn)
            fr = fr * T.eigsh_sync(H, 5)
        img = Overlapc(fr * cPs, ctx["Nx"], ctx["Ny"], mapid_s) / nrm
        fr = Splitc(img, mapid_s) * Ps
    return img


def band_merge(ctx, img_lo_src, img_hi_src):
    """Gauge-align the high-band source to the low-band source, merge in Fourier at the
    low_mask edge."""
    s = xp.vdot(img_hi_src, img_lo_src) / (xp.vdot(img_hi_src, img_hi_src) + 1e-30)
    F = xp.fft.fft2(img_lo_src) * ctx["low_mask"] + xp.fft.fft2(s * img_hi_src) * (~ctx["low_mask"])
    return xp.fft.ifft2(F)


if __name__ == "__main__":
    T.STEPD = int(os.environ.get("STEPD", 4))
    T.NUMITER = 5
    ctx = T.build(K)
    nf = ctx["nframes"]
    smooth = ctx["probe"] + 0
    blr = blr_from(smooth, LCORR)                       # same |probe|, speckle phase
    rng = np.random.default_rng(2)
    arng = np.random.default_rng(7)                     # probe-assignment rng (fixed)
    ov = 100 * (1 - ctx["step"] / nx)
    print(f"K={K} ({nf} frames x {nx}), overlap {ov:.0f}%, LCORR={LCORR}px, "
          f"MAXIT={MAXIT} SE={SE} REPS={REPS} (phi = fraction SMOOTH frames)")
    print(f"{'phi':>5} {'ph/px':>6} | {'low':>7} {'high':>7} {'total':>7}")
    g = np.arange(K) * ctx["step"]
    txa, tya = np.meshgrid(g, g, indexing="ij")
    txa = txa.ravel().astype(np.float64); tya = tya.ravel().astype(np.float64)
    for phi in PHILIST:
        mask = arng.permutation(nf) < int(round(phi * nf))
        P, clean_x, nrm, inorm = build_mixed(ctx, smooth, blr, mask)
        clean = np.asarray(clean_x)
        splan = None
        if SYNCMODE in ("smooth", "split") and mask.sum() > 1:
            splan = Gramiam_plan(xp.asarray(txa[mask]), xp.asarray(tya[mask]),
                                 int(mask.sum()), nx, nx, ctx["Nx"], ctx["Ny"])
        for PH in PHLIST:
            scale = PH / (float(clean.sum()) / clean.size)
            los, his = [], []
            for r in range(REPS):
                noisy = xp.asarray(rng.poisson(clean * scale).astype(np.float32) / scale)
                if SYNCMODE == "split" and 1 < mask.sum() < nf:
                    sfn = Precondition_calc(noisy[xp.asarray(mask)], bw=splan["bw"])
                    sfn = xp.where(xp.abs(sfn) < 1e-6, xp.asarray(1e-6, sfn.dtype), sfn)
                    img_S = run_subset(ctx, P, noisy, mask, splan, sfn, SE, MAXIT)
                    img_B = run_subset(ctx, P, noisy, ~mask, None, None, 0, MAXIT)
                    lo, hi = T.band_err(ctx, band_merge(ctx, img_S, img_B))
                    los.append(lo); his.append(hi)
                    continue
                if splan is not None and SYNCMODE == "smooth":
                    sfn = Precondition_calc(noisy[xp.asarray(mask)], bw=splan["bw"])
                    sfn = xp.where(xp.abs(sfn) < 1e-6, xp.asarray(1e-6, sfn.dtype), sfn)
                    sub = (mask, splan, sfn)
                    se_eff = SE
                elif SYNCMODE == "smooth":
                    sub, se_eff = None, 0                  # phi=0: no coherent subgraph
                else:
                    sub, se_eff = None, SE
                fn = Precondition_calc(noisy, bw=ctx["Gramiam"]["bw"])
                fn = xp.where(xp.abs(fn) < 1e-6, xp.asarray(1e-6, fn.dtype), fn)
                lo, hi = run_mixed(ctx, P, noisy, nrm, inorm, fn, se_eff, MAXIT, sub=sub)
                los.append(lo); his.append(hi)
            lo, hi = np.mean(los), np.mean(his)
            print(f"{phi:>5.2f} {PH:>6g} | {lo:>7.3f} {hi:>7.3f} {np.sqrt(lo**2 + hi**2):>7.3f}")
    print("\nEXPECT: pure smooth (phi=1) = best low band, pure BLR (phi=0) = best high band "
          "at low dose. If a mid phi holds the low band near phi=1 (smooth subgraph still "
          "percolates the sync) while pulling the high band toward phi=0, the mix Pareto-"
          "beats both pure probes (probe-diversity dose allocation).")
