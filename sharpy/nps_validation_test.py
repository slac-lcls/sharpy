"""NPS quantify/predict validation (the poisson-vst-lowcount plan's Q2).

Claim to test (Pelz-style red noise + our Fisher reading): the object-domain
reconstruction NOISE power spectrum is COLORED —
    NPS(q) ~ A/PH * [white amplitude floor]           for q > ~1/nx,
    NPS(q) ~ B/(PH * mu(q))  [red phase branch]       for q < ~1/nx,
where mu(q) is the connection-Laplacian eigenvalue curve measured from the SYNC
GRAMIAN (graph modes mapped to spatial frequency by 2D mode counting,
q_m ~ sqrt(m/pi)/(K*step)). Both branches scale 1/dose (shape dose-invariant).

Method: REPS independent Poisson realizations -> converged AP+eigsh-sync recons,
gauge-aligned; noise = recon - mean(recons) (removes the common bias); measured
NPS = azimuthal average of mean |F noise|^2. Compare: (1) low-band log-log slope
vs the -2-ish of 1/mu(q); (2) dose ratio NPS_1/NPS_2 vs PH_2/PH_1 per bin;
(3) shape match of the red branch against 1/mu(q) from the truth-H spectrum.

GPU (REPS recons):  python -u nps_validation_test.py
env: NX(16) K(20) STEPD(4) MAXIT(150) REPS(16) PHLIST("300 3000") NBINS(12)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NX", "16")

import numpy as np
import sync_bandpass_test as T
import Operators
from Operators import Illuminate_frames, Precondition_calc, GPU, xp
from scipy.sparse.linalg import eigsh as cpu_eigsh

MAXIT = int(os.environ.get("MAXIT", 150))
REPS = int(os.environ.get("REPS", 16))
PHLIST = [float(p) for p in os.environ.get("PHLIST", "300 3000").split()]
NBINS = int(os.environ.get("NBINS", 12))
rng = np.random.default_rng(31)


def recon(ctx):
    Operators.Eigensolver = T.eigsh_sync
    T.eig_reset(); T._eigsh_v0["v"] = None
    probe, cprobe, mapid = ctx["probe"], ctx["cprobe"], ctx["mapid"]
    img = xp.ones((ctx["Ny"], ctx["Nx"]), dtype=xp.complex64)
    frames = Illuminate_frames(T.Splitc(img, mapid), probe)
    for it in range(MAXIT):
        frames, _ = T.Project_data(frames, ctx["data"])
        om = T.synchronize_frames_c(frames, probe, ctx["frames_norm"], ctx["inorm_split"],
                                    ctx["Gramiam"], ctx["Gramiam"]["bw"], 5)
        frames = frames * om
        img = T.Overlapc(Illuminate_frames(frames, cprobe), ctx["Nx"], ctx["Ny"],
                         mapid) / ctx["normalization"]
        frames = Illuminate_frames(T.Splitc(img, mapid), probe)
    Operators.Eigensolver = T._POWER
    s = xp.vdot(img, ctx["truth"]) / (xp.vdot(img, img) + 1e-30)
    return (s * img).astype(xp.complex64)


if __name__ == "__main__":
    K = int(os.environ.get("K", 20))
    ctx = T.build(K)
    dc = ctx["data"] + 0
    # Gramian spectrum -> mu(q) prediction curve (truth-H, same path as uq_gap_test)
    tf = Illuminate_frames(T.Splitc(ctx["truth"], ctx["mapid"]), ctx["probe"])
    if GPU:
        Hs = Operators.Gramiam_calc_cuda(tf, ctx["Gramiam"], ctx["probe"],
                                         ctx["inorm_split"], ctx["frames_norm"]).get()
    else:
        fl = Illuminate_frames(tf, np.conj(ctx["probe"]))
        Hs = Operators.Gramiam_calc(fl, fl * ctx["inorm_split"], ctx["Gramiam"], ctx["frames_norm"])
    kmod = min(80, ctx["nframes"] - 2)
    lam = cpu_eigsh(Hs.astype(np.complex128), k=kmod, which="LM", return_eigenvectors=False)
    lam = np.sort(np.real(lam))[::-1]
    mu = np.maximum(1.0 - lam[1:] / lam[0], 1e-12)                # ascending in m
    q_m = np.sqrt(np.arange(1, mu.size + 1) / np.pi) / (K * ctx["step"])  # mode->q map

    # radial bins over the image plane
    qy, qx = np.meshgrid(np.fft.fftfreq(ctx["Ny"]), np.fft.fftfreq(ctx["Nx"]), indexing="ij")
    qr = np.sqrt(qx ** 2 + qy ** 2)
    edges = np.logspace(np.log10(1.0 / ctx["Nx"]), np.log10(0.5), NBINS + 1)
    qmid = np.sqrt(edges[:-1] * edges[1:])
    binid = np.digitize(qr.ravel(), edges) - 1

    nps = {}
    for PH in PHLIST:
        s = PH / (float(dc.sum()) / ctx["nframes"])
        dn = dc.get() if GPU else np.asarray(dc)
        recs = []
        for r in range(REPS):
            ctx["data"] = xp.asarray(rng.poisson(dn * s).astype(np.float32) / s)
            ctx["frames_norm"] = Precondition_calc(ctx["data"], bw=ctx["Gramiam"]["bw"])
            recs.append(recon(ctx))
        R = xp.stack(recs)
        M = R.mean(axis=0)
        noise = R - M[None]
        P = xp.mean(xp.abs(xp.fft.fft2(noise, axes=(1, 2))) ** 2, axis=0)
        P = (P.get() if GPU else np.asarray(P)).ravel()
        prof = np.array([P[binid == b].mean() if np.any(binid == b) else np.nan
                         for b in range(NBINS)])
        nps[PH] = prof
        print(f"PH={PH:.0f}: recon-noise NPS computed over {REPS} reps")

    print(f"\n{'q':>8} | " + " | ".join(f"NPS PH={PH:.0f}" for PH in PHLIST) +
          f" | {'ratio':>7} (expect {PHLIST[1]/PHLIST[0]:.0f}) | {'pred 1/mu(q)':>12}")
    predq = np.interp(qmid, q_m, 1.0 / mu, left=np.nan, right=np.nan)
    for b in range(NBINS):
        r = nps[PHLIST[0]][b] / nps[PHLIST[1]][b] if nps[PHLIST[1]][b] else np.nan
        print(f"{qmid[b]:>8.4f} | " + " | ".join(f"{nps[PH][b]:>10.3e}" for PH in PHLIST) +
              f" | {r:>7.2f} | {predq[b]:>12.3e}")
    # low-band log-log slope (q < 1/nx) + shape match against 1/mu(q)
    for PH in PHLIST:
        sel = (qmid < 1.0 / T.nx) & np.isfinite(nps[PH])
        if sel.sum() > 2:
            sl = np.polyfit(np.log(qmid[sel]), np.log(nps[PH][sel]), 1)[0]
            sel2 = sel & np.isfinite(predq)
            ratio = nps[PH][sel2] / predq[sel2]
            print(f"PH={PH:.0f}: low-band slope {sl:.2f}; NPS/(1/mu) shape spread "
                  f"{ratio.max()/ratio.min():.2f}x over {sel2.sum()} bins")
    print("\nEXPECT: red low-band branch tracking 1/mu(q) (slope ~ -2), white floor above "
          "q~1/nx, per-bin dose ratio ~ PH2/PH1 (both branches 1/dose).")
