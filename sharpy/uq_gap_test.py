"""UQ: does the sync-Gramian spectrum PREDICT recoverability of the global phase?

Hypothesis (Cramer-Rao reading): the connection Laplacian is (the low-freq-phase
block of) the Fisher information; per-mode variance ~ 1/(dose * lambda_i). The
worst mode = the Fiedler gap g of the normalized connection Laplacian (geometry-
only; uniform dose scales all edges equally). So the NOISE-WEIGHTED predictor for
the recovered low-band phase error is

    E_pred ∝ 1 / sqrt(PH * g)      (PH = photons/frame)

CAVEAT built in (from the eigsolver study): the raw preconditioned-adjacency gap
was NOT predictive of loop failure — the claim tested here is the weaker/correct
one: dose*gap predicts the recovered ERROR magnitude, checked by whether
E * sqrt(PH*g) collapses to ~constant across dose AND geometry.

Sweep: geometry (K, STEPD -> overlap) x dose (Poisson, photons/frame); run
AP + eigsh-sync (every iter) on the noisy data; report final low-band error,
the measured gap, and the collapse product. GPU or CPU (env NX etc. as in
sync_bandpass_test).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NX", "16")

import numpy as np
import sync_bandpass_test as T
import Operators
from Operators import (Illuminate_frames, Gramiam_calc, Precondition_calc, GPU, xp)
from scipy.sparse.linalg import eigsh as cpu_eigsh

MAXIT = int(os.environ.get("MAXIT", 150))
PH_LIST = [float(p) for p in os.environ.get("PHLIST", "30 100 300 1000 10000").split()]
GEOMS = [(20, 4), (40, 4), (20, 2)]      # (K, STEPD): STEPD=4 -> 75% overlap, 2 -> 50%
if os.environ.get("GEOMS"):              # e.g. GEOMS="20:4,40:4"
    GEOMS = [tuple(int(v) for v in g.split(":")) for g in os.environ["GEOMS"].split(",")]
rng = np.random.default_rng(11)


NMODES = int(os.environ.get("NMODES", 21))
WEIGHTED = os.environ.get("WEIGHTED", "1") == "1"


def mode_weights(ctx, V, lam):
    """MODE-TO-IMAGE weights: for graph mode v_m, the image perturbation from
    per-frame phases delta_phi = v_m is
        dO_m(r) = sum_i v_m,i * W_i(r) * O(r),   W_i = |probe|^2 footprint / norm,
    and w_m = ||low-band(dO_m)||^2 / ||O||^2. Replaces mode COUNTING with the
    actual weighting (the open UQ refinement)."""
    truth, probe, mapid = ctx["truth"], ctx["probe"], ctx["mapid"]
    tf = T.Splitc(truth, mapid) * (xp.abs(probe) ** 2)[None]
    w = np.zeros(V.shape[1])
    for m in range(V.shape[1]):
        vm = xp.asarray(V[:, m])
        dO = T.Overlapc(tf * vm[:, None, None].astype(xp.complex64),
                        ctx["Nx"], ctx["Ny"], mapid) / ctx["normalization"]
        D = xp.fft.fft2(dO)
        w[m] = float(xp.linalg.norm(D[ctx["low_mask"]])) ** 2 / ctx["Tnrm"] ** 2
    return w


def gramian_gap(ctx):
    """Spectrum of the preconditioned Gramian Hn from TRUTH frames (geometry only).
    Returns (fiedler_gap, tr21, tr_band):
      tr21    = sum_{i=2..21} 1/mu_i               (fixed-count trace, comparison)
      tr_band = sum over the LOW-BAND graph modes  (mode count ~ pi/4*(K*step/nx)^2
                = the # of per-frame-phase modes with spatial scale coarser than a
                frame, i.e. exactly the modes the q<1/nx error monitor integrates)
    Single gap = worst mode only; tr_band = the band-matched aggregate predictor."""
    truth_frames = Illuminate_frames(T.Splitc(ctx["truth"], ctx["mapid"]), ctx["probe"])
    if GPU:
        H = Operators.Gramiam_calc_cuda(truth_frames, ctx["Gramiam"], ctx["probe"],
                                        ctx["inorm_split"], ctx["frames_norm"])
        Hs = H.get()
    else:
        framesl = Illuminate_frames(truth_frames, xp.conj(ctx["probe"]))
        framesr = framesl * ctx["inorm_split"]
        Hs = Gramiam_calc(framesl, framesr, ctx["Gramiam"], ctx["frames_norm"])
    n = Hs.shape[0]
    m_band = int(np.ceil(0.25 * np.pi * (np.sqrt(n) * ctx["step"] / T.nx) ** 2)) + 1
    k = min(max(NMODES, m_band), n - 2)
    lam, V = cpu_eigsh(Hs.astype(np.complex128), k=k, which="LM")
    o = np.argsort(np.real(lam))[::-1]
    lam = np.real(lam)[o]; V = V[:, o]
    mu = 1.0 - lam[1:] / lam[0]                  # Laplacian eigenvalues, modes 2..k
    mu = np.maximum(mu, 1e-12)
    tr21 = float(np.sum(1.0 / mu[:NMODES - 1]))
    tr_band = float(np.sum(1.0 / mu[:m_band - 1]))
    tr_w = None
    if WEIGHTED:
        w = mode_weights(ctx, V[:, 1:], lam[1:])  # skip the consensus mode
        tr_w = float(np.sum(w / mu))              # E^2 ~ TrW / PH (mode-weighted)
    return float((lam[0] - lam[1]) / lam[0]), tr21, tr_band, m_band, tr_w


print(f"{'K':>3} {'STEPD':>5} {'ovl%':>4} {'frames':>6} | {'gap':>9} {'TrBand':>9} {'TrW':>9} {'M':>3} "
      f"| {'ph/frame':>8} | {'E_low':>8} {'E_high':>8} | {'E*s(PH*g)':>9} {'E*s(PH*nf/TrB)':>14} "
      f"{'E*s(PH/TrW)':>11}")
for K, stepd in GEOMS:
    T.STEPD = stepd
    ctx = T.build(K)
    data_clean = ctx["data"] + 0
    gap, tr21, trb, m_band, trw = gramian_gap(ctx)
    ovl = 100 * (1 - ctx["step"] / T.nx)
    REPS = int(os.environ.get("REPS", 1))
    for PH in PH_LIST:
        # Poisson noise at PH photons/frame (counts = data*s, s from clean mean);
        # REPS independent realizations -- single-draw scatter was shown to dominate
        # the apparent cross-geometry residual (PH=1000 ordering FLIPPED between draws).
        s = PH / (float(data_clean.sum()) / ctx["nframes"])
        dn = data_clean.get() if GPU else np.asarray(data_clean)
        Es = []
        for _ in range(REPS):
            ctx["data"] = xp.asarray(rng.poisson(dn * s).astype(np.float32) / s)
            ctx["frames_norm"] = Precondition_calc(ctx["data"], bw=ctx["Gramiam"]["bw"])
            c = T.run(ctx, 1, MAXIT, solver=T.eigsh_sync)
            Es.append((float(c[-1, 0]), float(c[-1, 1])))
        E_lo = float(np.mean([e[0] for e in Es])); E_sd = float(np.std([e[0] for e in Es]))
        E_hi = float(np.mean([e[1] for e in Es]))
        nf = ctx["nframes"]
        colw = E_lo * np.sqrt(PH / trw) if trw else float("nan")
        print(f"{K:>3} {stepd:>5} {ovl:>4.0f} {nf:>6} | {gap:>9.3e} {trb:>9.3e} "
              f"{(trw if trw else float('nan')):>9.3e} {m_band:>3} "
              f"| {PH:>8.0f} | {E_lo:>8.4f}±{E_sd:.4f} {E_hi:>8.4f} | {E_lo*np.sqrt(PH*gap):>9.4f} "
              f"{E_lo*np.sqrt(PH*nf/trb):>14.4f} {colw:>11.4f}")
    ctx["data"] = data_clean
print("\nCOLLAPSE TEST: single gap = worst mode; TrBand = band-matched mode COUNT; TrW = "
      "MODE-WEIGHTED trace (per-mode image weights w_m = low-band energy of the mode's "
      "image field; E^2 ~ TrW/PH, no ad-hoc nf normalization). The ~constant column across "
      "dose AND geometry is the UQ error bar.")
