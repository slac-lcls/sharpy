"""BLR revisit (thread #1): sweep the diffuser STRENGTH (phase rms, 0=smooth .. 1=pi) to
SEPARATE the two effects a speckle probe has -- the high-q CONDITIONING gain from the
low-q THRESHOLD loss -- and look for a mild-diffuser sweet spot.

The prior all-or-nothing BLR test (7c50050) found a full speckle probe RAISES the
long-range threshold. Rationale for a diffuser was better-conditioned overlaps (bigger
Gramian spectral gap = more robust sync + better position/high-q). Here we quantify BOTH
per diffuser strength: (a) conditioning = top-2 eigenvalue gap of the connection Gramian on
the truth exit-waves (bigger = better conditioned); (b) threshold = eigsh-sync low-band
NMSE across a dose sweep; also the high-band NMSE (resolution proxy). Fair: same |probe|
(known-good smooth spot) x amp*pi-rms band-limited random phase, identical flux.

*** CAVEAT: the high-band column here uses band_err's SINGLE GLOBAL GAUGE and does NOT show
a real speckle resolution gain -- under a PER-BAND gauge, speckle's high band equals smooth's
at every dose (~0.071). The apparent high-band improvement is a global-gauge coupling
artifact (see hybrid_probe_mix_test.py). The LOW-band threshold loss (a) is real: a speckle
probe is strictly worse. Trust the low-band column; treat the high-band column as diagnostic
only. ***

CPU / numpy authoritative.  /opt/anaconda3/bin/python3 blr_sweep_test.py
env: NX(16) K(24) STEPD(4) MAXIT(120) SE(1) REPS(4) LCORR(3)
     AMPLIST("0 0.25 0.5 1.0")  PHLIST("3 1 0.3 0.1 0.03")
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NX", "16")

import numpy as np
import sync_bandpass_test as T
from Operators import Illuminate_frames, Splitc, Gramiam_calc, Precondition_calc, xp
from blr_probe_noise_test import blr_phase, set_probe

nx = T.nx
K = int(os.environ.get("K", 24))
MAXIT = int(os.environ.get("MAXIT", 120))
SE = int(os.environ.get("SE", 1))
REPS = int(os.environ.get("REPS", 4))
LCORR = float(os.environ.get("LCORR", 3.0))
AMPLIST = [float(x) for x in os.environ.get("AMPLIST", "0 0.25 0.5 1.0").split()]
PHLIST = [float(p) for p in os.environ.get("PHLIST", "3 1 0.3 0.1 0.03").split()]


def blr_amp(probe, L, amp, seed=0):
    """same |probe|, phase = amp * (pi-rms band-limited random screen). amp=0 -> smooth."""
    p = np.asarray(probe)
    ph = amp * blr_phase(p.shape[0], L, seed)
    return xp.asarray((p * np.exp(1j * ph)).astype(np.complex64))


def spectral_gap(ctx):
    """top-2 eigenvalue gap of the connection Gramian on the TRUTH exit-waves = how
    well-conditioned the sync is for this probe (bigger = more robust). Returns lam1, gap."""
    from scipy.sparse.linalg import eigsh as arpack_eigsh
    probe = ctx["probe"]
    frames = Illuminate_frames(Splitc(ctx["truth"], ctx["mapid"]), probe)
    framesl = Illuminate_frames(frames, xp.conj(probe))
    framesr = framesl * ctx["inorm_split"]
    fn = Precondition_calc(ctx["clean"], bw=ctx["Gramiam"]["bw"])
    H = Gramiam_calc(framesl, framesr, ctx["Gramiam"], fn)
    Hc = (H.get() if hasattr(H, "get") else H)
    lam = np.sort(np.abs(arpack_eigsh(Hc.astype(np.complex128), k=2, which="LM",
                                      return_eigenvectors=False)))[::-1]
    return float(lam[0]), float((lam[0] - lam[1]) / (lam[0] + 1e-30))


if __name__ == "__main__":
    T.STEPD = int(os.environ.get("STEPD", 4))
    T.NUMITER = 5
    ctx = T.build(K)
    smooth = ctx["probe"] + 0
    rng = np.random.default_rng(1)
    ov = 100 * (1 - ctx["step"] / nx)
    print(f"K={K} ({ctx['nframes']} frames x {nx}), overlap {ov:.0f}%, LCORR={LCORR}px, "
          f"MAXIT={MAXIT} SE={SE} REPS={REPS}")
    hdr = " ".join(f"{p:>6g}" for p in PHLIST)
    print(f"{'amp*pi':>6} {'gap':>6} {'lam1':>7} | LOW-band(threshold) {hdr}  || HIGH-band {hdr}")
    for amp in AMPLIST:
        set_probe(ctx, blr_amp(smooth, LCORR, amp))
        clean = np.asarray(ctx["clean"])
        lam1, gap = spectral_gap(ctx)
        lows, highs = [], []
        for PH in PHLIST:
            scale = PH / (float(clean.sum()) / clean.size)
            lo, hi = [], []
            for r in range(REPS):
                noisy = xp.asarray(rng.poisson(clean * scale).astype(np.float32) / scale)
                ctx["data"] = noisy
                fn = Precondition_calc(noisy, bw=ctx["Gramiam"]["bw"])
                ctx["frames_norm"] = xp.where(xp.abs(fn) < 1e-6, xp.asarray(1e-6, fn.dtype), fn)
                curve = T.run(ctx, SE, MAXIT, solver=T.eigsh_sync)
                lo.append(curve[-1, 0]); hi.append(curve[-1, 1])
            lows.append(np.mean(lo)); highs.append(np.mean(hi))
        lostr = " ".join(f"{v:>6.3f}" for v in lows)
        histr = " ".join(f"{v:>6.3f}" for v in highs)
        print(f"{amp:>6.2f} {gap:>6.3f} {lam1:>7.3f} | {'':>19} {lostr}  || {'':>9} {histr}")
    print("\nEXPECT: if the diffuser genuinely conditions the graph, GAP should RISE with amp; "
          "if the low-q loss is the only real effect, LOW-band threshold worsens monotonically "
          "with amp while HIGH-band ~flat -> then the diffuser only costs, and any win must be "
          "sought in position retrieval (a separate test).")
