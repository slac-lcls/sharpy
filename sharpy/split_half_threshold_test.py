"""Truth-free noise-threshold detector: split-half (FRC-style) low-band correlation.

Above the BBP threshold the sync recovers the true long-range phase, so two INDEPENDENT
noise realizations at the same dose reconstruct the SAME low-frequency phase -> their
low-band correlation C is high. Below threshold both are noise -> C collapses. So C is a
truth-free predictor of recoverability. This checks that C tracks the actual (truth-based)
low-band NMSE across a dose sweep -- i.e. you can tell, without truth, when the long-range
phase is trustworthy.

C = |sum_low F1 conj(F2)| / sqrt(sum_low|F1|^2 sum_low|F2|^2)  (global gauge factors out).

CPU / numpy authoritative.  /opt/anaconda3/bin/python3 split_half_threshold_test.py
env: NX(16) K(24) STEPD(4) MAXIT(120) SE(1) REPS(3) PHLIST("3 1 0.3 0.1 0.03")
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NX", "16")

import numpy as np
import sync_bandpass_test as T
from Operators import Precondition_calc, xp

K = int(os.environ.get("K", 24))
MAXIT = int(os.environ.get("MAXIT", 120))
SE = int(os.environ.get("SE", 1))
REPS = int(os.environ.get("REPS", 3))
PHLIST = [float(p) for p in os.environ.get("PHLIST", "3 1 0.3 0.1 0.03").split()]


def recon_img(ctx, noisy):
    ctx["data"] = noisy
    fn = Precondition_calc(noisy, bw=ctx["Gramiam"]["bw"])
    ctx["frames_norm"] = xp.where(xp.abs(fn) < 1e-6, xp.asarray(1e-6, fn.dtype), fn)
    # return the final image (run the loop, then rebuild the image once more)
    from Operators import Illuminate_frames, Splitc, Overlapc
    T.eig_reset()
    probe, cprobe, mapid = ctx["probe"], ctx["cprobe"], ctx["mapid"]
    img = xp.ones((ctx["Ny"], ctx["Nx"]), dtype=xp.complex64)
    frames = Illuminate_frames(Splitc(img, mapid), probe)
    import Operators
    Operators.Eigensolver = T.eigsh_sync
    for it in range(MAXIT):
        frames, _ = T.Project_data(frames, noisy)
        if it % SE == 0:
            om = T.synchronize_frames_c(frames, probe, ctx["frames_norm"], ctx["inorm_split"],
                                        ctx["Gramiam"], ctx["Gramiam"]["bw"], 5)
            frames = frames * om
        img = Overlapc(Illuminate_frames(frames, cprobe), ctx["Nx"], ctx["Ny"], mapid) / ctx["normalization"]
        frames = Illuminate_frames(Splitc(img, mapid), probe)
    Operators.Eigensolver = T._POWER
    return img


def lowband_corr(img1, img2, mask):
    F1 = xp.fft.fft2(img1)[mask]; F2 = xp.fft.fft2(img2)[mask]
    num = xp.abs(xp.vdot(F1, F2))
    den = xp.sqrt(xp.vdot(F1, F1).real * xp.vdot(F2, F2).real) + 1e-30
    return float(num / den)


if __name__ == "__main__":
    T.STEPD = int(os.environ.get("STEPD", 4))
    T.NUMITER = 5
    ctx = T.build(K)
    clean = np.asarray(ctx["data"])
    mask = ctx["low_mask"]
    rng = np.random.default_rng(2)
    print(f"K={K} ({ctx['nframes']} frames x {nx if (nx:=T.nx) else 0}), overlap "
          f"{100*(1-ctx['step']/T.nx):.0f}%, MAXIT={MAXIT} REPS={REPS}")
    print(f"{'ph/px':>7} | {'true lf-NMSE':>12} | {'split-half C':>13}  (C high = recoverable)")
    for PH in PHLIST:
        scale = PH / (float(clean.sum()) / clean.size)
        nmses, corrs = [], []
        for r in range(REPS):
            n1 = xp.asarray(rng.poisson(clean * scale).astype(np.float32) / scale)
            n2 = xp.asarray(rng.poisson(clean * scale).astype(np.float32) / scale)
            img1 = recon_img(ctx, n1); img2 = recon_img(ctx, n2)
            corrs.append(lowband_corr(img1, img2, mask))
            nmses.append(T.band_err(ctx, img1)[0])
        print(f"{PH:>7.2f} | {np.mean(nmses):>12.3f} | {np.mean(corrs):>13.3f}")
    print("\nEXPECT: C (truth-free) should stay ~1 while the true lf-NMSE is low, and COLLAPSE "
          "at the same dose the true NMSE blows up -> C is a runtime recoverability readout.")
