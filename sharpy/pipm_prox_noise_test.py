"""Does a finite-beta PROX data step (pAGM / pIPM, Chang-Enfedaque-Marchesini SIAM JIS
2019, DOI 10.1137/18M1188446) beat the HARD magnitude projection at LOW dose?

In hard-projection AP every noise metric collapses to |z|=sqrt(data); with a finite prox
weight beta the metric finally matters (the paper's z-update). Per-pixel closed forms on
the Fourier amplitude a=|Z|, r=sqrt(f):
  hard :  x = r
  pAGM :  argmin 1/2(x-r)^2        + beta/2 (x-a)^2  ->  x = (r + beta*a)/(1+beta)
  pIPM :  argmin 1/2 x^2 - f log x + beta/2 (x-a)^2  ->  (1+beta)x^2 - beta*a*x - f = 0
          x = (beta*a + sqrt(beta^2 a^2 + 4(1+beta) f)) / (2(1+beta))   (Poisson ML)
The pAGM arm isolates the RELAXATION effect (finite beta, Gaussian-amplitude metric); the
pIPM-minus-pAGM difference isolates the POISSON-METRIC effect. Prior in-loop result
(poisson-vst-lowcount): KL beats sqrt(I) ~15% only below ~0.1 counts/px -- this tests it
in the sync/long-range setting. beta -> 0 recovers hard for both.

CPU / numpy authoritative.  /opt/anaconda3/bin/python3 pipm_prox_noise_test.py
env: NX(16) K(24) STEPD(4) MAXIT(120) SE(1) REPS(3) BETAS("0.5 2") PHLIST("1 0.3 0.1 0.03")
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NX", "16")

import numpy as np
import sync_bandpass_test as T
import Operators
from Operators import (Illuminate_frames, Splitc, Overlapc, Precondition_calc,
                       synchronize_frames_c, eig_reset, xp)

nx = T.nx
K = int(os.environ.get("K", 24))
MAXIT = int(os.environ.get("MAXIT", 120))
SE = int(os.environ.get("SE", 1))
REPS = int(os.environ.get("REPS", 3))
BETAS = [float(b) for b in os.environ.get("BETAS", "0.5 2").split()]
PHLIST = [float(p) for p in os.environ.get("PHLIST", "1 0.3 0.1 0.03").split()]


def prox_data(frames, data, beta, metric):
    """Fourier-magnitude prox step (replaces Project_data's hard projection)."""
    Z = xp.fft.fft2(frames)
    a = xp.abs(Z)
    a = xp.where(a < 1e-12, xp.asarray(1e-12, a.dtype), a)
    r = xp.sqrt(data)
    if metric == "hard":
        x = r
    elif metric == "pagm":
        x = (r + beta * a) / (1 + beta)
    elif metric == "pipm":
        x = (beta * a + xp.sqrt(beta * beta * a * a + 4 * (1 + beta) * data)) / (2 * (1 + beta))
    else:
        raise ValueError(metric)
    return xp.fft.ifft2(Z * (x / a).astype(xp.float32))


def run_prox(ctx, metric, beta, se, maxit):
    """T.run with the prox data step; eigsh-sync arm."""
    Operators.Eigensolver = T.eigsh_sync
    eig_reset(); T._eigsh_v0["v"] = None
    probe, cprobe, mapid = ctx["probe"], ctx["cprobe"], ctx["mapid"]
    img = xp.ones((ctx["Ny"], ctx["Nx"]), dtype=xp.complex64)
    frames = Illuminate_frames(Splitc(img, mapid), probe)
    for it in range(maxit):
        frames = prox_data(frames, ctx["data"], beta, metric)
        if se and it % se == 0:
            omega = synchronize_frames_c(frames, probe, ctx["frames_norm"], ctx["inorm_split"],
                                         ctx["Gramiam"], ctx["Gramiam"]["bw"], 5)
            frames = frames * omega
        img = Overlapc(Illuminate_frames(frames, cprobe), ctx["Nx"], ctx["Ny"], mapid) / ctx["normalization"]
        frames = Illuminate_frames(Splitc(img, mapid), probe)
    Operators.Eigensolver = T._POWER
    return T.band_err(ctx, img)


if __name__ == "__main__":
    T.STEPD = int(os.environ.get("STEPD", 4))
    T.NUMITER = 5
    ctx = T.build(K)
    clean = np.asarray(ctx["data"])
    rng = np.random.default_rng(3)
    arms = [("hard", 0.0)] + [(m, b) for b in BETAS for m in ("pagm", "pipm")]
    ov = 100 * (1 - ctx["step"] / nx)
    print(f"K={K} ({ctx['nframes']} frames x {nx}), overlap {ov:.0f}%, MAXIT={MAXIT} "
          f"SE={SE} REPS={REPS} (eigsh-sync arm)")
    hdr = " ".join(f"{m}b{b:g}".rjust(9) if m != "hard" else f"{'hard':>9}" for m, b in arms)
    print(f"{'ph/px':>6} {'band':>4} | {hdr}")
    for PH in PHLIST:
        scale = PH / (float(clean.sum()) / clean.size)
        acc = {am: [] for am in range(len(arms))}
        for r in range(REPS):
            noisy = xp.asarray(rng.poisson(clean * scale).astype(np.float32) / scale)
            ctx["data"] = noisy
            fn = Precondition_calc(noisy, bw=ctx["Gramiam"]["bw"])
            ctx["frames_norm"] = xp.where(xp.abs(fn) < 1e-6, xp.asarray(1e-6, fn.dtype), fn)
            for ai, (m, b) in enumerate(arms):
                acc[ai].append(run_prox(ctx, m, b, SE, MAXIT))
        lo = [np.mean([v[0] for v in acc[ai]]) for ai in range(len(arms))]
        hi = [np.mean([v[1] for v in acc[ai]]) for ai in range(len(arms))]
        print(f"{PH:>6g} {'lo':>4} | " + " ".join(f"{v:>9.3f}" for v in lo))
        print(f"{'':>6} {'hi':>4} | " + " ".join(f"{v:>9.3f}" for v in hi))
    print("\nEXPECT: all arms ~equal at 1 ph/px; below ~0.1 ph/px pIPM should edge pAGM "
          "(Poisson-metric effect) and both may edge hard (relaxation effect, don't fit "
          "noise exactly). If pAGM==pIPM everywhere, the metric doesn't matter here and "
          "only the relaxation does.")
