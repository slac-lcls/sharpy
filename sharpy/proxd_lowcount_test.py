"""Poisson/KL prox vs amplitude (AGM) vs Anscombe-VST at LOW COUNTS (~0.01 ph/px).

The plan's Q1 (poisson-vst-lowcount): non-blind, known positions, dense overlap,
MANY very noisy frames — does the KL/IPM prox really beat the sqrt-I amplitude
metric, or does a variance-stabilizing transform do just as well? The metric can
only matter in a PROXIMAL/relaxed data step (hard projection collapses all models
to |Fz| = sqrt(c)), so the loop here is
    Z <- prox_{tau, metric}(Z; counts)   in Fourier space (ProxD_noise),
    frames <- overlap consensus          (exact, known probe).
Sync OFF and only mild low-freq phase: at these doses the Gramian is far below its
noise threshold (sync-noise-feasibility) and would only confound the metric test.

Data are POISSON COUNTS in photon units (the object is rescaled so mean counts/px
hits the target; KL needs counts, not ADU). Anscombe arm: pointwise Newton prox of
(1/2)(2*sqrt(m^2+3/8) - 2*sqrt(c+3/8))^2 + (tau/2)(m-a)^2.

CPU: /opt/anaconda3/bin/python3 proxd_lowcount_test.py
env: NX(16) K(20) STEPD(4) MAXIT(200) REPS(3) PHR(1.0 rad phase) CBAR("0.003 0.01 0.03 0.1 1")
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NX", "16")

import numpy as np
import sync_bandpass_test as T
from Operators import ProxD_noise, xp

MAXIT = int(os.environ.get("MAXIT", 200))
REPS = int(os.environ.get("REPS", 3))
PHR = float(os.environ.get("PHR", 1.0))
CBAR = [float(c) for c in os.environ.get("CBAR", "0.003 0.01 0.03 0.1 1").split()]
TAUS = [0.5, 1.0, 2.0]
rng = np.random.default_rng(23)


def vst_prox(x, c, tau, eps=1e-8):
    """Pointwise prox of the Anscombe-stabilized amplitude misfit:
    argmin_m 0.5*(2*sqrt(m^2+3/8) - t)^2 + (tau/2)*(m-a)^2,  t = 2*sqrt(c+3/8)."""
    a = xp.sqrt(xp.real(x) ** 2 + xp.imag(x) ** 2 + eps)
    t = 2.0 * xp.sqrt(c + 0.375)
    m = a + 0.0
    for _ in range(8):                       # vectorized Newton
        s = xp.sqrt(m * m + 0.375)
        f = (2.0 * s - t) * (2.0 * m / s) + tau * (m - a)
        fp = 4.0 - (2.0 * t * 0.375) / (s ** 3) + tau   # d/dm[(2s-t)*2m/s] = 4 - 2t*(3/8)/s^3
        m = xp.maximum(m - f / xp.maximum(fp, 1e-6), 0.0)
    return x * (m / a)


def run(ctx, counts, method, tau):
    probe, cprobe, mapid = ctx["probe"], ctx["cprobe"], ctx["mapid"]
    img = xp.ones((ctx["Ny"], ctx["Nx"]), dtype=xp.complex64)
    frames = T.Illuminate_frames(T.Splitc(img, mapid), probe)
    for it in range(MAXIT):
        Z = xp.fft.fft2(frames)
        if method == "hard":
            Z = ProxD_noise(Z, counts, tau=None)
        elif method == "vst":
            Z = vst_prox(Z, counts, tau)
        else:
            Z = ProxD_noise(Z, counts, tau=tau, metric=method)
        frames = xp.fft.ifft2(Z).astype(xp.complex64)
        img = T.Overlapc(T.Illuminate_frames(frames, cprobe), ctx["Nx"], ctx["Ny"],
                         mapid) / ctx["normalization"]
        frames = T.Illuminate_frames(T.Splitc(img, mapid), probe)
    s = xp.vdot(img, ctx["truth"]) / (xp.vdot(img, img) + 1e-30)
    return float(xp.linalg.norm(s * img - ctx["truth"]) / xp.linalg.norm(ctx["truth"]))


if __name__ == "__main__":
    K = int(os.environ.get("K", 20))
    orig = T.phantom
    T.phantom = lambda Nx, Ny, seed=0: np.abs(orig(Nx, Ny, seed)) * np.exp(
        1j * PHR / 2.5 * np.angle(orig(Nx, Ny, seed))).astype(np.complex64)
    ctx = T.build(K)
    T.phantom = orig
    data0 = ctx["data"] + 0
    print(f"K={K}: {ctx['nframes']} frames x {T.nx}, img {ctx['Nx']}^2, phase +-{PHR} rad, "
          f"MAXIT={MAXIT}, REPS={REPS} (sync OFF; NMSE = total, gauge-aligned)")
    print(f"{'c/px':>7} | {'hard(AP)':>15} | {'AGM tau*':>15} | {'KL tau*':>15} | {'VST tau*':>15}")
    for cbar in CBAR:
        alpha2 = cbar / float(data0.mean())
        ctx["data"] = data0 * alpha2                     # photon units: mean counts/px = cbar
        ctx["truth"] = (ctx["truth"] * np.sqrt(alpha2)).astype(xp.complex64)
        ctx["Tnrm"] = float(xp.linalg.norm(xp.fft.fft2(ctx["truth"])))
        clean = ctx["data"] + 0
        res = {}
        for method in ("hard", "amplitude", "poisson", "vst"):
            taus = [None] if method == "hard" else TAUS
            best = (np.inf, None)
            for tau in taus:
                es = []
                r2 = np.random.default_rng(101)          # same noise across methods
                for _ in range(REPS):
                    cts = r2.poisson(np.asarray(clean)).astype(np.float32)
                    ctx["data"] = xp.asarray(cts)
                    es.append(run(ctx, ctx["data"], method, tau))
                m = float(np.mean(es))
                if m < best[0]:
                    best = (m, tau, float(np.std(es)))
            res[method] = best
        ctx["data"] = data0                              # restore scale for next cbar
        ctx["truth"] = (ctx["truth"] / np.sqrt(alpha2)).astype(xp.complex64)
        ctx["Tnrm"] = float(xp.linalg.norm(xp.fft.fft2(ctx["truth"])))
        def f(b):
            return f"{b[0]:.4f}±{b[2]:.3f} t{b[1]}" if b[1] else f"{b[0]:.4f}±{b[2]:.3f}"
        print(f"{cbar:>7} | {f(res['hard']):>15} | {f(res['amplitude']):>15} "
              f"| {f(res['poisson']):>15} | {f(res['vst']):>15}")
    print("\nQ1: does KL beat sqrt-I (AGM) at ~0.01 ph/px, and does the Anscombe VST close "
          "the gap? (hard projection = today's sharpy = the tau->0 limit of all three).")
