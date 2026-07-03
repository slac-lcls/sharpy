"""Q1 decisive form: FIXED object dose split over R independent noisy passes.

Total information held constant (D_obj photons/object-pixel); per-measurement
counts c ~ D_obj/(coverage*R) drop as R grows. Poisson SUFFICIENCY: the KL
likelihood of R i.i.d. draws depends only on the SUMMED counts, so KL should be
~flat in R; the sqrt-bias (E[sqrt(c)] != sqrt(E[c])) should degrade AGM/hard as
c -> 0. Identical positions => consensus re-averages copies, so the exact loop is
    frames <- mean_r prox(F frames; c_r)  ->  overlap consensus.
Sync OFF, mild phase (metric isolation).

GPU: python -u proxd_repeat_test.py   env: NX(16) K(46) STEPD(8) DOBJ(20)
     RLIST("1 4 16 64") MAXIT(200) REPS(2)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NX", "16")
import numpy as np
import sync_bandpass_test as T
from Operators import ProxD_noise, GPU, xp

MAXIT = int(os.environ.get("MAXIT", 200))
REPS = int(os.environ.get("REPS", 2))
DOBJ = float(os.environ.get("DOBJ", 20.0))
RLIST = [int(r) for r in os.environ.get("RLIST", "1 4 16 64").split()]
TAUS = [0.5, 2.0]
rng = np.random.default_rng(41)


def run(ctx, counts, method, tau):
    """counts: (R, nf, nx, nx) photon counts; frames <- mean_r prox(F frames; c_r)."""
    probe, cprobe, mapid = ctx["probe"], ctx["cprobe"], ctx["mapid"]
    img = xp.ones((ctx["Ny"], ctx["Nx"]), dtype=xp.complex64)
    frames = T.Illuminate_frames(T.Splitc(img, mapid), probe)
    R = counts.shape[0]
    for it in range(MAXIT):
        Z = xp.fft.fft2(frames)
        acc = xp.zeros_like(Z)
        for r in range(R):
            if method == "hard":
                acc += ProxD_noise(Z, counts[r], tau=None)
            else:
                acc += ProxD_noise(Z, counts[r], tau=tau, metric=method)
        frames = xp.fft.ifft2(acc / R).astype(xp.complex64)
        img = T.Overlapc(T.Illuminate_frames(frames, cprobe), ctx["Nx"], ctx["Ny"],
                         mapid) / ctx["normalization"]
        frames = T.Illuminate_frames(T.Splitc(img, mapid), probe)
    s = xp.vdot(img, ctx["truth"]) / (xp.vdot(img, img) + 1e-30)
    return float(xp.linalg.norm(s * img - ctx["truth"]) / xp.linalg.norm(ctx["truth"]))


if __name__ == "__main__":
    K = int(os.environ.get("K", 46))
    orig = T.phantom
    T.phantom = lambda Nx, Ny, seed=0: np.abs(orig(Nx, Ny, seed)) * np.exp(
        1j * 0.4 * np.angle(orig(Nx, Ny, seed))).astype(np.complex64)
    ctx = T.build(K)
    T.phantom = orig
    data0 = ctx["data"] + 0
    cov = ctx["nframes"] * T.nx * T.nx / (ctx["Nx"] * ctx["Ny"])   # mean coverage/pixel
    cbar1 = DOBJ / cov                                             # counts/px at R=1
    alpha2 = cbar1 / float(data0.mean())
    ctx["data"] = data0 * alpha2
    ctx["truth"] = (ctx["truth"] * np.sqrt(alpha2)).astype(xp.complex64)
    clean = ctx["data"].get() if GPU else np.asarray(ctx["data"])
    print(f"K={K}: {ctx['nframes']} frames x {T.nx}, img {ctx['Nx']}^2, coverage {cov:.0f}x, "
          f"D_obj={DOBJ} ph/obj-px fixed; MAXIT={MAXIT} REPS={REPS} (sync OFF)")
    print(f"{'R':>4} {'c/px':>8} | {'hard(AP)':>15} | {'AGM tau*':>16} | {'KL tau*':>16}")
    for R in RLIST:
        res = {}
        for method in ("hard", "amplitude", "poisson"):
            taus = [None] if method == "hard" else TAUS
            best = (np.inf, None, 0.0)
            for tau in taus:
                es = []
                r2 = np.random.default_rng(7)
                for _ in range(REPS):
                    cts = r2.poisson(np.broadcast_to(clean / R, (R,) + clean.shape)
                                     ).astype(np.float32)
                    es.append(run(ctx, xp.asarray(cts), method, tau))
                m = float(np.mean(es))
                if m < best[0]:
                    best = (m, tau, float(np.std(es)))
            res[method] = best
        def f(b):
            return f"{b[0]:.4f}±{b[2]:.3f} t{b[1]}" if b[1] else f"{b[0]:.4f}±{b[2]:.3f}"
        print(f"{R:>4} {cbar1/R:>8.4f} | {f(res['hard']):>15} | {f(res['amplitude']):>16} "
              f"| {f(res['poisson']):>16}")
    print("\nEXPECT if the metric matters: KL ~flat in R (Poisson sufficiency: likelihood "
          "sees only summed counts); hard/AGM degrade as c/px -> 0 (sqrt bias).")
