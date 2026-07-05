"""Does a band-limited-random (BLR / speckle / diffuser) probe lower the photon
threshold for long-range phase recovery vs a smooth probe? (arXiv:1402.0550 idea.)

A speckly probe decorrelates neighbouring frame overlaps -> the connection-graph edges
become higher-rank / better conditioned (bigger Fiedler gap) than a smooth probe's
near-DC-degenerate overlaps. Prediction: sync recovers the long-range phase at LOWER
dose, and the eigsh cluster-ambiguity shrinks.

Fair comparison: BOTH probes share the same smooth amplitude ENVELOPE (same spot /
overlap area) and the same total flux; only the phase differs -- flat (smooth) vs a
band-limited random phase screen of correlation length L (BLR). Sweeps Poisson dose;
metric = low-freq-band NMSE of the recovered phase, no-sync vs eigsh-sync.

CPU / numpy authoritative.  /opt/anaconda3/bin/python3 blr_probe_noise_test.py
env: NX(16) K(24) STEPD(4) MAXIT(120) SE(1) REPS(4) LCORR(3 speckle corr length, px)
     PHLIST("3 1 0.3 0.1 0.03")
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NX", "16")

import numpy as np
import sync_bandpass_test as T
from Operators import (Illuminate_frames, Splitc, Overlapc, Precondition_calc, xp)

nx = T.nx
K = int(os.environ.get("K", 24))
MAXIT = int(os.environ.get("MAXIT", 120))
SE = int(os.environ.get("SE", 1))
REPS = int(os.environ.get("REPS", 4))
LCORR = float(os.environ.get("LCORR", 3.0))
PHLIST = [float(p) for p in os.environ.get("PHLIST", "3 1 0.3 0.1 0.03").split()]


def envelope(nx, rfrac=0.32, taper=0.4):
    c = nx / 2.0
    y, x = np.meshgrid(np.arange(nx) - c, np.arange(nx) - c, indexing="ij")
    r = np.sqrt(x ** 2 + y ** 2)
    r_out = rfrac * nx; r_in = r_out * (1 - taper)
    t = np.clip((r_out - r) / (r_out - r_in), 0, 1)
    return 0.5 * (1 - np.cos(np.pi * t))


def blr_phase(nx, L, seed):
    """Band-limited random phase screen: white noise low-passed to correlation length L."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(nx)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    F = np.fft.fft2(rng.standard_normal((nx, nx)))
    ph = np.real(np.fft.ifft2(F * np.exp(-(KX ** 2 + KY ** 2) / (2 * (1.0 / L) ** 2))))
    return ph / (np.std(ph) + 1e-30) * np.pi        # ~unit-radian rms x pi


def blr_from(probe, L, seed=0):
    """BLR = same |probe| (known-good smooth spot) x a band-limited random phase screen.
    Isolates phase structure: identical amplitude/overlap/flux, only the phase differs."""
    p = np.asarray(probe)
    return xp.asarray((p * np.exp(1j * blr_phase(p.shape[0], L, seed))).astype(np.complex64))


def set_probe(ctx, probe):
    """Install a probe: recompute the probe-dependent data, normalization, plan norms."""
    ctx["probe"] = probe.astype(xp.complex64)
    ctx["cprobe"] = xp.conj(ctx["probe"])
    mapid = ctx["mapid"]
    ex = Splitc(ctx["truth"], mapid) * ctx["probe"][None]
    ctx["clean"] = (xp.abs(xp.fft.fft2(ex)) ** 2).astype(xp.float32)
    absP2 = xp.broadcast_to(xp.abs(ctx["probe"]) ** 2, (ctx["nframes"], nx, nx)).astype(xp.complex64)
    nrm = Overlapc(absP2, ctx["Nx"], ctx["Ny"], mapid)
    nrm = xp.where(xp.abs(nrm) < 1e-6 * float(xp.max(xp.abs(nrm))), xp.complex64(1), nrm)
    ctx["normalization"] = nrm
    ctx["inorm_split"] = Splitc(1.0 / nrm, mapid)
    return ctx


if __name__ == "__main__":
    T.STEPD = int(os.environ.get("STEPD", 4))
    T.NUMITER = 5
    ctx = T.build(K)
    smooth = ctx["probe"] + 0                          # the known-good band-pass default
    blr = blr_from(smooth, LCORR)                      # same |probe|, speckle phase
    rng = np.random.default_rng(1)
    ov = 100 * (1 - ctx["step"] / nx)
    print(f"K={K} ({ctx['nframes']} frames x {nx}), img {ctx['Nx']}^2, overlap {ov:.0f}%, "
          f"BLR corr length {LCORR}px, MAXIT={MAXIT} SE={SE} REPS={REPS}")
    print(f"{'probe':>7} {'ph/px':>7} | {'no-sync':>8} {'eigsh-sync':>11}")
    for label, probe in (("smooth", smooth), ("BLR", blr)):
        set_probe(ctx, probe)
        clean = np.asarray(ctx["clean"])
        for PH in PHLIST:
            scale = PH / (float(clean.sum()) / clean.size)
            ns, sy = [], []
            for r in range(REPS):
                noisy = xp.asarray(rng.poisson(clean * scale).astype(np.float32) / scale)
                ctx["data"] = noisy
                fn = Precondition_calc(noisy, bw=ctx["Gramiam"]["bw"])
                ctx["frames_norm"] = xp.where(xp.abs(fn) < 1e-6, xp.asarray(1e-6, fn.dtype), fn)
                ns.append(T.run(ctx, 0, MAXIT)[-1, 0])
                sy.append(T.run(ctx, SE, MAXIT, solver=T.eigsh_sync)[-1, 0])
            print(f"{label:>7} {PH:>7.2f} | {np.mean(ns):>8.3f} {np.mean(sy):>11.3f}")
    print("\nEXPECT: if the speckle probe conditions the overlap graph, BLR eigsh-sync recovers "
          "the low band (lower NMSE) at doses where the smooth probe has already collapsed.")
