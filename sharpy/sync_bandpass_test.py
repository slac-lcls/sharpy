"""Band-pass monitor of reconstruction error (LOW-freq vs HIGH-freq) with vs without
the Gramian sync, to SEE where sync acts — and whether the win GROWS with N.

Premise (Stefano): low-freq AMPLITUDE is trivial (it comes straight from the data);
the hard/slow mode is the low-freq PHASE. The Gramian sync targets exactly that. So
split the reconstruction error into a LOW band and a HIGH band and watch each vs
iteration, AP-only vs AP+periodic-sync. Prediction: sync collapses the LOW band fast
(coarse-correction of the slow global phase) while the HIGH band converges at the same
AP rate either way — and the AP-only low band gets SLOWER as N grows (Fiedler ~1/N
diffusion) while the synced one stays ~flat (the framewise-strong-scaling claim).

Sync eigensolver: the in-loop power iteration (num_iter~5) under-converges in the
varying-low-freq-phase regime (L2-budget-limited) — the eigsolver study's winner there
is a FULL eigsh(LM) top-eigenvector, so this test patches Operators.Eigensolver with an
ARPACK eigsh drop-in (arm "eigsh"), keeping the power arm for reference.

PLAIN AP only — no OD2P / domain decomposition. Dense-overlap sim with a genuine
low-freq PHASE structure (else sync is irrelevant).
Convergence question -> CPU/numpy authoritative.  /opt/anaconda3/bin/python3 sync_bandpass_test.py
env: NX(64) KLIST("10 20 30" => nframes K^2) STEPD(4 => step=NX/STEPD) MAXIT(150)
     SYNCEVERY(5) NUMITER(5, power arm) QCUT(0.12)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config
import Operators
from Operators import (map_frames, Splitc, Overlapc, Illuminate_frames, Project_data,
                       Gramiam_plan, Precondition_calc, synchronize_frames_c, eig_reset, xp)
from scipy.sparse.linalg import eigsh as sp_eigsh

nx = ny = int(os.environ.get("NX", 64))
STEPD = int(os.environ.get("STEPD", 4))
QCUT = float(os.environ.get("QCUT", 0))   # low-band edge in cycles/px; 0 = auto 1/nx (inter-frame band)
PHS = float(os.environ.get("PHS", 2.0))   # phase-field width, cycles across the IMAGE

_POWER = Operators.Eigensolver          # the committed power-iteration solver
_eigsh_v0 = {"v": None}                 # warm start across sync calls within a run


def eigsh_sync(H, num_iter, v0=None, tol=1e-7):
    """Drop-in for Operators.Eigensolver: FULL top eigenvector via ARPACK eigsh(LM).

    The eigensolver-study winner for the varying-low-freq-phase regime (in-loop
    power at num_iter~5 is L2-budget-limited and under-converges exactly there).
    complex128 for ARPACK stability; returns unit-modulus, common-phase-removed
    omega shaped (nframes,1,1), matching Eigensolver's contract.
    """
    n = H.shape[0]
    w0 = _eigsh_v0["v"]
    if w0 is not None and w0.shape[0] != n:
        w0 = None
    lam, V = sp_eigsh(H.astype(np.complex128), k=1, which="LM", v0=w0, tol=1e-8)
    w = V[:, 0]
    _eigsh_v0["v"] = w.copy()
    w = w / (np.abs(w) + 1e-30)                      # unit modulus per frame
    s = np.conj(w.sum()); s /= (abs(s) + 1e-30)      # remove the common phase
    return (w * s).astype(np.complex64).reshape(n, 1, 1)


def phantom(Nx, Ny, seed=0):
    """Smooth amplitude (within-frame scale, AP's job) + GLOBAL-scale phase (sync's job).

    The phase field must vary on scales LARGE vs the frame (a per-frame constant
    phase can only represent q < 1/nx): filter width ~PHS cycles across the whole
    image, so the number of phase oscillations across the FOV is FIXED as the image
    grows with K — the global slow mode the Fiedler ~1/N diffusion has to cross.
    """
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(Nx); KX, KY = np.meshgrid(k, k)
    F = np.fft.fft2(rng.standard_normal((Nx, Ny)))
    amp = np.real(np.fft.ifft2(F * np.exp(-(KX**2 + KY**2) / (2 * 0.05**2))))   # smooth amplitude (~20px)
    sq = PHS / Nx                                                               # phase: ~PHS cycles/image
    F2 = np.fft.fft2(rng.standard_normal((Nx, Ny)))
    ph = np.real(np.fft.ifft2(F2 * np.exp(-(KX**2 + KY**2) / (2 * sq**2))))     # IMAGE-scale phase
    amp = (amp - amp.min()) / (amp.max() - amp.min()); amp = 0.5 + 0.5 * amp
    ph = ph / np.abs(ph).max() * 2.5                                            # ~±2.5 rad long-range phase
    return (amp * np.exp(1j * ph)).astype(np.complex64)


def build(K):
    """Dense-overlap sim context for a KxK grid (nframes = K^2)."""
    step = max(1, nx // STEPD)
    Nx = Ny = (K - 1) * step + nx
    g = xp.arange(K) * step
    tx, ty = xp.meshgrid(g, g, indexing="ij")
    tx = tx.ravel().astype(np.float64); ty = ty.ravel().astype(np.float64)
    nframes = tx.size

    truth = xp.asarray(phantom(Nx, Ny)).astype(xp.complex64)
    c = nx // 2
    X, Y = xp.meshgrid(xp.arange(nx) - c, xp.arange(nx) - c)
    probe = xp.exp(-(X**2 + Y**2) / (2.0 * (0.2 * nx)**2)).astype(xp.complex64)
    probe = (probe / xp.abs(probe).max()).astype(xp.complex64)
    mapid = map_frames(tx, ty, nx, ny, Nx, Ny)
    data = (xp.abs(xp.fft.fft2(Splitc(truth, mapid) * probe[None]))**2).astype(xp.float32)

    absP2 = xp.broadcast_to(xp.abs(probe)**2, (nframes, nx, ny)).astype(xp.complex64)
    normalization = Overlapc(absP2, Nx, Ny, mapid)
    normalization = xp.where(xp.abs(normalization) < 1e-6 * float(xp.max(xp.abs(normalization))),
                             xp.complex64(1), normalization)
    # sync plan (as Solvers wires it)
    Gramiam = Gramiam_plan(tx, ty, nframes, nx, ny, Nx, Ny)
    inormalization_split = Splitc(1.0 / normalization, mapid)
    frames_norm = Precondition_calc(data, bw=Gramiam["bw"])

    # radial frequency masks for the band split: LOW = inter-frame band (q < 1/nx),
    # the only band a per-frame constant phase (the sync) can act on.
    qy, qx = xp.meshgrid(xp.fft.fftfreq(Ny), xp.fft.fftfreq(Nx), indexing="ij")
    qr = xp.sqrt(qx**2 + qy**2)
    qlo = QCUT if QCUT > 0 else 1.0 / nx
    low_mask = qr < qlo
    Tnrm = float(xp.linalg.norm(xp.fft.fft2(truth)))
    return dict(K=K, step=step, Nx=Nx, Ny=Ny, nframes=nframes, truth=truth, probe=probe,
                cprobe=xp.conj(probe), mapid=mapid, data=data, normalization=normalization,
                Gramiam=Gramiam, inorm_split=inormalization_split, frames_norm=frames_norm,
                low_mask=low_mask, Tnrm=Tnrm)


def band_err(ctx, img):
    """relative LOW-band and HIGH-band reconstruction error (Parseval: low^2+high^2 = nmse^2)."""
    truth = ctx["truth"]
    s = xp.vdot(img, truth) / (xp.vdot(img, img) + 1e-30)      # global complex gauge
    D = xp.fft.fft2(s * img - truth)
    lo = float(xp.linalg.norm(D[ctx["low_mask"]])) / ctx["Tnrm"]
    hi = float(xp.linalg.norm(D[~ctx["low_mask"]])) / ctx["Tnrm"]
    return lo, hi


def run(ctx, sync_every, maxit, solver=None):
    """Plain AP (data projection + overlap), optional periodic Gramian sync."""
    Operators.Eigensolver = solver if solver is not None else _POWER
    eig_reset(); _eigsh_v0["v"] = None
    probe, cprobe, mapid = ctx["probe"], ctx["cprobe"], ctx["mapid"]
    img = xp.ones((ctx["Ny"], ctx["Nx"]), dtype=xp.complex64)
    frames = Illuminate_frames(Splitc(img, mapid), probe)
    curve = []
    for it in range(maxit):
        frames, _ = Project_data(frames, ctx["data"])          # data projection P_a
        if sync_every and it % sync_every == 0:
            omega = synchronize_frames_c(frames, probe, ctx["frames_norm"], ctx["inorm_split"],
                                         ctx["Gramiam"], ctx["Gramiam"]["bw"], NUMITER)
            frames = frames * omega
        img = Overlapc(Illuminate_frames(frames, cprobe), ctx["Nx"], ctx["Ny"], mapid) / ctx["normalization"]
        frames = Illuminate_frames(Splitc(img, mapid), probe)
        curve.append(band_err(ctx, img))
    Operators.Eigensolver = _POWER
    return np.array(curve)                                     # (maxit, 2) [low, high]


def iters_to(col, thr):
    w = np.where(col < thr)[0]
    return int(w[0]) + 1 if w.size else None


def fmt(n):
    return f"{n:>5d}" if n is not None else f"{'-':>5}"


if __name__ == "__main__":
    MAXIT = int(os.environ.get("MAXIT", 300))
    SE = int(os.environ.get("SYNCEVERY", 1))
    NUMITER = int(os.environ.get("NUMITER", 5))
    LOWTHR = float(os.environ.get("LOWTHR", 0.1))
    KLIST = [int(k) for k in os.environ.get("KLIST", "10 20 30").split()]
    summary = []
    for K in KLIST:
        ctx = build(K)
        print(f"\n===== K={K}: img {ctx['Nx']}x{ctx['Ny']}, {ctx['nframes']} frames x {nx}, "
              f"step {ctx['step']} (overlap {100*(1-ctx['step']/nx):.0f}%), low-band q<1/nx =====")
        ap = run(ctx, 0, MAXIT)
        po = run(ctx, SE, MAXIT, solver=None)          # power arm (committed in-loop solver)
        ei = run(ctx, SE, MAXIT, solver=eigsh_sync)    # eigsh arm (study winner)
        print(f"{'iter':>5} | {'AP low':>8} {'AP high':>8} | {'pow low':>8} {'pow high':>8} | {'eig low':>8} {'eig high':>8}")
        for it in [0, 4, 9, 19, 39, 79, MAXIT - 1]:
            if it < MAXIT:
                print(f"{it+1:>5} | {ap[it,0]:>8.4f} {ap[it,1]:>8.4f} | {po[it,0]:>8.4f} {po[it,1]:>8.4f} "
                      f"| {ei[it,0]:>8.4f} {ei[it,1]:>8.4f}")
        row = (K, ctx["nframes"],
               iters_to(ap[:, 0], LOWTHR), iters_to(po[:, 0], LOWTHR), iters_to(ei[:, 0], LOWTHR),
               ap[-1, 0], po[-1, 0], ei[-1, 0], ap[-1, 1], ei[-1, 1])
        summary.append(row)

    print(f"\n===== SCALING: iters to LOW-band < {LOWTHR} (and final low err) =====")
    print(f"{'K':>3} {'frames':>7} | {'AP':>5} {'power':>5} {'eigsh':>5} | "
          f"{'AP fin':>8} {'pow fin':>8} {'eig fin':>8} | {'AP hi':>7} {'eig hi':>7}")
    for K, nf, a, p, e, af, pf, ef, ah, eh in summary:
        print(f"{K:>3} {nf:>7} | {fmt(a)} {fmt(p)} {fmt(e)} | {af:>8.4f} {pf:>8.4f} {ef:>8.4f} "
              f"| {ah:>7.4f} {eh:>7.4f}")
    print("\nEXPECT: AP-only low-band iters GROW with N (Fiedler ~1/N diffusion); eigsh-sync stays "
          "~flat (coarse-correction); HIGH band ~same all arms (sync is low-freq only).")
