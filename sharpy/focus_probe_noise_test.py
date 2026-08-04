"""In-focus vs out-of-focus (small vs EXPANDED) smooth probe: does illuminating a larger
area LOWER the photon threshold for long-range phase recovery?

Motivation (SM): people expand/defocus the probe to illuminate a larger area and imprint
more detector pixels per shot; the FEL is attenuated so the sample survives, but the
attenuation is ADJUSTABLE -> total flux/shot is a free knob. Hypothesis: a broad smooth
illumination has a NARROW, near-forward-CONCENTRATED diffraction -> MORE low-q energy (the
OPPOSITE of a BLR/speckle probe, which spreads it out) AND a bigger overlap at fixed scan.
The long-range sync rides on those low-q near-forward photons + the overlap graph, so an
expanded smooth probe should LOWER the long-range threshold -- the "good" version of what
BLR tried (BLR RAISED it, 7c50050).

Two coupled effects as the spot grows (both HELP, we report both so the mechanism is
visible, not hidden): (a) diffraction q_rms shrinks -> photons concentrate at low q;
(b) spot/step overlap grows (the known ~10x lever). Fair dosing: each probe is power-
normalized (identical flux/shot) and Poisson dose is per-detector-pixel (PH) so the
expected TOTAL counts/shot are identical across sizes -- only the DISTRIBUTION differs
(= "attenuation adjusted to equal measured signal").

Primary sweep = real-space spot size (SIGLIST, Gaussian sigma / nx). Optional DEFOCUS=1
does a unitary Fresnel-defocus sweep (DZLIST) of the default probe instead (flux-exact,
same-pupil "out-of-focus" rather than a bigger optic) for contrast.

CPU / numpy authoritative.  /opt/anaconda3/bin/python3 focus_probe_noise_test.py
env: NX(16) K(24) STEPD(4) MAXIT(120) SE(1) REPS(4)
     SIGLIST("0.10 0.15 0.20 0.30 0.42")   PHLIST("3 1 0.3 0.1 0.03")
     DEFOCUS(0)  DZLIST("0 150 400 900")
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NX", "16")

import numpy as np
import sync_bandpass_test as T
from Operators import Precondition_calc, xp
from blr_probe_noise_test import set_probe

nx = T.nx
K = int(os.environ.get("K", 24))
MAXIT = int(os.environ.get("MAXIT", 120))
SE = int(os.environ.get("SE", 1))
REPS = int(os.environ.get("REPS", 4))
SIGLIST = [float(x) for x in os.environ.get("SIGLIST", "0.10 0.15 0.20 0.30 0.42").split()]
PHLIST = [float(p) for p in os.environ.get("PHLIST", "3 1 0.3 0.1 0.03").split()]
DEFOCUS = int(os.environ.get("DEFOCUS", 0))
DZLIST = [float(x) for x in os.environ.get("DZLIST", "0 150 400 900").split()]


def gauss_probe(sigma_frac, ref_power):
    """Smooth Gaussian probe of rms width sigma_frac*nx, power-normalized to ref_power
    (identical total flux/shot -> fair across sizes)."""
    c = nx / 2.0
    X, Y = np.meshgrid(np.arange(nx) - c, np.arange(nx) - c, indexing="ij")
    p = np.exp(-(X ** 2 + Y ** 2) / (2.0 * (sigma_frac * nx) ** 2))
    p = p / np.sqrt((np.abs(p) ** 2).sum()) * np.sqrt(ref_power)
    return xp.asarray(p.astype(np.complex64))


def defocus(probe, dz):
    """Unitary Fresnel-defocus (quadratic pupil phase) -> preserves total power exactly.
    dz=0 in-focus; larger dz spreads the real-space spot (same-pupil out-of-focus)."""
    p = np.asarray(probe)
    k = np.fft.fftfreq(p.shape[0])
    KX, KY = np.meshgrid(k, k, indexing="ij")
    H = np.exp(-1j * np.pi * dz * (KX ** 2 + KY ** 2))
    return xp.asarray(np.fft.ifft2(np.fft.fft2(p) * H).astype(np.complex64))


def spot_radius(probe):
    """rms real-space radius of |probe|^2 (illuminated-area size, px)."""
    p = np.abs(np.asarray(probe)) ** 2
    n = p.shape[0]; c = n / 2.0
    y, x = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing="ij")
    w = p / (p.sum() + 1e-30)
    return float(np.sqrt((w * (x ** 2 + y ** 2)).sum()))


def q_rms(clean):
    """rms radial spatial frequency of the (per-frame-summed) diffraction, cycles/px.
    Small = photons concentrated near forward (low q). This is the mechanism readout."""
    c = np.asarray(clean).sum(axis=0)                     # sum over frames -> (nx, nx)
    n = c.shape[-1]
    qy, qx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
    qr2 = qx ** 2 + qy ** 2
    return float(np.sqrt((c * qr2).sum() / (c.sum() + 1e-30)))


if __name__ == "__main__":
    T.STEPD = int(os.environ.get("STEPD", 4))
    T.NUMITER = 5
    ctx = T.build(K)
    base = np.asarray(ctx["probe"])
    ref_power = float((np.abs(base) ** 2).sum())
    rng = np.random.default_rng(1)
    step = ctx["step"]
    if DEFOCUS:
        variants = [("dz=" + format(dz, "g"), defocus(ctx["probe"], dz)) for dz in DZLIST]
        knob = "Fresnel defocus (flux-exact, same pupil)"
    else:
        variants = [("sig=" + format(s, "g"), gauss_probe(s, ref_power)) for s in SIGLIST]
        knob = "real-space spot size (power-matched)"
    print(f"K={K} ({ctx['nframes']} frames x {nx}), img {ctx['Nx']}^2, scan step={step}px, "
          f"MAXIT={MAXIT} SE={SE} REPS={REPS}  [{knob}]")
    hdr = " ".join(f"{p:>7g}" for p in PHLIST)
    print(f"{'probe':>10} {'r_spot':>6} {'ovlp%':>6} {'q_rms':>6} | {hdr}   (eigsh-sync lf-NMSE, ph/px)")
    for label, probe in variants:
        set_probe(ctx, probe)
        clean = np.asarray(ctx["clean"])
        rs = spot_radius(probe)
        ovlp = 100.0 * max(0.0, 1.0 - step / (2.0 * rs))     # spot-diameter vs step overlap proxy
        qr = q_rms(clean)
        row = []
        for PH in PHLIST:
            scale = PH / (float(clean.sum()) / clean.size)
            sy = []
            for r in range(REPS):
                noisy = xp.asarray(rng.poisson(clean * scale).astype(np.float32) / scale)
                ctx["data"] = noisy
                fn = Precondition_calc(noisy, bw=ctx["Gramiam"]["bw"])
                ctx["frames_norm"] = xp.where(xp.abs(fn) < 1e-6, xp.asarray(1e-6, fn.dtype), fn)
                sy.append(T.run(ctx, SE, MAXIT, solver=T.eigsh_sync)[-1, 0])
            row.append(np.mean(sy))
        vals = " ".join(f"{v:>7.3f}" for v in row)
        print(f"{label:>10} {rs:>6.2f} {ovlp:>6.1f} {qr:>6.3f} | {vals}")
    print("\nEXPECT: as the spot grows (r_spot up, q_rms down = photons pile at low q, overlap up), "
          "the eigsh-sync lf-NMSE should collapse at a LOWER dose -> an expanded smooth probe "
          "LOWERS the long-range threshold (opposite of BLR).")
