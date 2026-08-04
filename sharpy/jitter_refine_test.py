"""Jitter + position refinement: does redundancy MAKE the refinement well-posed?

The board claim (od2p-domain-decomposition): "redundancy compensates NOISE (variance)
not JITTER (bias/blur); jitter needs position refinement, which redundancy makes
well-posed." This wires the per-frame position solver (position_solve_diag, the
diagonal 2x2 Gauss-Newton on the sub-pixel shift) into the plain AP loop and tests
all three legs:

  1. AP ignoring jitter  -> NMSE FLOORS at a jitter-set bias, flat in dose/redundancy
     (blur is systematic, averaging can't remove it).
  2. AP + position refinement -> recovers the shifts, NMSE drops toward the noise floor.
  3. The refinement quality vs REDUNDANCY x DOSE -> refinement needs enough
     overlap constraints per position; redundancy is what makes it well-posed at low
     dose (few photons/frame), where a single frame can't localize its own shift.

Nominal positions = integer grid (mapid); TRUE positions = grid + sub-pixel i.i.d.
jitter. Data = |F( split(truth) * shift_probe(probe, xi_true) )|^2 (+ Poisson).
Reconstruction assumes the grid and refines the residual shift xi_hat from zero.

CPU: /opt/anaconda3/bin/python3 jitter_refine_test.py
env: NX(24) STEPD(4) MAXIT(150) JIT(0.4 px rms) REFEVERY(3) PHR(1.0)
     KLIST("14 20 28") PHLIST("30 1e9")   (1e9 = noise-free)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config
from Operators import (map_frames, Splitc, Overlapc, Illuminate_frames, Project_data, make_probe, xp)
from position_retrieval import (probe_derivatives, shift_probe_fourier, position_solve_diag,
                                 position_solve_coupled, position_plan, shift_rmse)

nx = ny = int(os.environ.get("NX", 24))
STEPD = int(os.environ.get("STEPD", 4))
JIT = float(os.environ.get("JIT", 0.4))
PHR = float(os.environ.get("PHR", 1.0))
REFEVERY = int(os.environ.get("REFEVERY", 1))
rng = np.random.default_rng(7)


def phantom(Nx, Ny, seed=0):
    r = np.random.default_rng(seed)
    k = np.fft.fftfreq(Nx); KX, KY = np.meshgrid(k, k)
    amp = np.real(np.fft.ifft2(np.fft.fft2(r.standard_normal((Nx, Ny))) *
                               np.exp(-(KX**2 + KY**2) / (2 * 0.08**2))))
    ph = np.real(np.fft.ifft2(np.fft.fft2(r.standard_normal((Nx, Ny))) *
                              np.exp(-(KX**2 + KY**2) / (2 * (2.0/Nx)**2))))
    amp = (amp - amp.min()) / (amp.max() - amp.min()); amp = 0.5 + 0.5*amp
    ph = ph / np.abs(ph).max() * PHR
    return (amp * np.exp(1j*ph)).astype(np.complex64)


def build(K):
    step = max(1, nx // STEPD)
    Nx = Ny = (K-1)*step + nx
    g = xp.arange(K)*step
    tx, ty = xp.meshgrid(g, g, indexing="ij")
    tx = tx.ravel().astype(np.float64); ty = ty.ravel().astype(np.float64)
    nframes = tx.size
    truth = xp.asarray(phantom(Nx, Ny)).astype(xp.complex64)
    # Position sensitivity ~ probe structure/gradient. The smooth default probe is
    # nearly position-INSENSITIVE (a 1px shift barely changes the exit wave -> jitter
    # harmless AND refinement toothless). A filled-disk aperture (r2 controls the
    # probe speckle scale) makes jitter bite -> a meaningful refinement test.
    R2 = float(os.environ.get("PROBE_R2", 0.15))
    FXY = float(os.environ.get("PROBE_FXY", 0.0))
    probe = make_probe(nx, ny, r1=0.0, r2=R2, fx=FXY, fy=-FXY)[0].astype(xp.complex64)
    probe = (probe/xp.abs(probe).max()).astype(xp.complex64)
    mapid = map_frames(tx, ty, nx, ny, Nx, Ny)
    pplan = position_plan(tx, ty, nframes, nx, ny, Nx, Ny)
    # TRUE sub-pixel jitter, i.i.d. per frame
    jx = xp.asarray(rng.standard_normal(nframes))*JIT
    jy = xp.asarray(rng.standard_normal(nframes))*JIT
    jx -= jx.mean(); jy -= jy.mean()
    # honest data: probe shifted by the true jitter (all Taylor orders)
    pshift = shift_probe_fourier(probe, jx, jy)                 # (nframes,nx,ny)
    exit_w = Splitc(truth, mapid) * pshift
    clean = (xp.abs(xp.fft.fft2(exit_w))**2).astype(xp.float32)
    absP2 = xp.broadcast_to(xp.abs(probe)**2, (nframes, nx, ny)).astype(xp.complex64)
    return dict(K=K, step=step, Nx=Nx, Ny=Ny, nframes=nframes, truth=truth, probe=probe,
                cprobe=xp.conj(probe), mapid=mapid, clean=clean, absP2=absP2, jx=jx, jy=jy,
                pplan=pplan)


def img_nmse(ctx, img):
    s = xp.vdot(img, ctx["truth"]) / (xp.vdot(img, img) + 1e-30)
    return float(xp.linalg.norm(s*img - ctx["truth"]) / xp.linalg.norm(ctx["truth"]))


DAMP = float(os.environ.get("DAMP", 0.3))          # step damping on the position update


def run(ctx, data, refine, maxit):
    """AP with per-frame shifted probe; refine in {'none','diag','coupled'}.

    Position steps are DAMPED (xi += DAMP*dxi): a single-frame Gauss-Newton on a
    noisy exit wave is high-variance, so damping is the poor-man's regularizer
    (the coupled solver adds the real cross-frame regularization via overlap)."""
    probe, cprobe, mapid = ctx["probe"], ctx["cprobe"], ctx["mapid"]
    Nx, Ny, nf = ctx["Nx"], ctx["Ny"], ctx["nframes"]
    dp = probe_derivatives(probe)
    xi_x = xp.zeros(nf); xi_y = xp.zeros(nf)
    img = xp.ones((Ny, Nx), dtype=xp.complex64)
    for it in range(maxit):
        pO = shift_probe_fourier(probe, xi_x, xi_y)            # current shifted probe stack
        frames = Splitc(img, mapid) * pO                       # per-frame illumination
        frames, _ = Project_data(frames, data)
        if refine != "none" and it % REFEVERY == 0 and it > 0:
            if refine == "coupled":
                nx_, ny_ = position_solve_coupled(frames, dp, img, mapid, Nx, Ny,
                                                  xi_x, xi_y, ctx["pplan"])
            else:
                nx_, ny_ = position_solve_diag(frames, dp, img, mapid, Nx, Ny,
                                               xi_x, xi_y, method="exact")
            xi_x = xi_x + DAMP * (nx_ - xi_x)                  # damped update
            xi_y = xi_y + DAMP * (ny_ - xi_y)
            pO = shift_probe_fourier(probe, xi_x, xi_y)
        cpO = xp.conj(pO)
        norm = Overlapc(xp.abs(pO)**2, Nx, Ny, mapid)
        norm = xp.where(xp.abs(norm) < 1e-6*float(xp.max(xp.abs(norm))), xp.complex64(1), norm)
        img = Overlapc(frames * cpO, Nx, Ny, mapid) / norm
    rmse = float(shift_rmse(ctx["jx"], ctx["jy"], xi_x, xi_y))
    return img_nmse(ctx, img), rmse


if __name__ == "__main__":
    MAXIT = int(os.environ.get("MAXIT", 200))
    KLIST = [int(k) for k in os.environ.get("KLIST", "14 20 28").split()]
    PHLIST = [float(p) for p in os.environ.get("PHLIST", "30 300 1e9").split()]
    print(f"nx={nx} jitter rms {JIT}px, phase +-{PHR} rad, refine every {REFEVERY}, "
          f"damp {DAMP}, MAXIT={MAXIT}  (start shift RMSE = sqrt2*JIT ~ {np.sqrt(2)*JIT:.2f})")
    print(f"{'K':>3} {'frames':>6} {'ph/fr':>6} | {'no-ref':>8} {'ideal':>8} | "
          f"{'diag N':>8} {'diagRMSE':>8} | {'coup N':>8} {'coupRMSE':>8}")
    for K in KLIST:
        ctx = build(K)
        jx0, jy0 = ctx["jx"] + 0, ctx["jy"] + 0
        for PH in PHLIST:
            def dose(cl):
                if PH >= 1e9:
                    return cl
                s = PH / (float(cl.sum()) / ctx["nframes"])
                dn = cl.get() if config.GPU else np.asarray(cl)
                return xp.asarray(rng.poisson(dn*s).astype(np.float32)/s)
            data = dose(ctx["clean"])
            n_noref, _ = run(ctx, data, "none", MAXIT)
            n_diag, r_diag = run(ctx, data, "diag", MAXIT)
            n_coup, r_coup = run(ctx, data, "coupled", MAXIT)
            # ideal (no jitter) reference at the same dose
            exitw = Splitc(ctx["truth"], ctx["mapid"]) * ctx["probe"][None]
            cl2 = dose((xp.abs(xp.fft.fft2(exitw))**2).astype(xp.float32))
            ctx["jx"] = xp.zeros(ctx["nframes"]); ctx["jy"] = xp.zeros(ctx["nframes"])
            n_ideal, _ = run(ctx, cl2, "none", MAXIT); ctx["jx"], ctx["jy"] = jx0+0, jy0+0
            phs = "inf" if PH >= 1e9 else f"{PH:.0f}"
            print(f"{K:>3} {ctx['nframes']:>6} {phs:>6} | {n_noref:>8.4f} {n_ideal:>8.4f} | "
                  f"{n_diag:>8.4f} {r_diag:>8.3f} | {n_coup:>8.4f} {r_coup:>8.3f}")
    print("\nREAD: no-ref vs ideal = jitter's image cost (BIAS). diag = per-frame solver "
          "(no cross-frame info -> diverges at low dose). coupled = overlap-coupled solver "
          "(redundancy makes the shift well-posed) -> should track ideal where diag blows up.")
