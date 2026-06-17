"""
Faithful Fig. 7 reproduction (position retrieval), arXiv:1209.4924.

In-code analogue of the MATLAB driver horse_shoe_x1.m: recover unknown
per-frame scan-position errors jointly with the image, from intensity-only
data, and plot the position-error metric

    eps_xi = sum |xi - xi_truth|^2 / sum |xi_truth|^2

vs iteration.

Geometry matches horse_shoe_x1.m:
  * gold-balls test image (sharpy/gold.mat, var img0), cropped to Nx x Ny
  * zone-plate probe (make_probe radii matched to the MATLAB ones)
  * nx = ny = 32, step Dx = Dy = 3.5, 16 x 16 = 256 frames, hexagonal packing
  * Nx = Ny = 144
  * random sub-pixel position errors recovered from an integer-grid start

Because sharpy's frame map is integer-valued, the *entire* sub-pixel offset
from the integer grid (the 3.5-step fractional part + random jitter +
the unknown perturbation) is carried by the probe Taylor model and
retrieved from a zero estimate -- a slightly harder, cleaner demonstration
than the MATLAB (which assumes the 3.5-grid fractional part is known).

Run:  python position_fig7.py            # prints curve, saves PNG if mpl
"""

import os
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

if config.GPU:
    import cupy as cp

    xp = cp

    def tonp(a):
        return cp.asnumpy(a)
else:
    xp = np

    def tonp(a):
        return np.asarray(a)

from Operators import make_probe, map_frames, Splitc
from position_retrieval import probe_derivatives, taylor_shift_probe
import Solvers


# ---- geometry (horse_shoe_x1.m) --------------------------------------------
NX_FRAME = 32          # nx
DStep = 3.5            # Dx = Dy
NSCAN = 16             # nnx = nny
UNKNOWN_AMP = 2.0      # xix = (rand-.5)*4  -> uniform [-2, 2]
JITTER_FRAC = 0.25     # ixr = ix + (rand-.5)*Dx/4   (Dx/4 / Dx = 1/4)


def build_problem(seed=0):
    rng = np.random.default_rng(seed)
    nx = ny = NX_FRAME
    Dx = Dy = DStep
    nnx = nny = NSCAN

    # Periodic image: the step-Dx scan tiles it with wrap-around (map_frames is
    # periodic), so every pixel is covered.  (sharpy's Overlapc assumes full
    # coverage; the MATLAB's padded 144x144 leaves uncovered corners.)
    Nx = int(round(nnx * Dx))   # 56
    Ny = Nx

    # gold-balls test image, center-cropped to (Nx, Ny), normalized
    img0 = loadmat(os.path.join(os.path.dirname(__file__), "gold.mat"))["img0"]
    cy, cx = img0.shape[0] // 2, img0.shape[1] // 2
    truth = img0[cy - Nx // 2 : cy + Nx // 2, cx - Ny // 2 : cx + Ny // 2]
    truth = (truth / np.abs(truth).max()).astype(np.complex64)
    truth = xp.asarray(truth)

    # zone-plate probe (radii matched to MATLAB .025*nx*3, .085*nx*3)
    probe = make_probe(nx, ny, r1=0.075, r2=0.255)
    if isinstance(probe, tuple):
        probe = probe[0]
    probe = xp.asarray(probe / xp.abs(probe).max(), dtype=xp.complex64)

    # hexagonal scan positions (close packing: shear x by floor(Dx/2) on odd rows)
    # periodic, tiling the image (wrap handled by map_frames)
    ix1 = np.arange(nnx) * Dx
    iy1 = np.arange(nny) * Dy
    ix, iy = np.meshgrid(ix1, iy1, indexing="ij")
    xshift = np.floor(Dx / 2) * (np.arange(1, len(ix1) + 1) % 2)
    ix = ix + xshift[:, None]

    # known random jitter (+/- Dx/4)
    ix = ix + (rng.random(ix.shape) - 0.5) * Dx * JITTER_FRAC
    iy = iy + (rng.random(iy.shape) - 0.5) * Dy * JITTER_FRAC

    ix = ix.ravel()
    iy = iy.ravel()
    nframes = ix.size

    # unknown perturbation to retrieve
    xix = (rng.random(nframes) - 0.5) * 2 * UNKNOWN_AMP
    xiy = (rng.random(nframes) - 0.5) * 2 * UNKNOWN_AMP

    # split into integer base (-> map) and total sub-pixel offset (-> probe)
    base_x = np.floor(ix + xix)
    base_y = np.floor(iy + xiy)
    sub_x = (ix + xix) - base_x
    sub_y = (iy + xiy) - base_y

    translations_x = xp.asarray(base_x.astype(np.float64))
    translations_y = xp.asarray(base_y.astype(np.float64))
    xi_x_truth = xp.asarray(sub_x)
    xi_y_truth = xp.asarray(sub_y)

    # generate intensity data with the true sub-pixel offsets in the probe
    dp = probe_derivatives(probe)
    probe_shifted = taylor_shift_probe(dp, xi_x_truth, xi_y_truth)["O"]
    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)
    frames = Splitc(truth, mapid) * probe_shifted
    frames_data = xp.abs(xp.fft.fft2(frames)) ** 2

    return dict(
        frames_data=frames_data, probe=probe, truth=truth,
        translations_x=translations_x, translations_y=translations_y,
        nx=nx, ny=ny, Nx=Nx, Ny=Ny,
        xi_x_truth=xi_x_truth, xi_y_truth=xi_y_truth,
    )


def run(seed=0, maxiter=1000, position_start=100, method="diag"):
    p = build_problem(seed=seed)
    rng = np.random.default_rng(seed + 999)
    img0 = xp.asarray(
        (rng.standard_normal((p["Nx"], p["Ny"]))
         + 1j * rng.standard_normal((p["Nx"], p["Ny"]))).astype(np.complex64)
    )

    img, frames, xhat_x, xhat_y, res = Solvers.Alternating_projections_position(
        img0, p["probe"], p["frames_data"],
        p["translations_x"], p["translations_y"],
        p["nx"], p["ny"], p["Nx"], p["Ny"],
        maxiter=maxiter, position_start=position_start, position_every=1,
        method=method,
        img_truth=p["truth"],
        xi_x_truth=p["xi_x_truth"], xi_y_truth=p["xi_y_truth"],
        residuals_interval=1,
    )

    eps_xi = tonp(res[:, 3])
    img_mse = tonp(res[:, 0])
    print(f"frames: {p['frames_data'].shape[0]}, image: {p['Nx']}x{p['Ny']}")
    print(f"eps_xi  start(@{position_start})={eps_xi[position_start]:.3e}  "
          f"end={eps_xi[-1]:.3e}")
    print(f"img MSE start={img_mse[position_start]:.3e}  end={img_mse[-1]:.3e}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        it = np.arange(len(eps_xi))
        plt.figure(figsize=(7, 5))
        plt.semilogy(it, img_mse, label=r"$\varepsilon_0$ (image)", lw=2)
        plt.semilogy(it, eps_xi, label=r"$\varepsilon_\xi$ (position)", lw=2)
        plt.xlabel("iteration"); plt.ylabel("normalized error")
        plt.legend(); plt.title("Fig. 7 reproduction: position retrieval")
        plt.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "position_fig7.png")
        plt.savefig(out, dpi=120)
        print("saved", out)
    except Exception as e:
        print("(plot skipped:", e, ")")

    return eps_xi, img_mse


if __name__ == "__main__":
    run()
