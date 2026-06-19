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
import Solvers


# ---- geometry (horse_shoe_x1.m) --------------------------------------------
NX_FRAME = 32          # nx
DStep = 3.5            # Dx = Dy
NSCAN = 16             # nnx = nny
UNKNOWN_AMP = 0.4      # sub-pixel error to retrieve, uniform [-0.4, 0.4] px
# NOTE: the Taylor/trust-region solver is SUB-PIXEL only (the integer frame map
# is fixed; the shift rides on the probe). Errors >~0.5 px need integer-map
# re-planning, which is not implemented. The paper's "±2px" Fig 7 effectively
# absorbed the integer part into the map and retrieved only the <1px remainder.
JITTER_FRAC = 0.25     # ixr = ix + (rand-.5)*Dx/4   (Dx/4 / Dx = 1/4)


def build_problem(seed=0):
    """Realistic experimental workflow: simulate a transmission object with
    baked-in sub-pixel position errors to an .h5, then load it back (with the
    resolution/translation round-trip), exactly as ptycho_simulate.py ->
    ptycho_reconstruct.py."""
    import tempfile
    from position_simulate import simulate_to_h5, load_h5

    rng = np.random.default_rng(seed)
    nframes = NSCAN * NSCAN
    # unknown per-frame sub-pixel position errors to retrieve (uniform +/- AMP)
    xi_x = (rng.random(nframes) - 0.5) * 2 * UNKNOWN_AMP
    xi_y = (rng.random(nframes) - 0.5) * 2 * UNKNOWN_AMP

    fn = os.path.join(tempfile.gettempdir(), f"position_fig7_{seed}.h5")
    # high contrast (|obj|min ~ exp(-4.1) ~ 0.016) to match the paper's Fig 7
    # gold object; the realistic weak-contrast regime (contrast~0.69) is much
    # harder -- see position_simulate / the realism finding.
    simulate_to_h5(fn, xi_x, xi_y, nx=NX_FRAME, nnx=NSCAN, step=DStep,
                   r1=0.075, r2=0.255, contrast=4.1)
    p = load_h5(fn)
    # to xp (CPU here); keep the loaded keys the solver expects
    for k in ("frames_data", "probe", "truth", "translations_x", "translations_y",
              "xi_x_truth", "xi_y_truth"):
        p[k] = xp.asarray(p[k])
    return p


def run(seed=0, maxiter=1000, position_start=100, method="diag"):
    p = build_problem(seed=seed)
    # X-ray contrast is small: the image is transmission, ~1 everywhere.
    # Start at ones (physically motivated; also avoids the zero fixed point).
    img0 = xp.ones((p["Nx"], p["Ny"]), dtype=xp.complex64)

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
