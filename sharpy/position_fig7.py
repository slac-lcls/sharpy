"""
Fig. 7 reproduction (position retrieval), arXiv:1209.4924 Section IV.

Setup matches the MATLAB horse_shoe_x1.m:
  * 16x16 = 256 frames, 32x32 px frame, step=3.5 px, hexagonal packing
  * R1=0.075, R2=0.255  (horse_shoe_x1.m: r1=.025*nx*3, r2=.085*nx*3)
  * Unknown xi perturbations: uniform in [-A, A] per component
  * Outer loop: RAAR beta=0.75 (doraar0_shift.m)

Results are cached to position_fig7_results.npz so re-running just re-plots.

Run:  python position_fig7.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

if config.GPU:
    import cupy as cp
    xp = cp
    def tonp(a): return cp.asnumpy(a)
else:
    xp = np
    def tonp(a): return np.asarray(a)

import Solvers

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---- geometry (horse_shoe_x1.m) --------------------------------------------
NX_FRAME  = 32       # nx = ny
DStep     = 3.5      # step size (pixels)
NSCAN     = 16       # nnx = nny  (16x16 = 256 frames)
R1        = 0.075    # zone-plate inner radius (horse_shoe_x1.m: r1=.025*nx*3=0.075*nx)
R2        = 0.255    # zone-plate outer radius (horse_shoe_x1.m: r2=.085*nx*3=0.255*nx)
CONTRAST  = 4.1      # Beer-Lambert exponent (high-contrast gold balls)

# Paper Fig 7: <|xi_0|> = 2.5 resolution elements, res_el = 1/(r2*nx) = 0.122 px
# => mean 2D |xi| = 2.5 * 0.122 = 0.306 px
# For uniform xi_x, xi_y in [-A, A]: E[|xi|] = 0.765*A => A = 0.306/0.765 = 0.40 px
UNKNOWN_AMP = 2.5 / (0.765 * R2 * NX_FRAME)   # ~0.40 px per component

# Configs: (label, solver_fn_name, solver_kwargs, shift_model)
CONFIGS = [
    ("AP + diag",
     "Alternating_projections_position",
     dict(method="diag", diag_method="taylor"),
     "fourier"),
    ("AP + diag + exact",
     "Alternating_projections_position",
     dict(method="diag", diag_method="exact"),
     "fourier"),
    ("RAAR + diag + exact",
     "RAAR_position",
     dict(beta=0.75, method="diag", diag_method="exact"),
     "fourier"),
    ("RAAR + diag (MATLAB-consistent)",
     "RAAR_position",
     dict(beta=0.75, method="diag", diag_method="taylor"),
     "taylor"),
    ("ADMM rho=0.5 + diag + exact",
     "ADMM_position",
     dict(rho=0.5, dual_step=0.0, method="diag", diag_method="exact"),
     "fourier"),
]


def build_problem(seed=0, shift_model="fourier"):
    import tempfile
    from position_simulate import simulate_to_h5, load_h5

    rng = np.random.default_rng(seed)
    nframes = NSCAN * NSCAN

    xi_x = (rng.random(nframes) - 0.5) * 2 * UNKNOWN_AMP
    xi_y = (rng.random(nframes) - 0.5) * 2 * UNKNOWN_AMP
    mean_xi = float(np.mean(np.sqrt(xi_x**2 + xi_y**2)))
    res_el = 1.0 / (R2 * NX_FRAME)
    print(f"  A={UNKNOWN_AMP:.3f} px/component, "
          f"(1/k)sum|xi_i|={mean_xi:.3f} px = {mean_xi/res_el:.2f} res_el  "
          f"(res_el={res_el:.3f} px)")

    fn = os.path.join(tempfile.gettempdir(), f"position_fig7_{seed}.h5")
    simulate_to_h5(
        fn, xi_x, xi_y,
        nx=NX_FRAME, nnx=NSCAN, step=DStep, r1=R1, r2=R2, contrast=CONTRAST,
        shift_model=shift_model,
    )
    p = load_h5(fn)
    for k in ("frames_data", "probe", "truth", "translations_x", "translations_y",
              "xi_x_truth", "xi_y_truth"):
        p[k] = xp.asarray(p[k])
    return p


def align_global(recon, truth):
    """Optimal complex scalar alignment: s*recon where s = <recon,truth>/<recon,recon>."""
    r = recon.ravel()
    t = truth.ravel()
    s = np.vdot(r, t) / (np.vdot(r, r) + 1e-30)
    return recon * s


def run_one(p, solver_fn, maxiter, position_start, **kw):
    img0 = xp.ones((p["Nx"], p["Ny"]), dtype=xp.complex64)
    t0 = time.perf_counter()
    img_final, _, _, _, res = solver_fn(
        img0, p["probe"], p["frames_data"],
        p["translations_x"], p["translations_y"],
        p["nx"], p["ny"], p["Nx"], p["Ny"],
        maxiter=maxiter,
        position_start=position_start,
        position_every=1,
        img_truth=p["truth"],
        xi_x_truth=p["xi_x_truth"],
        xi_y_truth=p["xi_y_truth"],
        residuals_interval=1,
        **kw,
    )
    dt = time.perf_counter() - t0
    return tonp(res[:, 3]), tonp(res[:, 0]), tonp(img_final), dt  # eps_xi, img_mse, img, time


def run_comparison(seed=0, maxiter=10000, position_start=100,
                   cache=os.path.join(_HERE, "position_fig7_results.npz"),
                   force=False):
    print(f"seed={seed}  maxiter={maxiter}  position_start={position_start}")

    # ---- load cache if available and maxiter matches -------------------------
    results  = {}   # label -> (eps_xi, img_mse)
    images   = {}   # label -> img_np   (NOT cached to npz — rerun if missing)
    timings  = {}   # label -> wall-clock seconds
    if not force and os.path.exists(cache):
        data = np.load(cache, allow_pickle=True)
        cached_maxiter = int(data["maxiter"])
        if cached_maxiter == maxiter:
            print(f"  loading cached results from {cache}")
            for label, *_ in CONFIGS:
                key = label.replace(" ", "_").replace("+", "p").replace("(", "").replace(")", "")
                if f"{key}_eps" in data:
                    results[label] = (data[f"{key}_eps"], data[f"{key}_mse"])
                    if f"{key}_dt" in data:
                        timings[label] = float(data[f"{key}_dt"])
                    if f"{key}_img" in data:
                        images[label] = data[f"{key}_img"]
                    print(f"  loaded {label}")

    # ---- run missing configs -------------------------------------------------
    print(f"\n{'method':>35} | {'eps_xi @start':>13} | {'eps_xi @end':>11} | {'img_mse @end':>12} | {'time(s)':>8}")
    print("-" * 90)
    for label, fn_name, kw, shift_model in CONFIGS:
        if label in results and label in images:
            eps_xi, img_mse = results[label]
            dt = timings.get(label, float("nan"))
            print(f"{label:>35} | {eps_xi[position_start]:>13.3e} | {eps_xi[-1]:>11.3e}"
                  f" | {img_mse[-1]:>12.3e} | {dt:>8.1f}")
            continue
        print(f"  running {label} ...", flush=True)
        p = build_problem(seed=seed, shift_model=shift_model)
        fn = getattr(Solvers, fn_name)
        eps_xi, img_mse, img_np, dt = run_one(p, fn, maxiter=maxiter,
                                               position_start=position_start, **kw)
        results[label] = (eps_xi, img_mse)
        images[label]  = img_np
        timings[label] = dt
        # keep truth for recon plot (last problem built)
        if "truth_np" not in dir():
            pass
        print(f"{label:>35} | {eps_xi[position_start]:>13.3e} | {eps_xi[-1]:>11.3e}"
              f" | {img_mse[-1]:>12.3e} | {dt:>8.1f}")

    # ---- save cache ----------------------------------------------------------
    save_dict = {"maxiter": maxiter, "seed": seed}
    for label, *_ in CONFIGS:
        key = label.replace(" ", "_").replace("+", "p").replace("(", "").replace(")", "")
        if label in results:
            save_dict[f"{key}_eps"] = results[label][0]
            save_dict[f"{key}_mse"] = results[label][1]
        if label in timings:
            save_dict[f"{key}_dt"] = timings[label]
        if label in images:
            save_dict[f"{key}_img"] = images[label]
    np.savez(cache, **save_dict)
    print(f"\n  results cached to {cache}")

    # ---- need images for recon plot: re-run any that are missing -------------
    if not images:
        print("  re-running to collect final images (not cached) ...")
        for label, fn_name, kw, shift_model in CONFIGS:
            p = build_problem(seed=seed, shift_model=shift_model)
            fn = getattr(Solvers, fn_name)
            _, _, img_np, _ = run_one(p, fn, maxiter=maxiter,
                                       position_start=position_start, **kw)
            images[label] = img_np
        truth_np = tonp(p["truth"])
    else:
        # truth from the last problem built during the run loop
        truth_np = tonp(build_problem(seed=seed, shift_model="fourier")["truth"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        styles = ["-", "--", "-.", ":"]
        it = np.arange(maxiter)

        # ---- Figure 1: convergence curves (eps_xi + img_mse) ----------------
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, idx, ylabel in zip(axes, [0, 1],
                                   [r"$\varepsilon_\xi$ (position error)",
                                    r"$\varepsilon_0$ (image MSE)"]):
            for (label, *_), c, ls in zip(CONFIGS, colors, styles):
                dt_str = f"  [{timings[label]:.0f}s]" if label in timings else ""
                ax.semilogy(it, results[label][idx],
                            label=label + dt_str, color=c, ls=ls, lw=2)
            ax.axvline(position_start, color="gray", ls="--", lw=1,
                       label=f"position start ({position_start})")
            ax.set_xlabel("iteration")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)
            ax.set_title(ylabel)
            ax.grid(True, which="both", alpha=0.3)

        fig.suptitle(
            f"Fig. 7 reproduction  —  {NSCAN}×{NSCAN} frames, "
            f"$\\langle|\\xi_0|\\rangle$=2.5 res_el, R2={R2}, {maxiter} iters",
            fontsize=11,
        )
        plt.tight_layout()
        out1 = os.path.join(_HERE, "position_fig7.png")
        plt.savefig(out1, dpi=120)
        print(f"  saved {out1}")
        plt.close(fig)

        # ---- Figure 2: reconstruction panels ---------------------------------
        # Rows: |amp|, phase.  Cols: truth, then one per method.
        n_methods = len(CONFIGS)
        fig2, axes2 = plt.subplots(2, n_methods + 1,
                                   figsize=(3.2 * (n_methods + 1), 7))

        def _show(ax, data, title, cmap, vmin=None, vmax=None):
            im = ax.imshow(data.T, origin="lower", cmap=cmap,
                           interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        truth_amp = np.abs(truth_np)
        _show(axes2[0, 0], truth_amp,   "Truth |obj|",   "viridis")
        _show(axes2[1, 0], np.angle(truth_np), "Truth phase", "hsv",
              vmin=-np.pi, vmax=np.pi)

        amp_range = (truth_amp.min(), truth_amp.max())
        for ci, (label, *_) in enumerate(CONFIGS):
            img_np = images[label]
            recon  = align_global(img_np, truth_np)
            residual = np.abs(truth_np - recon)
            eps_xi_f = results[label][0][-1]
            img_mse_f = results[label][1][-1]
            dt = timings.get(label, float("nan"))
            short = label.replace("MATLAB-consistent", "MATLAB")
            _show(axes2[0, ci + 1], np.abs(recon),
                  f"{short}\n|recon|  ε_ξ={eps_xi_f:.2e}",
                  "viridis", *amp_range)
            _show(axes2[1, ci + 1], residual,
                  f"|truth−recon|  mse={img_mse_f:.2e}\n{dt:.0f}s",
                  "hot", vmin=0)

        fig2.suptitle(
            f"Reconstruction comparison  —  {maxiter} iters, seed={seed}",
            fontsize=11,
        )
        plt.tight_layout()
        out2 = os.path.join(_HERE, "position_fig7_recon.png")
        plt.savefig(out2, dpi=120)
        print(f"  saved {out2}")
        plt.close(fig2)

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"(plot skipped: {e})")

    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore cache, re-run all")
    ap.add_argument("--maxiter", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_comparison(seed=args.seed, maxiter=args.maxiter, force=args.force)
