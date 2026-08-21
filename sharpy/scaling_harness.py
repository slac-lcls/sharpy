"""
Scaling harness for sharpy ptycho-Gramian (GPU).

Sweeps over frame sizes and frame counts, logs per-operator timing and peak
GPU memory after each run, writes one row per run to scaling_results.csv,
and saves a 3-panel bottleneck plot to scaling_harness.png.

Usage (interactive GPU node on Perlmutter):
    python scaling_harness.py

The sweep is defined by SWEEP below — edit it to add/remove size points.
"""

import sys
import os
import csv
import copy
import tempfile
import numpy as np
import h5py
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import config

if not config.GPU:
    sys.exit("scaling_harness.py requires config.GPU = True")

import cupy as cp

import Operators
import Solvers
from Operators import (
    make_probe, make_translations, Split_plan, Illuminate_frames,
    cropmat, Gramiam_plan, reset_times, get_times,
)
from wrap_ops import split_cuda, overlap_cuda

# ── sweep parameters ─────────────────────────────────────────────────────────
# Each entry is (nx, nnx): nx = frame size (pixels), nnx = frames per side.
# nframes = nnx². Dx is set to nx//4 (~75 % overlap) for each point.
SWEEP = [
    (32,   8),   #    64 frames, 32² px/frame
    (32,  16),   #   256 frames
    (32,  32),   #  1024 frames
    (64,   8),   #    64 frames, 64² px/frame
    (64,  16),   #   256 frames
    (64,  32),   #  1024 frames
    (128,  8),   #    64 frames, 128² px/frame
    (128, 16),   #   256 frames
]

MAXITER       = 50   # iterations per run — enough for steady-state throughput
SYNC_INTERVAL = 1    # apply Gramian sync every iteration
NUM_EIG_ITER  = 50   # power-iteration steps inside synchronize_frames_c
CSV_OUT       = "scaling_results.csv"
PLOT_OUT      = "scaling_harness.png"


# ── simulate one problem and write h5 ────────────────────────────────────────
def simulate(nx, nnx, Dx, fname):
    """Generate synthetic ptychography data and save to fname (h5)."""
    ny, nny, Dy = nx, nnx, Dx
    nframes = nnx * nny

    # image size from scan geometry
    Nx_scan = nnx * Dx
    Ny_scan = nny * Dy

    illumination, _ = make_probe(nx, ny, r1=0.075, r2=0.255, fx=+20, fy=-20)
    translations_x, translations_y = make_translations(Dx, Dy, nnx, nny, Nx_scan, Ny_scan)

    Nx = int(np.ceil(float(cp.max(translations_x) - cp.min(translations_x))))
    Ny = Nx

    # synthetic complex transmission object (|obj| ≈ 1, small phase/amplitude)
    np.random.seed(42)
    density = np.random.rand(Nx, Ny).astype(np.float32)
    truth = cp.array(np.exp(0.69 * (-1 + 0.5j) * density).astype(np.complex64))

    illumination = cp.array(illumination, dtype=cp.complex64)
    translations = (translations_x + 1j * translations_y).astype(cp.complex64)

    frames_out = cp.zeros((nframes, nx, ny), dtype=cp.complex64)
    split_cuda(truth, frames_out, translations, illumination)

    frames_np = frames_out.get()
    frames_data = np.abs(np.fft.fft2(frames_np)) ** 2

    with h5py.File(fname, "w") as fid:
        fid.create_dataset("truth",          data=truth.get())
        fid.create_dataset("data",           data=frames_data.astype(np.float32))
        fid.create_dataset("probe",          data=illumination.get())
        fid.create_dataset("translations_x", data=translations_x.get())
        fid.create_dataset("translations_y", data=translations_y.get())
        fid.create_dataset("wavelength",          data=0.1)
        fid.create_dataset("detector_pixel_size", data=100.0)
        fid.create_dataset("detector_distance",   data=nx * 100.0 / 0.1)


# ── reconstruct and return timing + memory ────────────────────────────────────
def reconstruct(fname):
    """Load h5, run AP+sync, return dict of timings and peak GPU memory."""
    with h5py.File(fname, "r") as fid:
        data          = cp.array(fid["data"],          dtype=cp.float32)
        illumination  = cp.array(fid["probe"],         dtype=cp.complex64)
        translations_x = cp.array(fid["translations_x"])
        translations_y = cp.array(fid["translations_y"])
        truth         = cp.array(fid["truth"],         dtype=cp.complex64)

    nframes, nx, ny = data.shape
    Nx = int(cp.ceil(cp.max(translations_x) - cp.min(translations_x)).item())
    Ny = Nx

    Gplan = Gramiam_plan(translations_x, translations_y, nframes, nx, ny, Nx, Ny, bw=0)
    if Gplan["col"].size == 0:
        raise RuntimeError(
            f"No overlapping frame pairs for nx={nx}, nframes={nframes}. "
            "Increase Dx overlap or check scan geometry."
        )

    img_initial = cp.ones((Nx, Ny), dtype=cp.complex64)

    # reset both timer dicts, free cached GPU memory before measuring
    reset_times()
    Solvers.reset_times()
    cp.get_default_memory_pool().free_all_blocks()
    mempool = cp.get_default_memory_pool()

    t_wall = timer()
    Solvers.Alternating_projections_c(
        True,            # sync=True (Gramian sync every iteration)
        img_initial,
        Gplan,
        illumination,
        translations_x,
        translations_y,
        overlap_cuda,
        split_cuda,
        data,
        False,           # refine_illumination
        MAXITER,
        normalization=None,
        img_truth=truth,
        residuals_interval=MAXITER,   # fires once at ii=0 — the START, not the end (the rule is
                                      # `not mod(ii, interval)`). Harmless: this return is unused.
        sync_interval=SYNC_INTERVAL,
        num_iter=NUM_EIG_ITER,
    )
    t_wall = timer() - t_wall

    peak_mem_mb = mempool.used_bytes() / 1e6

    return {
        "wall_time_s": t_wall,
        "peak_mem_mb": peak_mem_mb,
        "nframes": nframes,
        "nx": nx,
        "Nx": Nx,
        "op":     copy.deepcopy(get_times()),
        "solver": copy.deepcopy(Solvers.get_times()),
    }


# ── sweep ─────────────────────────────────────────────────────────────────────
def run_sweep():
    rows = []
    op_keys     = list(Operators.timers.keys())
    solver_keys = list(Solvers.timers.keys())

    for nx, nnx in SWEEP:
        Dx = max(1, nx // 4)
        nframes = nnx * nnx
        print(f"\n── nx={nx:3d}  nnx={nnx:3d}  nframes={nframes:5d}  Dx={Dx} ──")

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            fname = f.name
        try:
            print("  simulate ...", end=" ", flush=True)
            simulate(nx, nnx, Dx, fname)
            print("done")

            print("  reconstruct ...", end=" ", flush=True)
            r = reconstruct(fname)
            print(f"done  wall={r['wall_time_s']:.2f}s  mem={r['peak_mem_mb']:.0f} MB")
        finally:
            os.unlink(fname)

        row = {
            "nx":          nx,
            "nnx":         nnx,
            "nframes":     nframes,
            "Nx":          r["Nx"],
            "wall_time_s": r["wall_time_s"],
            "peak_mem_mb": r["peak_mem_mb"],
        }
        for k in op_keys:
            row[f"op_{k}"] = r["op"].get(k, 0.0)
        for k in solver_keys:
            row[f"solver_{k}"] = r["solver"].get(k, 0.0)
        rows.append(row)

    return rows


# ── CSV output ────────────────────────────────────────────────────────────────
def write_csv(rows, path=CSV_OUT):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV written to {path}")


# ── plot ──────────────────────────────────────────────────────────────────────
OP_KEYS_PLOT = [
    "Overlap", "Split", "Prox_data", "Propagate",
    "Gramiam", "Gramiam_completion", "Eigensolver",
    "Precondition", "Sync_setup",
]

def plot_results(rows, path=PLOT_OUT):
    nx_vals  = sorted(set(r["nx"]  for r in rows))
    colors   = plt.cm.tab10(np.linspace(0, 0.9, len(nx_vals)))
    nx_color = dict(zip(nx_vals, colors))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"Sharpy scaling harness  (maxiter={MAXITER}, sync_interval={SYNC_INTERVAL})")

    # panel 1 — wall time vs nframes
    ax = axes[0]
    for nx in nx_vals:
        sub = [r for r in rows if r["nx"] == nx]
        ax.plot([r["nframes"] for r in sub],
                [r["wall_time_s"] for r in sub],
                "o-", color=nx_color[nx], label=f"nx={nx}")
    ax.set_xscale("log")
    ax.set_xlabel("nframes")
    ax.set_ylabel("wall time (s)")
    ax.set_title(f"Wall time  ({MAXITER} iters)")
    ax.legend()

    # panel 2 — operator breakdown stacked bar (largest nnx per nx)
    ax = axes[1]
    x_ticks, x_labels = [], []
    bottoms = np.zeros(len(nx_vals))
    bar_data = {k: [] for k in OP_KEYS_PLOT}
    for i, nx in enumerate(nx_vals):
        sub = [r for r in rows if r["nx"] == nx]
        r = sub[-1]    # biggest nnx for this nx
        x_ticks.append(i)
        x_labels.append(f"nx={nx}\nn={r['nframes']}")
        for k in OP_KEYS_PLOT:
            bar_data[k].append(r["op"].get(k, 0.0))
    bottoms = np.zeros(len(nx_vals))
    for k in OP_KEYS_PLOT:
        vals = np.array(bar_data[k])
        ax.bar(x_ticks, vals, bottom=bottoms, label=k)
        bottoms += vals
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("cumulative time (s)")
    ax.set_title("Operator breakdown (largest nnx per nx)")
    ax.legend(fontsize=7)

    # panel 3 — peak GPU memory vs nframes
    ax = axes[2]
    for nx in nx_vals:
        sub = [r for r in rows if r["nx"] == nx]
        ax.plot([r["nframes"] for r in sub],
                [r["peak_mem_mb"] for r in sub],
                "o-", color=nx_color[nx], label=f"nx={nx}")
    ax.set_xscale("log")
    ax.set_xlabel("nframes")
    ax.set_ylabel("peak GPU memory (MB)")
    ax.set_title("Peak GPU memory")
    ax.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"Plot saved to {path}")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rows = run_sweep()
    write_csv(rows)
    plot_results(rows)
