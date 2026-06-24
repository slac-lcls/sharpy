"""
Memory-limit harness for sharpy ptycho-Gramian (GPU).

For each frame size (nx), doubles the number of frames (nnx) until the GPU
runs out of memory. Reports the last size that fit and which stage failed.
Writes a summary table and a bar chart.

Usage:
    python memory_limit_harness.py

Expected output:
    A table of (nx, max_nnx, max_nframes, failed_at_stage) and a bar chart
    showing the memory limit per frame size. The 'failed_at_stage' column
    tells you which wall you hit — e.g. "Gramiam_plan" means the O(nframes²)
    neighbor-finding matrix is the bottleneck, not the frame stack itself.
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
    sys.exit("memory_limit_harness.py requires config.GPU = True")

import cupy as cp

import Operators
import Solvers
from Operators import (
    make_probe, make_translations, Gramiam_plan, reset_times, get_times,
)
from wrap_ops import split_cuda, overlap_cuda

# ── sweep parameters ──────────────────────────────────────────────────────────
# For each nx (frame size), we double nnx (frames per side) until OOM.
# nnx starts at START_NNX and is doubled up to MAX_NNX.
FRAME_SIZES = [32, 64, 128, 256]
START_NNX   = 4       # nnx at which to begin (nframes = START_NNX²)
MAX_NNX     = 512     # stop even if no OOM (safety cap)
MAXITER     = 5       # minimal iters — we only care about memory, not convergence
CSV_OUT     = "memory_limit_results.csv"
PLOT_OUT    = "memory_limit_harness.png"

OOM = (cp.cuda.memory.OutOfMemoryError, MemoryError)


# ── helpers ───────────────────────────────────────────────────────────────────
def free_gpu():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def peak_mb():
    return cp.get_default_memory_pool().used_bytes() / 1e6


def total_gpu_mb():
    return cp.cuda.Device().mem_info[1] / 1e6   # total physical memory


# ── simulate ──────────────────────────────────────────────────────────────────
def simulate(nx, nnx, Dx, fname):
    ny, nny, Dy = nx, nnx, Dx
    nframes = nnx * nny

    Nx_scan = nnx * Dx
    Ny_scan = nny * Dy

    illumination, _ = make_probe(nx, ny, r1=0.075, r2=0.255, fx=+20, fy=-20)
    translations_x, translations_y = make_translations(Dx, Dy, nnx, nny, Nx_scan, Ny_scan)

    Nx = int(np.ceil(float(cp.max(translations_x) - cp.min(translations_x))))
    Ny = Nx

    np.random.seed(42)
    density = np.random.rand(Nx, Ny).astype(np.float32)
    truth = cp.array(np.exp(0.69 * (-1 + 0.5j) * density).astype(np.complex64))
    illumination = cp.array(illumination, dtype=cp.complex64)
    translations = (translations_x + 1j * translations_y).astype(cp.complex64)

    frames_out = cp.zeros((nframes, nx, ny), dtype=cp.complex64)
    split_cuda(truth, frames_out, translations, illumination)
    frames_data = np.abs(np.fft.fft2(frames_out.get())) ** 2

    with h5py.File(fname, "w") as fid:
        fid.create_dataset("truth",          data=truth.get())
        fid.create_dataset("data",           data=frames_data.astype(np.float32))
        fid.create_dataset("probe",          data=illumination.get())
        fid.create_dataset("translations_x", data=translations_x.get())
        fid.create_dataset("translations_y", data=translations_y.get())
        fid.create_dataset("wavelength",          data=0.1)
        fid.create_dataset("detector_pixel_size", data=100.0)
        fid.create_dataset("detector_distance",   data=nx * 100.0 / 0.1)


# ── probe one (nx, nnx) point ─────────────────────────────────────────────────
def probe_one(nx, nnx, Dx, fname):
    """
    Try to simulate + plan + reconstruct for (nx, nnx).

    Returns:
        ("ok",   peak_mb,  Nx, nframes)  — if it fit
        ("oom",  stage,    Nx, nframes)  — if it ran out of memory
            stage is one of: "simulate", "Gramiam_plan", "reconstruct"
    """
    nframes = nnx * nnx

    # ── stage 1: simulate (allocates frame stack on GPU briefly) ─────────────
    try:
        simulate(nx, nnx, Dx, fname)
    except OOM:
        return "oom", "simulate", None, nframes

    # ── stage 2: load data from h5 (CPU→GPU transfer) ────────────────────────
    try:
        with h5py.File(fname, "r") as fid:
            data          = cp.array(fid["data"],          dtype=cp.float32)
            illumination  = cp.array(fid["probe"],         dtype=cp.complex64)
            translations_x = cp.array(fid["translations_x"])
            translations_y = cp.array(fid["translations_y"])
            truth         = cp.array(fid["truth"],         dtype=cp.complex64)
    except OOM:
        return "oom", "data_load", None, nframes

    nf, nx_, ny_ = data.shape
    Nx = int(cp.ceil(cp.max(translations_x) - cp.min(translations_x)).item())
    Ny = Nx

    # ── stage 3: Gramiam_plan (builds nframes×nframes pairwise matrix) ────────
    try:
        Gplan = Gramiam_plan(translations_x, translations_y, nframes, nx_, ny_, Nx, Ny, bw=0)
    except OOM:
        return "oom", "Gramiam_plan", Nx, nframes

    if Gplan["col"].size == 0:
        return "oom", "no_overlap", Nx, nframes

    # ── stage 4: reconstruct ──────────────────────────────────────────────────
    img_initial = cp.ones((Nx, Ny), dtype=cp.complex64)
    reset_times()
    Solvers.reset_times()

    try:
        Solvers.Alternating_projections_c(
            True,
            img_initial,
            Gplan,
            illumination,
            translations_x,
            translations_y,
            overlap_cuda,
            split_cuda,
            data,
            False,
            MAXITER,
            normalization=None,
            img_truth=truth,
            residuals_interval=MAXITER,
            sync_interval=1,
            num_iter=10,
        )
    except OOM:
        return "oom", "reconstruct", Nx, nframes

    mem = peak_mb()
    return "ok", mem, Nx, nframes


# ── sweep one nx: double nnx until OOM ───────────────────────────────────────
def sweep_nx(nx):
    Dx = max(1, nx // 4)
    print(f"\n══ nx={nx}  Dx={Dx}  (GPU total: {total_gpu_mb():.0f} MB) ══")

    last_ok   = None   # (nnx, nframes, peak_mb, Nx)
    first_oom = None   # (nnx, nframes, stage)

    nnx = START_NNX
    while nnx <= MAX_NNX:
        nframes = nnx * nnx
        frame_stack_mb = nframes * nx * nx * 8 / 1e6   # complex64

        print(f"  nnx={nnx:4d}  nframes={nframes:6d}  "
              f"frame_stack≈{frame_stack_mb:.0f} MB ... ", end="", flush=True)

        free_gpu()
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            fname = f.name
        try:
            status, info, Nx, nf = probe_one(nx, nnx, Dx, fname)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)
        free_gpu()

        if status == "ok":
            peak = info
            print(f"OK  peak={peak:.0f} MB  Nx={Nx}")
            last_ok = {"nnx": nnx, "nframes": nf, "peak_mb": peak, "Nx": Nx}
        else:
            stage = info
            print(f"OOM at stage={stage}")
            first_oom = {"nnx": nnx, "nframes": nf, "stage": stage}
            break

        nnx *= 2

    return last_ok, first_oom


# ── main ──────────────────────────────────────────────────────────────────────
def run():
    results = []

    for nx in FRAME_SIZES:
        last_ok, first_oom = sweep_nx(nx)

        row = {"nx": nx}
        if last_ok:
            row.update({
                "max_nnx":    last_ok["nnx"],
                "max_nframes": last_ok["nframes"],
                "peak_mb":    last_ok["peak_mb"],
                "image_Nx":   last_ok["Nx"],
            })
        else:
            row.update({"max_nnx": 0, "max_nframes": 0, "peak_mb": 0, "image_Nx": 0})

        if first_oom:
            row.update({
                "oom_nnx":    first_oom["nnx"],
                "oom_nframes": first_oom["nframes"],
                "failed_at":  first_oom["stage"],
            })
        else:
            row.update({"oom_nnx": None, "oom_nframes": None, "failed_at": "none (hit MAX_NNX)"})

        results.append(row)

    return results


def print_summary(results):
    print("\n" + "═" * 70)
    print(f"{'nx':>6}  {'max nnx':>8}  {'max nframes':>12}  {'peak MB':>9}  {'failed at'}")
    print("─" * 70)
    for r in results:
        print(f"{r['nx']:>6}  {r['max_nnx']:>8}  {r['max_nframes']:>12}  "
              f"{r['peak_mb']:>9.0f}  {r['failed_at']}")
    print("═" * 70)


def write_csv(results, path=CSV_OUT):
    if not results:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"CSV written to {path}")


def plot_results(results, path=PLOT_OUT):
    nx_vals     = [r["nx"]          for r in results]
    max_nframes = [r["max_nframes"] for r in results]
    peak_mbs    = [r["peak_mb"]     for r in results]
    stages      = [r["failed_at"]   for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Memory limit harness — max problem size before OOM")

    # panel 1: max nframes per frame size
    ax = axes[0]
    bars = ax.bar([str(nx) for nx in nx_vals], max_nframes, color="steelblue")
    for bar, st in zip(bars, stages):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                st, ha="center", va="bottom", fontsize=8, rotation=15)
    ax.set_xlabel("frame size nx")
    ax.set_ylabel("max nframes (last successful)")
    ax.set_title("Max frames before OOM\n(label = stage that failed)")
    ax.set_yscale("log")

    # panel 2: peak GPU memory at the limit
    ax = axes[1]
    total = total_gpu_mb()
    ax.bar([str(nx) for nx in nx_vals], peak_mbs, color="coral",
           label="peak used")
    ax.axhline(total, color="k", linestyle="--", label=f"total GPU ({total:.0f} MB)")
    ax.set_xlabel("frame size nx")
    ax.set_ylabel("peak GPU memory (MB)")
    ax.set_title("Peak memory at the limit")
    ax.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"Plot saved to {path}")


if __name__ == "__main__":
    results = run()
    print_summary(results)
    write_csv(results)
    plot_results(results)
