# -*- coding: utf-8 -*-
"""
Before/after MPI scaling figures for the object-AllReduce -> halo-exchange fix.
Run on any node (no GPU needed): python plot_scaling_halo.py
Outputs: breakdown_before_after.png, strong_scaling_before_after.png,
         weak_scaling_before_after.png

Data below is measured on the SAME S3DF ampere nodes / same H5 datasets for
both "before" (SHARPY_NO_HALO=1, the original mpi_allSum path) and "after"
(exchange_object_halo) -- an apples-to-apples comparison, unlike the historical
Perlmutter numbers in plot_scaling.py.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── data (sharpy_mpi_skeleton.py --benchmark-strong / --benchmark-weak,
#          5 warmup + 50 timed iters, CUDA-synced) ─────────────────────────────
ranks_s = np.array([1, 2, 4, 8, 16])
compute_s_before = np.array([7.8, 4.5, 2.3, 1.3, 0.7])
sync_s_before    = np.array([3.5, 9.5, 13.4, 15.3, 17.3])
compute_s_after  = np.array([7.8, 4.6, 2.3, 1.3, 0.7])
sync_s_after     = np.array([0.0, 4.1, 3.5, 3.4, 3.6])
total_s_before = compute_s_before + sync_s_before
total_s_after  = compute_s_after + sync_s_after

ranks_w = np.array([1, 2, 4, 8])
NNX_w   = np.array([45, 64, 90, 128])
Nx_w    = np.array([1908, 2668, 3708, 5228])
compute_w_before = np.array([17.5, 4.6, 22.2, 4.8])
sync_w_before    = np.array([4.5, 39.2, 84.4, 234.4])
compute_w_after  = np.array([14.6, 4.6, 19.2, 4.7])
sync_w_after     = np.array([0.0, 17.1, 32.9, 59.7])
total_w_before = compute_w_before + sync_w_before
total_w_after  = compute_w_after + sync_w_after

# ── style ─────────────────────────────────────────────────────────────────────
C_COMPUTE = "#2a78d6"   # blue
C_SYNC    = "#eb6834"   # orange
plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "figure.dpi":     150,
})


def _paired_stacked(ax, ranks, comp_b, sync_b, comp_a, sync_a, title, xlabels=None):
    """One rank-count group per x tick, Before/After bars side by side, each
    stacked Compute+Sync."""
    x = np.arange(len(ranks))
    w = 0.32
    ax.bar(x - w/2 - 0.02, comp_b, w, color=C_COMPUTE, alpha=0.5, label="Compute (before)")
    ax.bar(x - w/2 - 0.02, sync_b, w, bottom=comp_b, color=C_SYNC, alpha=0.5, label="Sync (before)")
    ax.bar(x + w/2 + 0.02, comp_a, w, color=C_COMPUTE, label="Compute (after)")
    ax.bar(x + w/2 + 0.02, sync_a, w, bottom=comp_a, color=C_SYNC, label="Sync (after)")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels if xlabels is not None else ranks)
    ax.set_title(title, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f ms"))
    for xi in x:
        ax.text(xi - w/2 - 0.02, -0.02, "B", transform=ax.get_xaxis_transform(),
                 ha="center", va="top", fontsize=7.5, color="#898781")
        ax.text(xi + w/2 + 0.02, -0.02, "A", transform=ax.get_xaxis_transform(),
                 ha="center", va="top", fontsize=7.5, color="#898781")


# ── Figure 1: compute vs sync breakdown, before/after paired ──────────────────
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)

_paired_stacked(ax_l, ranks_s, compute_s_before, sync_s_before, compute_s_after, sync_s_after,
                "Strong scaling\n(fixed 4096 frames, 56 MB object)")
ax_l.set_ylim(0, max(total_s_before.max(), total_s_after.max()) * 1.25)
ax_l.set_xlabel("MPI ranks   (B = before, A = after)")
ax_l.set_ylabel("Time per iteration")

xlabels_w = [f"{r}\n({nx}px obj)" for r, nx in zip(ranks_w, Nx_w)]
_paired_stacked(ax_r, ranks_w, compute_w_before, sync_w_before, compute_w_after, sync_w_after,
                "Weak scaling\n(~2048 frames/rank, growing object)", xlabels=xlabels_w)
ax_r.set_ylim(0, max(total_w_before.max(), total_w_after.max()) * 1.25)
ax_r.set_xlabel("MPI ranks   (B = before, A = after)")

handles, labels = ax_l.get_legend_handles_labels()
fig.legend(handles[:2], ["Compute (PASS A+B)", "Sync (AllReduce / halo)"],
           loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.10), framealpha=0.8)
fig.suptitle("Compute vs. synchronisation — before (full AllReduce) vs after (halo exchange)\n"
             "faded bar = before  ·  solid bar = after   (same S3DF ampere nodes, same datasets)",
             fontsize=11.5, y=1.24)
fig.tight_layout()
fig.savefig("breakdown_before_after.png", bbox_inches="tight")
plt.close(fig)
print("saved breakdown_before_after.png")

# ── Figure 2: strong scaling speedup/efficiency, before vs after ──────────────
speedup_before = total_s_before[0] / total_s_before
speedup_after  = total_s_after[0] / total_s_after
eff_before = speedup_before / ranks_s * 100
eff_after  = speedup_after / ranks_s * 100

fig, ax1 = plt.subplots(figsize=(6.5, 4.6))
ax2 = ax1.twinx()
ax2.spines["top"].set_visible(False)

C_IDEAL = "#999999"
ax1.plot(ranks_s, ranks_s.astype(float), "--", color=C_IDEAL, lw=1.5, label="Ideal speedup")
ax1.plot(ranks_s, speedup_before, "o--", color=C_COMPUTE, lw=2, ms=7, alpha=0.5, label="Speedup (before)")
ax1.plot(ranks_s, speedup_after, "o-", color=C_COMPUTE, lw=2, ms=7, label="Speedup (after)")
ax2.plot(ranks_s, eff_before, "s--", color=C_SYNC, lw=1.5, ms=6, alpha=0.5, label="Efficiency % (before)")
ax2.plot(ranks_s, eff_after, "s-", color=C_SYNC, lw=1.5, ms=6, label="Efficiency % (after)")

ax1.set_xlabel("MPI ranks")
ax1.set_ylabel("Speedup  T₁ / Tₙ", color=C_COMPUTE)
ax2.set_ylabel("Parallel efficiency (%)", color=C_SYNC)
ax1.tick_params(axis="y", labelcolor=C_COMPUTE)
ax2.tick_params(axis="y", labelcolor=C_SYNC)
ax1.set_xticks(ranks_s)
ax1.set_ylim(0, max(ranks_s.max(), speedup_after.max()) * 1.1)
ax2.set_ylim(0, 110)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8, framealpha=0.9, ncol=1)
ax1.set_title("Strong scaling — 4096 frames, fixed 56 MB object\n(S3DF ampere, before vs after halo exchange)", fontsize=11)
fig.tight_layout()
fig.savefig("strong_scaling_before_after.png", bbox_inches="tight")
plt.close(fig)
print("saved strong_scaling_before_after.png")

# ── Figure 3: weak scaling wall time, before/after paired bars ────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.6))
_paired_stacked(ax, ranks_w, compute_w_before, sync_w_before, compute_w_after, sync_w_after,
                "Weak scaling — ~2048 frames/rank, growing object\n(S3DF ampere, before vs after halo exchange)",
                xlabels=xlabels_w)
ax.set_xlabel("MPI ranks  (obj = object size px)   (B = before, A = after)")
ax.set_ylabel("Wall time per iteration (ms)")
ax.set_ylim(0, max(total_w_before.max(), total_w_after.max()) * 1.2)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], ["Compute (PASS A+B)", "Sync (AllReduce / halo)"], fontsize=9, framealpha=0.8)
fig.tight_layout()
fig.savefig("weak_scaling_before_after.png", bbox_inches="tight")
plt.close(fig)
print("saved weak_scaling_before_after.png")
