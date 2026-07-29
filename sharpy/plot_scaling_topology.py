# -*- coding: utf-8 -*-
"""
Intra-node vs inter-node sensitivity, before (full AllReduce) vs after
(halo exchange) -- the controlled comparison the original Perlmutter chart
did (same rank count, forced onto 1 node vs spread across nodes), run fresh
on S3DF ampere for both code paths.

Run on any node (no GPU needed): python plot_scaling_topology.py
Outputs: topology_before_after.png
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── data (sharpy_mpi_skeleton.py --benchmark-strong, same 4096-frame dataset,
#          --nodes=1 for intra vs --nodes=N --ntasks-per-node=1 for inter) ────
ranks = np.array([2, 4])

# before: full mpi_allSum (SHARPY_NO_HALO=1)
before_intra_compute = np.array([4.6, 2.3])
before_intra_sync    = np.array([9.9, 11.3])
before_inter_compute = np.array([4.6, 2.3])
before_inter_sync    = np.array([16.5, 21.8])

# after: exchange_object_halo
after_intra_compute = np.array([4.6, 2.1])
after_intra_sync    = np.array([4.2, 3.5])
after_inter_compute = np.array([3.9, 2.2])
after_inter_sync    = np.array([3.6, 5.2])

C_COMPUTE = "#2a78d6"
C_SYNC    = "#eb6834"
plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "figure.dpi":     150,
})


def _panel(ax, title, intra_c, intra_s, inter_c, inter_s):
    x = np.arange(len(ranks))
    w = 0.32
    ax.bar(x - w/2 - 0.02, intra_c, w, color=C_COMPUTE, label="Compute")
    ax.bar(x - w/2 - 0.02, intra_s, w, bottom=intra_c, color=C_SYNC, label="Sync")
    ax.bar(x + w/2 + 0.02, inter_c, w, color=C_COMPUTE, hatch="////", edgecolor="white", linewidth=0.6)
    ax.bar(x + w/2 + 0.02, inter_s, w, bottom=inter_c, color=C_SYNC, hatch="////", edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ranks)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("MPI ranks")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f ms"))
    for xi in x:
        ax.text(xi - w/2 - 0.02, -0.02, "intra", transform=ax.get_xaxis_transform(),
                 ha="center", va="top", fontsize=7.5, color="#898781")
        ax.text(xi + w/2 + 0.02, -0.02, "inter", transform=ax.get_xaxis_transform(),
                 ha="center", va="top", fontsize=7.5, color="#898781")


fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
_panel(ax_l, "Before — full AllReduce", before_intra_compute, before_intra_sync,
       before_inter_compute, before_inter_sync)
_panel(ax_r, "After — halo exchange", after_intra_compute, after_intra_sync,
       after_inter_compute, after_inter_sync)
ax_l.set_ylabel("Time per iteration")
ymax = max((before_intra_compute + before_intra_sync).max(),
           (before_inter_compute + before_inter_sync).max()) * 1.25
ax_l.set_ylim(0, ymax)

from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor=C_COMPUTE, label="Compute"),
    Patch(facecolor=C_SYNC, label="Sync"),
    Patch(facecolor="white", edgecolor="#52514e", label="Intra-node (solid)"),
    Patch(facecolor="white", edgecolor="#52514e", hatch="////", label="Inter-node (hatched)"),
]
fig.legend(handles=legend_elems, loc="upper center", ncol=4, fontsize=8.5,
           bbox_to_anchor=(0.5, 1.06), framealpha=0.8)
fig.suptitle("Inter- vs intra-node sensitivity — same rank count, forced node placement\n"
             "S3DF ampere, fixed 4096-frame dataset",
             fontsize=11.5, y=1.19)
fig.tight_layout()
fig.savefig("topology_before_after.png", bbox_inches="tight")
plt.close(fig)
print("saved topology_before_after.png")

# ── print the ratio story ──────────────────────────────────────────────────
for i, r in enumerate(ranks):
    b_ratio = before_inter_sync[i] / before_intra_sync[i]
    a_ratio = after_inter_sync[i] / after_intra_sync[i]
    print(f"ranks={r}: inter/intra sync ratio  before={b_ratio:.2f}x  after={a_ratio:.2f}x")
