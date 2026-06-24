#!/usr/bin/env python3
"""Plot the sync noise-feasibility envelope (report figure) from sync_noise_envelope.npz.
Panel A: long-range-phase error vs photons/frame for increasing OVERLAP (threshold shifts
left -> overlap buys noise tolerance). Panel B: same vs SCALE (frame count) at fixed
photons/frame (threshold ~flat -> feasible at fixed dose/frame; at fixed TOTAL dose, more
frames = fewer photons/frame = moving left across the threshold).

  python plot_noise_figure.py [path/to/sync_noise_envelope.npz] [out_basename]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

npz = sys.argv[1] if len(sys.argv) > 1 else "sync_noise_envelope.npz"
out = sys.argv[2] if len(sys.argv) > 2 else "fig_noise_envelope"
d = np.load(npz, allow_pickle=True)
P = d["photons"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.3))
cmap = plt.cm.viridis

# --- Panel A: overlap ---
# r2 = lens aperture: SMALLER r2 = LARGER probe = MORE overlap area (more shared photons/pair)
r2s = [0.40, 0.30, 0.22]
for k, r2 in enumerate(r2s):
    eig = d[f"A_r2_{r2}_eigsh"]
    tag = "  (small probe, less overlap)" if r2 == max(r2s) else "  (large probe, more overlap)" if r2 == min(r2s) else ""
    axA.plot(P, eig, "-o", color=cmap(k / max(1, len(r2s) - 1)), label=f"r2={r2}{tag}", lw=2, ms=5)
axA.axhline(np.mean([d[f"A_r2_{r2}_nosync"] for r2 in r2s]), color="grey", ls="--", lw=1.2,
            label="no-sync")
axA.set_xscale("log")
axA.set_xlabel("photons / frame"); axA.set_ylabel("long-range-phase NMSE (low-freq band)")
axA.set_title("(a) More overlap → lower photon threshold\n(4096 frames; smaller r2 = larger probe = more overlap)")
axA.set_ylim(0, 1.05); axA.grid(True, which="both", alpha=0.3); axA.legend(fontsize=8, loc="lower left")

# --- Panel B: scale ---
nnxs = [32, 64, 96]
for k, nnx in enumerate(nnxs):
    nf = int(d[f"B_n_{nnx}_nf"])
    eig = d[f"B_n_{nnx}_eigsh"]
    axB.plot(P, eig, "-s", color=cmap(k / max(1, len(nnxs) - 1)),
             label=f"{nf} frames", lw=2, ms=5)
axB.axhline(np.mean([d[f"B_n_{nnx}_nosync"] for nnx in nnxs]), color="grey", ls="--", lw=1.2,
            label="no-sync")
axB.set_xscale("log")
axB.set_xlabel("photons / frame"); axB.set_ylabel("long-range-phase NMSE")
axB.set_title("(b) Threshold ~flat vs scale at fixed dose/frame\n(overlap fixed)")
axB.set_ylim(0, 1.05); axB.grid(True, which="both", alpha=0.3); axB.legend(fontsize=8, loc="lower left")

fig.suptitle("Photon-noise feasibility envelope for synchronization "
             "(long-range phase recovery; lower = better)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=150)
print("wrote", out + ".pdf", "and", out + ".png")
