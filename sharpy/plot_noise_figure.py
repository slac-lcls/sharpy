#!/usr/bin/env python3
"""Plot the sync noise-feasibility envelope (report figure) from sync_noise_envelope.npz.
X-axis = photons/PIXEL (transferable across frame size). Panel A: long-range-phase error
vs photons/pixel for increasing OVERLAP. Panel B: same vs SCALE (frame count) at fixed
overlap (threshold ~flat in photons/pixel -> feasible at fixed dose/pixel).

  python plot_noise_figure.py [npz] [out_basename]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

npz = sys.argv[1] if len(sys.argv) > 1 else "sync_noise_envelope.npz"
out = sys.argv[2] if len(sys.argv) > 2 else "fig_noise_envelope"
d = np.load(npz, allow_pickle=True)
P = d["pppx"]                                   # photons/pixel
NX = int(d["nx"]) if "nx" in d else 128
cmap = plt.cm.viridis

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.3))

# --- Panel A: overlap ---
r2s = [0.40, 0.30, 0.22]
for k, r2 in enumerate(r2s):
    eig = d[f"A_r2_{r2}_eigsh"]
    tag = "  (small probe, less overlap)" if r2 == max(r2s) else "  (large probe, more overlap)" if r2 == min(r2s) else ""
    axA.plot(P, eig, "-o", color=cmap(k / max(1, len(r2s) - 1)), label=f"r2={r2}{tag}", lw=2, ms=5)
axA.axhline(np.mean([d[f"A_r2_{r2}_nosync"] for r2 in r2s]), color="grey", ls="--", lw=1.2, label="no-sync")
axA.axvline(0.5, color="crimson", ls=":", lw=1.2, alpha=0.7)
axA.set_xscale("log")
axA.set_xlabel("photons / pixel"); axA.set_ylabel("long-range-phase NMSE (low-freq band)")
axA.set_title(f"(a) More overlap -> lower photon threshold\n({NX}x{NX} px frames; smaller r2 = larger probe)")
axA.set_ylim(0, 1.05); axA.grid(True, which="both", alpha=0.3); axA.legend(fontsize=8, loc="upper right")

# --- Panel B: scale ---
nnxs = [16, 24, 32]
for k, nnx in enumerate(nnxs):
    if f"B_n_{nnx}_eigsh" not in d:
        continue
    nf = int(d[f"B_n_{nnx}_nf"]); eig = d[f"B_n_{nnx}_eigsh"]
    axB.plot(P, eig, "-s", color=cmap(k / max(1, len(nnxs) - 1)), label=f"{nf} frames", lw=2, ms=5)
axB.axhline(np.mean([d[f"B_n_{nnx}_nosync"] for nnx in nnxs if f"B_n_{nnx}_nosync" in d]),
            color="grey", ls="--", lw=1.2, label="no-sync")
axB.axvline(0.5, color="crimson", ls=":", lw=1.2, alpha=0.7, label="~0.5 ph/px")
axB.set_xscale("log")
axB.set_xlabel("photons / pixel"); axB.set_ylabel("long-range-phase NMSE")
axB.set_title(f"(b) Threshold ~flat vs frame count\n({NX}x{NX} px frames; overlap fixed)")
axB.set_ylim(0, 1.05); axB.grid(True, which="both", alpha=0.3); axB.legend(fontsize=8, loc="upper right")

fig.suptitle(f"Photon-noise feasibility envelope for synchronization "
             f"({NX}x{NX} px frames; lower = better; threshold ~0.5 photons/pixel)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=150)
print("wrote", out + ".pdf", "and", out + ".png")
