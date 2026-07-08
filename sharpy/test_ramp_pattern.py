"""
Diagnostic: verify 1D ramp shift has small LOCAL pairwise diffs but large global drift.

Geometry: single line of N frames, step ~ nx/2 (50% overlap), drift accumulates along scan.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
config.GPU = False
import numpy as np

from position_retrieval import position_plan

nx        = 32
step      = 16     # ~50% overlap: each frame overlaps only with ~2 neighbors each side
N         = 50     # number of frames in the line
nbr_step  = 0.10   # per-frame drift increment (px)
Nx        = N * step
Ny        = nx

tx = np.arange(N, dtype=np.float64) * step
ty = np.zeros(N, dtype=np.float64)
nframes = N

# Linearly accumulating drift along scan
xi_x = (np.arange(N) - N/2) * nbr_step   # mean-removed linear ramp
xi_y = np.zeros(N)

plan = position_plan(tx, ty, nframes, nx, nx, Nx, Ny)
col, row_p = plan['col'], plan['row']
offd = col != row_p
col_o = col[offd]; row_o = row_p[offd]
diffs = np.abs(xi_x[col_o] - xi_x[row_o])
scan_dist = np.abs(col_o.astype(int) - row_o.astype(int))

print(f"1D scan: N={N} frames, step={step}, nx={nx}, overlap={nx-step} px")
print(f"Drift: per-frame increment={nbr_step} px")
print(f"max|xi_x| = {np.abs(xi_x).max():.3f} px  (total drift range = {xi_x.max()-xi_x.min():.2f} px)")
print(f"\nPosition plan: {np.sum(offd)} off-diagonal pairs")
print(f"  scan-distance distribution of neighbor pairs:")
for d in sorted(set(scan_dist)):
    n = np.sum(scan_dist == d)
    mean_diff = diffs[scan_dist == d].mean()
    print(f"    |i-j|={d}: {n} pairs, mean pairwise |dxi|={mean_diff:.4f} px")

print(f"\nAll pairwise |xi_x[i]-xi_x[j]|:")
print(f"  min  = {diffs.min():.4f} px")
print(f"  mean = {diffs.mean():.4f} px")
print(f"  max  = {diffs.max():.4f} px")
print(f"  p50  = {np.percentile(diffs, 50):.4f} px")
print(f"  p95  = {np.percentile(diffs, 95):.4f} px")
print(f"\nRatio max_abs / mean_pairwise = {np.abs(xi_x).max() / diffs.mean():.1f}x")
print(f"(want >> 1 for the long-range drift regime)")
