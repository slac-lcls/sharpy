"""
Why does the local (diagonal) position solver fail on a large smooth drift,
even though each frame's error RELATIVE to its overlapping neighbors stays tiny?

Claim: the gate is the ABSOLUTE per-frame shift (frame vs the consensus image),
NOT the pairwise neighbor difference. The diagonal solver registers each frame
against the synthesized image, and its Taylor model is only valid for
|xi| <~ 1/k_max. A smooth ramp keeps pairwise diffs tiny but lets the absolute
shift grow without bound -> it fails once the absolute shift leaves the capture
range, with pairwise still far below any "2 px" threshold.

This test ISOLATES the capture range: the TRUE image is held fixed (so the
"corrupted-consensus" failure mode is removed) and the data is generated with
the EXACT band-limited shift (shift_probe_fourier), not the Taylor model the
solver inverts -- so the only thing that can break is the Taylor capture range.

Run:  python position_capture_test.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: F401  (CPU/GPU switch; this test is CPU)

from Operators import make_probe, map_frames, Splitc
from position_retrieval import (
    probe_derivatives,
    shift_probe_fourier,
    apodize_probe,
    position_solve_diag,
    shift_rmse,
)
from position_simulate import transmission_object


def build_scene(nx=32, nnx=16, step=3.5, r1=0.075, r2=0.255, contrast=4.1):
    """Apodized zone-plate probe + transmission object + a periodic scan grid."""
    ny = nx
    Nx = Ny = int(round(nnx * step))

    truth = transmission_object(Nx, Ny, contrast=contrast)

    probe = make_probe(nx, ny, r1=r1, r2=r2)
    if isinstance(probe, tuple):
        probe = probe[0]
    probe = np.asarray(probe / np.abs(probe).max(), dtype=np.complex64)
    probe = apodize_probe(probe)

    gx, gy = np.meshgrid(np.arange(nnx), np.arange(nnx), indexing="ij")
    gx = gx.ravel()
    gy = gy.ravel()
    tx = (gx * step + np.floor(step / 2) * (gx % 2)).astype(np.float64)
    ty = (gy * step).astype(np.float64)
    tx = np.round(tx)
    ty = np.round(ty)
    mapid = map_frames(tx, ty, nx, ny, Nx, Ny)
    dp = probe_derivatives(probe)
    return dict(truth=truth, probe=probe, dp=dp, mapid=mapid,
                gx=gx, gy=gy, tx=tx, ty=ty, nx=nx, Nx=Nx, Ny=Ny,
                nframes=gx.size, step=step)


def overlap_neighbor_maxdiff(xi_x, xi_y, gx, gy):
    """Largest |dxi| between spatially adjacent (overlapping) scan points."""
    nf = xi_x.size
    pos = {(int(gx[i]), int(gy[i])): i for i in range(nf)}
    worst = 0.0
    for i in range(nf):
        for d in ((1, 0), (0, 1), (1, 1), (1, -1)):
            j = pos.get((int(gx[i]) + d[0], int(gy[i]) + d[1]))
            if j is None:
                continue
            dr = np.hypot(xi_x[i] - xi_x[j], xi_y[i] - xi_y[j])
            worst = max(worst, dr)
    return worst


def recover(scene, xi_x, xi_y, niter=40, max_step=0.5):
    """Bake the TRUE (Fourier) shift into the data, recover from xi=0, fixed image."""
    probe_shifted = shift_probe_fourier(scene["probe"], xi_x, xi_y)
    frames = Splitc(scene["truth"], scene["mapid"]) * probe_shifted
    hx = np.zeros(scene["nframes"])
    hy = np.zeros(scene["nframes"])
    for _ in range(niter):
        hx, hy = position_solve_diag(frames, scene["dp"], scene["truth"],
                                     scene["mapid"], scene["Nx"], scene["Ny"],
                                     hx, hy, max_step=max_step)
    return shift_rmse(xi_x, xi_y, hx, hy)


def centered_ramp(scene, max_abs):
    """Smooth 1-D spatial ramp along the scan x-axis, mean removed, so the
    largest |xi - mean| equals max_abs. Pairwise (neighbor) diff is tiny."""
    g = scene["gx"].astype(np.float64)
    g = g - g.mean()
    g = g / np.abs(g).max() * max_abs       # so max |centered xi| = max_abs
    return g.copy(), np.zeros_like(g)


def iid_field(scene, max_abs, seed=0):
    """i.i.d. random field with the same max |xi - mean| (large pairwise diffs)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(scene["nframes"])
    y = rng.standard_normal(scene["nframes"])
    x -= x.mean(); y -= y.mean()
    s = max(np.abs(x).max(), np.abs(y).max())
    return x / s * max_abs, y / s * max_abs


if __name__ == "__main__":
    scene = build_scene()
    print(f"zone-plate probe, transmission object, {scene['nframes']} frames, "
          f"image {scene['Nx']}x{scene['Ny']}, fixed TRUE image\n")

    # ---- Part 1: capture-range sweep on a SMOOTH ramp -----------------------
    print("Part 1 -- smooth ramp, sweep absolute magnitude")
    print(f"{'max|xi| (abs)':>13} {'max neighbor diff':>18} {'final Delta_r':>14}")
    print("-" * 48)
    for max_abs in (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0):
        xx, yy = centered_ramp(scene, max_abs)
        pw = overlap_neighbor_maxdiff(xx, yy, scene["gx"], scene["gy"])
        dr = recover(scene, xx, yy)
        flag = "  <-- capture lost" if dr > 0.1 * max_abs and dr > 0.1 else ""
        print(f"{max_abs:>13.2f} {pw:>18.3f} {dr:>14.3e}{flag}")

    # ---- Part 2: same absolute, tiny vs large pairwise ---------------------
    print("\nPart 2 -- ramp (tiny pairwise) vs i.i.d. (large pairwise), matched max|xi|")
    print(f"{'max|xi| (abs)':>13} {'field':>6} {'max nbr diff':>13} {'final Delta_r':>14}")
    print("-" * 50)
    for max_abs in (0.8, 4.0):
        for label, (xx, yy) in (("ramp", centered_ramp(scene, max_abs)),
                                ("iid", iid_field(scene, max_abs))):
            pw = overlap_neighbor_maxdiff(xx, yy, scene["gx"], scene["gy"])
            dr = recover(scene, xx, yy)
            print(f"{max_abs:>13.2f} {label:>6} {pw:>13.3f} {dr:>14.3e}")

    print("\nReading the result:")
    print("  Part 1: Delta_r stays ~0 until max|xi| ~ 1 px, then blows up --")
    print("          while the neighbor diff is still far below 2 px.")
    print("  Part 2: ramp and i.i.d. pass/fail TOGETHER at the same max|xi|,")
    print("          despite wildly different pairwise diffs -> the gate is the")
    print("          ABSOLUTE per-frame shift (vs the image), not pairwise.")
