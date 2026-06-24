#!/usr/bin/env python3
"""CPU validation of the wired-in inverse-iteration sync eigensolver.

Calls Operators.Eigensolver_invit (the A4 / arXiv:1209.4924 inverse-iteration on the
connection Laplacian, now wired into synchronize_frames_c via SHARPY_SYNC=invit) on a
synthetic zero-diagonal connection-adjacency H built from a real G x G overlap graph
with a low-frequency object phase, and checks it recovers the known phase. Compares to
an inline power-iteration baseline (Operators.Eigensolver is GPU-only on CPU). Also
times the sparse solve vs frame count.

  G=48 CONTRAST=0.85 python invit_wiring_test.py
"""
import os, time
import numpy as np
from scipy import sparse

import config; config.GPU = False
import Operators as op

CONTRAST = float(os.environ.get("CONTRAST", 0.85))
SIGMA, R = 1.6, 4


def low_freq_phase(G, cycles):
    xs, ys = np.meshgrid(np.linspace(-1, 1, G), np.linspace(-1, 1, G), indexing="ij")
    return (cycles * 2 * np.pi * (xs ** 2 + ys ** 2) / 2.0).ravel()


def frame_weights(G, contrast):
    N = G * G
    if contrast <= 0:
        return np.ones(N)
    xs, ys = np.meshgrid(np.linspace(0, 1, G), np.linspace(0, 1, G), indexing="ij")
    blob = np.zeros((G, G))
    for cx, cy, a in [(0.5, 0.5, 0.9), (0.3, 0.7, 0.7), (0.75, 0.35, 0.6)]:
        blob += a * np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / 0.02))
    return np.clip(1.0 - contrast * np.clip(blob, 0, 1), 0.1, 1.0).ravel()


def build_adj(G, cycles, contrast):
    """Zero-diagonal complex connection adjacency H (what synchronize_frames_c assembles)."""
    N = G * G
    s = frame_weights(G, contrast)
    theta = low_freq_phase(G, cycles)
    xs, ys = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    xs = xs.ravel(); ys = ys.ravel()
    ri, ci, vv = [], [], []
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            d2 = dx * dx + dy * dy
            if d2 == 0 or d2 > R * R:        # exclude self -> zero diagonal
                continue
            xn, yn = xs + dx, ys + dy
            ok = (xn >= 0) & (xn < G) & (yn >= 0) & (yn < G)
            i = np.where(ok)[0]; j = xn[ok] + yn[ok] * G
            w = s[i] * s[j] * np.exp(-d2 / (SIGMA * SIGMA))
            ri.append(i); ci.append(j)
            vv.append(w * np.exp(1j * (theta[i] - theta[j])))
    H = sparse.csr_matrix((np.concatenate(vv), (np.concatenate(ri), np.concatenate(ci))),
                          shape=(N, N)).astype(np.complex128)
    return H, np.exp(1j * theta)


def phase_err_rms(w, phi):
    wn = w / (np.abs(w) + 1e-30)
    g = np.vdot(phi, wn); wn = wn * np.conj(g) / (np.abs(g) + 1e-30)
    return float(np.sqrt(np.mean(np.angle(wn * np.conj(phi)) ** 2)))


def alignment(w, phi):
    wn = w / (np.abs(w) + 1e-30)
    return float(np.abs(np.vdot(phi, wn)) / phi.size)


def power(H, it):
    d = np.asarray(abs(H).sum(1)).ravel(); dm12 = sparse.diags(1 / np.sqrt(d))
    Hn = (dm12 @ H @ dm12).tocsr()
    v = np.ones(H.shape[0], complex); v /= np.linalg.norm(v)
    for _ in range(it):
        v = Hn @ v; v /= np.linalg.norm(v)
    return v


print(f"=== correctness through Operators.Eigensolver_invit (G=48, contrast={CONTRAST}) ===")
G = 48
for cycles in (0.0, 1.0, 2.0):
    H, phi = build_adj(G, cycles, CONTRAST)
    omega = op.Eigensolver_invit(H, eps=1e-4, steps=2).ravel()      # (nframes,1,1)->(nframes,)
    assert omega.shape == (G * G,) and np.allclose(np.abs(omega), 1.0), "bad drop-in shape/gauge"
    pw = power(H, 100)
    print(f"  cycles={cycles}: invit phase_err={phase_err_rms(omega, phi):.2e} "
          f"align={alignment(omega, phi):.3f}  |  power(100) align={alignment(pw, phi):.3f}")

print("\n=== CPU scaling of the sparse Laplacian solve (cycles=1) ===")
print(f"  {'frames':>8}{'invit ms':>10}{'phase_err':>12}{'align':>8}")
for G in (32, 64, 96, 128):
    H, phi = build_adj(G, 1.0, CONTRAST)
    t0 = time.perf_counter(); omega = op.Eigensolver_invit(H, eps=1e-4, steps=2).ravel()
    ms = (time.perf_counter() - t0) * 1e3
    print(f"  {G*G:>8}{ms:>10.1f}{phase_err_rms(omega, phi):>12.2e}{alignment(omega, phi):>8.3f}")

print("\nExpected: invit nails the phase at every cycles while power(100) plateaus at high")
print("cycles; solve time grows mildly with frame count (sparse direct factor+solve).")
