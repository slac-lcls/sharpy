"""OD2P scaffold -- Overlapping Domain Decomposition + ADMM for ptychography
(arXiv:2011.00162, Chang-Glowinski-Marchesini-Tai-Wang-Zeng) on sharpy operators.

The IMAGE is split into overlapping subdomains (by SCAN tile); each subdomain runs
a local ptycho solve on ITS frames, coupled to a consensus object on the overlap
BANDS via ADMM. The global objective = sum of per-subdomain fidelities + equality
on shared pixels; ADMM drives a stationary point of it. Ptycho is NONCONVEX, so
OD2P and plain global AP reach DIFFERENT (both valid) fixed points -- validate on
NMSE-vs-truth (OD2P should match or BEAT global AP, per the paper's robustness),
NOT on OD2P==AP. One framework -> single-GPU streaming | MPI | batch (see the
TODO(human) deployment hooks at the bottom).

No new operators: Split_k / Overlap_k = Splitc / Overlapc with `mapid` sliced by
the subdomain's frame indices. Only new structures: subdomain groups, overlap-band
mask, per-subdomain normalization, consensus `w` + per-subdomain dual `lam`.

  python od2p_admm_scaffold.py         (CPU: validates OD2P w ~= global AP object)
  srun ... python od2p_admm_scaffold.py
  env: NX(96) KG(20) SUBTILE(m, m*m subdomains; 2) NOUT(40) NIN(3) RHO(0.5) MAXITER(120)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config
from Operators import (map_frames, Splitc, Overlapc, Illuminate_frames,
                       Project_data, xp)

GPU = config.GPU
eps = xp.float32(1e-8)


def phantom(Nx, Ny, contrast=1.5, seed=0):
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.standard_normal((Nx, Ny)))
    k = np.fft.fftfreq(Nx); KX, KY = np.meshgrid(k, k)
    sm = np.real(np.fft.ifft2(F * np.exp(-(KX ** 2 + KY ** 2) / (2 * 0.03 ** 2))))
    sm = (sm - sm.min()) / (sm.max() - sm.min())
    return ((0.5 + 0.5 * sm) * np.exp(1j * contrast * (sm - 0.5))).astype(np.complex64)


# ---- geometry / data ----
nx = ny = int(os.environ.get("NX", 96))
K = int(os.environ.get("KG", 20))
step = nx // 4
Nx = Ny = (K - 1) * step + nx
g = xp.arange(K) * step
tx, ty = xp.meshgrid(g, g, indexing="ij")
tx = tx.ravel().astype(float); ty = ty.ravel().astype(float)
nframes = tx.size

truth = xp.asarray(phantom(Nx, Ny)).astype(xp.complex64)
c = nx // 2
X, Y = xp.meshgrid(xp.arange(nx) - c, xp.arange(nx) - c)
probe = xp.exp(-(X ** 2 + Y ** 2) / (2.0 * (0.18 * nx) ** 2)).astype(xp.complex64)
probe = (probe / xp.abs(probe).max()).astype(xp.complex64)
cprobe = xp.conj(probe)
mapid = map_frames(tx, ty, nx, ny, Nx, Ny)
data = (xp.abs(xp.fft.fft2(Splitc(truth, mapid) * probe[None])) ** 2).astype(xp.float32)


def _norm(frame_idx, mk):
    absP2 = xp.broadcast_to(xp.abs(probe) ** 2, (frame_idx.size, nx, ny)).astype(xp.complex64)
    n = Overlapc(absP2, Nx, Ny, mk)
    return xp.where(xp.abs(n) < 1e-6 * float(xp.max(xp.abs(n))), xp.complex64(1), n)


normalization = _norm(xp.arange(nframes), mapid)             # global (for the AP reference)


# ---- global AP reference (what OD2P must converge to) ----
def ap_ref(maxiter):
    img = xp.ones((Ny, Nx), dtype=xp.complex64)
    for _ in range(maxiter):
        z = Illuminate_frames(Splitc(img, mapid), probe)
        z, _ = Project_data(z, data)
        img = Overlapc(Illuminate_frames(z, cprobe), Nx, Ny, mapid) / normalization
    return img


# ---- subdomains (2D SCAN tiling) + overlap band ----
def assign(a, m):
    r = (a - a.min()) / (a.max() - a.min() + 1e-9)
    return xp.clip((r * m).astype(int), 0, m - 1)


def build_subdomains(m):
    sub = assign(tx, m) * m + assign(ty, m)                  # 0..m*m-1
    M = m * m
    grp = [xp.where(sub == k)[0] for k in range(M)]
    mapk = [mapid[grp[k]] for k in range(M)]
    supp, cover = [], xp.zeros((Ny, Nx), int)
    for k in range(M):
        s = xp.zeros(Nx * Ny, bool); s[xp.unique(mapk[k])] = True
        s = s.reshape(Ny, Nx); supp.append(s); cover += s.astype(int)
    band = (cover >= 2)                                      # M_B: shared pixels (the halo)
    norm_k = [_norm(grp[k], mapk[k]) for k in range(M)]
    return M, grp, mapk, supp, cover, band, norm_k


def local_solve(k, w, lam, mapk, norm_k, band, n_inner, rho):
    """A few AP iters on subdomain k's frames, pulled toward consensus on the band."""
    u = w + 0.0
    rb = rho * band
    for _ in range(n_inner):
        z = Illuminate_frames(Splitc(u, mapk[k]), probe)     # only subdomain-k frames resident
        z, _ = Project_data(z, data[grp[k]])                 # AGM data prox (finite-tau -> ProxD_noise)
        acc = Overlapc(Illuminate_frames(z, cprobe), Nx, Ny, mapk[k])
        u = (acc + rb * (w - lam[k])) / (norm_k[k] + rb + eps)   # = sharpy divide + band Tikhonov
    return u


def od2p(m, n_outer, n_inner, rho):
    M, grp_, mapk, supp, cover, band, norm_k = build_subdomains(m)
    global grp; grp = grp_
    w = xp.ones((Ny, Nx), dtype=xp.complex64)
    lam = [xp.zeros((Ny, Nx), dtype=xp.complex64) for _ in range(M)]
    covf = xp.maximum(cover, 1).astype(xp.float32)
    for _ in range(n_outer):
        uk = [local_solve(k, w, lam, mapk, norm_k, band, n_inner, rho) for k in range(M)]
        num = xp.zeros((Ny, Nx), dtype=xp.complex64)
        for k in range(M):
            num = num + supp[k] * (uk[k] + lam[k])           # consensus: sum over covering subdomains
        w = xp.where(band, num / covf, num)                  # band: average; interior: single owner
        for k in range(M):
            lam[k] = lam[k] + band * (uk[k] - w)             # dual (band only)
    return w, M, int(band.sum())


def nmse(a, b):
    s = xp.vdot(a, b) / (xp.vdot(a, a) + 1e-30)
    return float(xp.linalg.norm(s * a - b) / xp.linalg.norm(b))


if __name__ == "__main__":
    m = int(os.environ.get("SUBTILE", 2))
    NOUT = int(os.environ.get("NOUT", 40)); NIN = int(os.environ.get("NIN", 3))
    RHO = float(os.environ.get("RHO", 0.5))
    MAXITER = int(os.environ.get("MAXITER", 120))

    ref = ap_ref(MAXITER)                                    # global AP object
    w, M, nband = od2p(m, NOUT, NIN, RHO)
    print(f"img {Nx}x{Ny}, frames {nframes} x {nx}, subdomains {M} ({m}x{m}), "
          f"band {nband}/{Nx*Ny} px ({100*nband/(Nx*Ny):.1f}%)")
    print(f"global AP ({MAXITER} it)      NMSE vs truth : {nmse(ref, truth):.4e}")
    print(f"OD2P ({NOUT}x{NIN}, rho={RHO})   NMSE vs truth : {nmse(w, truth):.4e}  <- match/beat AP")
    print(f"OD2P vs global AP object (nonconvex: differ, both valid): {nmse(w, ref):.4e}")
    print("OD2P scaffold OK" if nmse(w, truth) < 1.3 * nmse(ref, truth) else "CHECK rho/n_outer")

# ===========================================================================
# DEPLOYMENT hooks (this loop is the algorithm; these turn it into a system):
#
# TODO(human): SINGLE-GPU STREAMING -- run the `for k` in od2p() SEQUENTIALLY;
#   keep only {w, lam (thin bands)} + subdomain-k {u, data[grp[k]], frames} on GPU.
#   Double-buffer: async-prefetch data[grp[k+1]] on a 2nd stream during subdomain
#   k's n_inner iters (n_inner amortizes the PCIe copy -> hides the bus).
#
# TODO(human, +Yuan): MPI -- subdomain k -> rank k; the consensus `num`/`w` step
#   becomes a HALO EXCHANGE of band pixels (uk+lam) with neighbour ranks (small,
#   NVLink), not a whole-object allreduce. See sharpy_mpi_porting_notes.md.
#
# TODO(human): BLIND PROBE -- probe is global; add a probe consensus (average the
#   per-subdomain probe updates) or keep the probe global and decompose only object.
#
# TODO(human): SYNC -- run the global Gramian phase-sync between outer ADMM steps
#   first (cheap/sparse via KD-tree); later decompose it (per-subdomain power
#   iteration + band consensus on the phases).
# ===========================================================================
