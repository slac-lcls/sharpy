"""Does Gramian batch-synchronization fix consensus across independently-solved
frame batches, and how far into photon noise does the ALIGNMENT survive?

Scenario (Stefano): keep the FOV, split frames into batches (= OD2P subdomains),
solve each batch with MANY inner iters (the streaming lever), each drifting into
its OWN global-phase gauge; then synchronize before combining. Test whether the
batch Gramian recovers the gauges, vs Poisson dose.

Setup: each batch reconstructs its subdomain from a RANDOM init phase (AP preserves
the object global phase = a free gauge even with known probe), so batches end in
different gauges. Combine two ways:
  naive  : average the batches directly (destructive across gauges)
  gramian: M x M band-Gramian  B_kl = <u_k, u_l> over shared (overlap) pixels ->
           top eigenvector = per-batch phases -> rotate to a common gauge, then average.
Reference = global single-domain AP at the SAME dose = the noise FLOOR (no gauge split).

Predictions:
  (a) ALIGNMENT robust: gramian ~= floor across doses; naive >> floor; alignment
      keeps working far below the ~100-300 ph/frame FRAME-sync threshold (M unknowns,
      each edge aggregates the whole band = many frame-pairs x pixels).
  (b) NOISE FLOOR unbeaten: both floor and gramian rise together at low dose -- sync
      fixes gauge consistency, adds no photons.

  python od2p_batchsync_test.py            # CPU quick
  srun ... python od2p_batchsync_test.py   # A100
  env: SUBTILE(m; 3 -> 9 batches) NIN(50) APIT(120) SEED(0)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import od2p_admm_scaffold as S

xp = S.xp
GPU = S.GPU
truth, probe, cprobe, mapid = S.truth, S.probe, S.cprobe, S.mapid
Nx, Ny, nx, nframes = S.Nx, S.Ny, S.nx, S.nframes
Splitc, Overlapc, Illuminate = S.Splitc, S.Overlapc, S.Illuminate_frames
Project_data, normalization, nmse = S.Project_data, S.normalization, S.nmse

# clean expected-intensity pattern (photons before scaling), per frame
clean = xp.abs(xp.fft.fft2(Splitc(truth, mapid) * probe[None])) ** 2
clean = clean / float(clean.mean())                       # mean 1 -> dose = mean photons/pixel


def make_data(dose, seed):
    xp.random.seed(seed)
    if dose >= 1e5:
        return (clean * dose).astype(xp.float32)          # ~noise-free
    return xp.random.poisson(clean * dose).astype(xp.float32)


def global_ap(data_n, iters):
    u = xp.ones((Ny, Nx), dtype=xp.complex64)
    for _ in range(iters):
        z = Illuminate(Splitc(u, mapid), probe)
        z, _ = Project_data(z, data_n)
        u = Overlapc(Illuminate(z, cprobe), Nx, Ny, mapid) / normalization
    return u


def batch_solve(k, data_n, mapk, norm_k, nin, init_phase):
    u = xp.ones((Ny, Nx), dtype=xp.complex64) * xp.exp(1j * xp.float32(init_phase))
    for _ in range(nin):
        z = Illuminate(Splitc(u, mapk[k]), probe)
        z, _ = Project_data(z, data_n[S.grp[k]])
        u = Overlapc(Illuminate(z, cprobe), Nx, Ny, mapk[k]) / norm_k[k]
    return u


def gramian_align(uk, supp, band):
    M = len(uk)
    V = [((supp[k] & band).astype(uk[k].dtype)) * uk[k] for k in range(M)]   # band-restricted
    B = xp.zeros((M, M), dtype=xp.complex128)
    for k in range(M):
        for l in range(M):
            B[k, l] = xp.vdot(V[k].ravel(), V[l].ravel())                    # <v_k,v_l>, Hermitian
    ph = xp.linalg.eigh(B)[1][:, -1]                                         # top eigvec: v_k = e^{-i th_k} p_k
    ph = ph / (xp.abs(ph) + 1e-30)
    return [uk[k] * ph[k] for k in range(M)]                                 # align: u_k * v_k (NOT conj)


def combine(uk, supp, cover, band, align):
    if align:
        uk = gramian_align(uk, supp, band)
    num = xp.zeros((Ny, Nx), dtype=xp.complex64)
    for k in range(len(uk)):
        num = num + supp[k].astype(uk[k].dtype) * uk[k]
    return num / xp.maximum(cover, 1).astype(xp.float32)


if __name__ == "__main__":
    m = int(os.environ.get("SUBTILE", 3))
    NIN = int(os.environ.get("NIN", 50))
    APIT = int(os.environ.get("APIT", 120))
    SEED = int(os.environ.get("SEED", 0))

    M, grp, mapk, supp, cover, band, norm_k = S.build_subdomains(m)
    S.grp = grp
    rng = np.random.default_rng(SEED)
    phases = rng.uniform(0, 2 * np.pi, M)                 # each batch's own gauge

    print(f"img {Nx}x{Ny}, {nframes} frames x {nx}, {M} batches ({m}x{m}), "
          f"band {int(band.sum())}/{Nx*Ny} px ({100*float(band.sum())/(Nx*Ny):.1f}%), NIN={NIN}")
    print(f"{'dose(ph/px)':>12} {'floor(AP)':>10} {'naive avg':>10} {'gramian':>10} {'gram/floor':>10}")
    for dose in [1e5, 100, 30, 10, 3, 1]:
        data_n = make_data(dose, SEED)
        floor = nmse(global_ap(data_n, APIT), truth)
        uk = [batch_solve(k, data_n, mapk, norm_k, NIN, phases[k]) for k in range(M)]
        naive = nmse(combine(uk, supp, cover, band, align=False), truth)
        gram = nmse(combine(uk, supp, cover, band, align=True), truth)
        tag = "clean" if dose >= 1e5 else f"{dose:g}"
        print(f"{tag:>12} {floor:>10.4f} {naive:>10.4f} {gram:>10.4f} {gram/max(floor,1e-9):>10.2f}")
    print("\n(a) alignment robust if gramian<<naive and gramian~floor down to low dose")
    print("(b) noise floor unbeaten if floor & gramian both rise together at low dose")
