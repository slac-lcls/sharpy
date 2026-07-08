"""Detailed per-step timing breakdown of position_solve_coupled at a given nnx."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import numpy as np
if config.GPU:
    import cupy as cp
    xp = cp
    def sync():
        cp.cuda.Stream.null.synchronize()
else:
    xp = np
    def sync():
        pass

from Operators import make_probe, map_frames, Splitc, Overlapc
from position_retrieval import (
    probe_derivatives, taylor_shift_probe, shift_probe_fourier,
    position_plan, _braket_coupled, _block_from_pairs, _sparse, _splinalg,
)

nnx   = 48
step  = 4
nx    = 32
r2    = 0.10
r1    = r2 * 0.3
Nx    = Ny = nnx * step
NITER = 5   # average over this many iters (skip first for JIT warmup)

print(f"GPU={config.GPU}  nnx={nnx}  nframes={nnx**2}  Nx={Nx}  nx={nx}")
print(f"Sparse system size: {2*nnx**2} x {2*nnx**2}\n")
sys.stdout.flush()

# --- setup ---
probe_np, _ = make_probe(nx, nx, r1=r1, r2=r2)
if config.GPU: probe_np = probe_np.get()
probe_np = (np.asarray(probe_np) / np.abs(np.asarray(probe_np)).max()).astype(np.complex64)
probe = xp.asarray(probe_np)

rng = np.random.default_rng(7)
a = rng.standard_normal((Nx,Ny)) + 1j*rng.standard_normal((Nx,Ny))
A = np.fft.fft2(a)
fx,fy = np.meshgrid(np.fft.fftfreq(Nx), np.fft.fftfreq(Ny), indexing='ij')
A *= np.exp(-(fx**2+fy**2)/(2*0.10**2))
truth = xp.asarray(np.fft.ifft2(A).astype(np.complex64))

tx_np, ty_np = np.meshgrid(np.arange(nnx)*step, np.arange(nnx)*step, indexing='ij')
tx_np = tx_np.ravel().astype(np.float64)
ty_np = ty_np.ravel().astype(np.float64)
tx = xp.asarray(tx_np); ty = xp.asarray(ty_np)
nframes = len(tx_np)

max_abs = 0.10 * (nnx-1) / 2.0
gx = np.tile(np.arange(nnx), nnx).astype(np.float64); gx -= gx.mean()
xi_x = xp.asarray(gx / np.abs(gx).max() * max_abs)
xi_y = xp.zeros(nframes)

dp = probe_derivatives(probe)
probe_shifted = shift_probe_fourier(probe, xi_x, xi_y)
mapid = map_frames(tx, ty, nx, nx, Nx, Ny)
frames = Splitc(truth, mapid) * probe_shifted
plan = position_plan(tx, ty, nframes, nx, nx, Nx, Ny)
col, row, dx, dy, bw = plan['col'], plan['row'], plan['dx'], plan['dy'], plan['bw']
print(f"nnz={len(col)}  (frame pairs in sparse system)\n")
sys.stdout.flush()

hx = xi_x * 0.5   # start near solution so solver is representative
hy = xi_y * 0.5

# accumulate timings
t_probe   = 0.0   # step 0: taylor_shift_probe + QQinv + psi
t_diag    = 0.0   # step 0b: diagonal terms (ccx, ccy, rhs)
t_braket  = 0.0   # step 1: _braket_coupled x3
t_assemble = 0.0  # step 2: _block_from_pairs, bmat, Jacobi precond
t_solve   = 0.0   # step 3: spsolve
t_total   = 0.0

for it in range(NITER + 1):  # iter 0 = warmup
    t0_total = time.time()
    reg = 1e-10

    # --- step 0: probe eval + QQinv + psi ---
    sync(); t0 = time.time()
    st = taylor_shift_probe(dp, hx, hy)
    probe_O, probe_x, probe_y = st['O'], st['x'], st['y']
    QQinv = 1.0 / (Overlapc(xp.abs(probe_O)**2, Nx, Ny, mapid) + reg)
    psi_img = QQinv * (Overlapc(frames * xp.conj(probe_O), Nx, Ny, mapid) + reg*truth)
    psi = Splitc(psi_img, mapid)
    QQinv_split = Splitc(QQinv, mapid)
    zR1 = probe_x * psi; zR2 = probe_y * psi; zu = frames - probe_O * psi
    sync(); dt_probe = time.time() - t0

    # --- step 0b: diagonal terms ---
    sync(); t0 = time.time()
    def fsum(a): return xp.sum(a, axis=(1,2))
    ccx  = fsum(xp.abs(zR1)**2).real
    ccy  = fsum(xp.abs(zR2)**2).real
    cxy  = fsum(xp.real(xp.conj(zR1)*zR2))
    rhs1 = fsum(xp.real(xp.conj(zu)*zR1))
    rhs2 = fsum(xp.real(xp.conj(zu)*zR2))
    sync(); dt_diag = time.time() - t0

    # --- step 1: braket x3 ---
    sync(); t0 = time.time()
    ab11, ba11 = _braket_coupled(frames, probe_x, probe_x, QQinv_split, col, row, dx, dy, bw)
    ab22, ba22 = _braket_coupled(frames, probe_y, probe_y, QQinv_split, col, row, dx, dy, bw)
    abx,  bax  = _braket_coupled(frames, probe_x, probe_y, QQinv_split, col, row, dx, dy, bw)
    sync(); dt_braket = time.time() - t0

    # --- step 2: assemble sparse system ---
    sync(); t0 = time.time()
    H1 = _block_from_pairs(-ab11, -ba11, col, row, ccx, nframes).real
    H2 = _block_from_pairs(-ab22, -ba22, col, row, ccy, nframes).real
    Hx = _block_from_pairs(-abx,  -bax,  col, row, cxy, nframes).real
    H  = _sparse.bmat([[H1, Hx], [Hx.T, H2]], format='csr')
    rhs = xp.concatenate([2.0*rhs1, 2.0*rhs2])
    d   = 1.0 / xp.sqrt(xp.concatenate([ccx, ccy]) + 1e-30)
    D   = _sparse.diags(d)
    HH  = (D @ H @ D).tocsr()
    HH  = (HH + HH.T) * 0.5
    sync(); dt_assemble = time.time() - t0

    # --- step 3: sparse solve (CG on GPU, spsolve on CPU) ---
    sync(); t0 = time.time()
    if config.GPU:
        y, info = _splinalg.cg(HH, D @ rhs, maxiter=500)
    else:
        y = _splinalg.spsolve(HH, D @ rhs)
    sync(); dt_solve = time.time() - t0

    dt_total = time.time() - t0_total

    label = "WARMUP" if it == 0 else f"iter {it}"
    print(f"{label:>8}: probe={dt_probe:.3f}s  diag={dt_diag:.3f}s  "
          f"braket={dt_braket:.3f}s  assemble={dt_assemble:.3f}s  "
          f"solve={dt_solve:.3f}s  total={dt_total:.3f}s")
    sys.stdout.flush()

    if it > 0:
        t_probe    += dt_probe
        t_diag     += dt_diag
        t_braket   += dt_braket
        t_assemble += dt_assemble
        t_solve    += dt_solve
        t_total    += dt_total

N = NITER
print(f"\n{'='*60}")
print(f"Average over {N} iters:")
print(f"  0. probe+QQinv+psi : {t_probe/N:.3f}s  ({100*t_probe/t_total:.1f}%)")
print(f"  0b. diag terms      : {t_diag/N:.3f}s  ({100*t_diag/t_total:.1f}%)")
print(f"  1. braket x3 (GPU)  : {t_braket/N:.3f}s  ({100*t_braket/t_total:.1f}%)")
print(f"  2. assemble (CPU)   : {t_assemble/N:.3f}s  ({100*t_assemble/t_total:.1f}%)")
print(f"  3. spsolve (CPU)    : {t_solve/N:.3f}s  ({100*t_solve/t_total:.1f}%)")
print(f"  TOTAL               : {t_total/N:.3f}s/iter")
