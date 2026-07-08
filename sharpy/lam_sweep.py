"""Extended lambda sweep + autocorrelation conditioning diagnostic."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import numpy as np
import scipy.sparse.linalg as spla
from Operators import make_probe, map_frames, Splitc, Overlapc
from position_retrieval import (probe_derivatives, taylor_shift_probe,
    position_solve_coupled, position_solve_diag, position_plan, shift_rmse)


# ---------------------------------------------------------------------------
# Part 1: autocorrelation ratio vs condition number across r2 values
# ---------------------------------------------------------------------------

def probe_deriv_autocorr(r2, nx=32, step=4):
    """Normalized autocorrelation of W(k)=|kx|^2 |A(k)|^2 at lag=step in x.

    R_W(step) / R_W(0): close to 1 means off-diagonal terms ≈ diagonal (bad);
    close to 0 means good decorrelation (well-conditioned).
    """
    ki = np.fft.fftfreq(nx) * nx          # frequency bins, shape (nx,)
    KX, KY = np.meshgrid(ki, ki, indexing='ij')
    RR = np.sqrt(KX**2 + KY**2)
    r1 = max(r2 * 0.3, 0.01)              # keep r1/r2 ratio fixed at 0.3
    A = ((RR >= r1 * nx) & (RR <= r2 * nx)).astype(float)
    W = KX**2 * A
    if W.sum() == 0:
        return 0.0
    phase = np.exp(2j * np.pi * KX * step / nx)
    Rw_step = np.sum(W * phase).real
    Rw_0    = np.sum(W)
    return Rw_step / Rw_0


def build_scene(r2, nx=32, nnx=12, step=4, seed=2):
    """Build a ptychography scene for a given r2."""
    r1 = max(r2 * 0.3, 0.01)
    Nx = Ny = nnx * step
    rng = np.random.default_rng(seed)

    probe = make_probe(nx, nx, r1=r1, r2=r2)
    if isinstance(probe, tuple):
        probe = probe[0]
    probe = np.asarray(probe, dtype=np.complex128)
    pmax = np.abs(probe).max()
    if pmax < 1e-30:
        probe = np.ones((nx, nx), dtype=np.complex64)
    else:
        probe = (probe / pmax).astype(np.complex64)

    a = rng.standard_normal((Nx, Ny)) + 1j * rng.standard_normal((Nx, Ny))
    A = np.fft.fft2(a)
    fx, fy = np.meshgrid(np.fft.fftfreq(Nx), np.fft.fftfreq(Ny), indexing='ij')
    A *= np.exp(-(fx**2 + fy**2) / (2 * 0.10**2))
    truth = np.asarray(np.fft.ifft2(A).astype(np.complex64))

    tx, ty = np.meshgrid(np.arange(nnx)*step, np.arange(nnx)*step, indexing='ij')
    tx = tx.ravel().astype(np.float64)
    ty = ty.ravel().astype(np.float64)
    nframes = tx.size
    xi_x = rng.standard_normal(nframes) * 0.35
    xi_y = rng.standard_normal(nframes) * 0.35

    dp = probe_derivatives(probe)
    probe_shifted = taylor_shift_probe(dp, xi_x, xi_y)['O']
    mapid = map_frames(tx, ty, nx, nx, Nx, Ny)
    frames_clean = Splitc(truth, mapid) * probe_shifted
    plan = position_plan(tx, ty, nframes, nx, nx, Nx, Ny)
    return dict(probe=probe, dp=dp, truth=truth, mapid=mapid,
                frames_clean=frames_clean, plan=plan, xi_x=xi_x, xi_y=xi_y,
                nframes=nframes, Nx=Nx, Ny=Ny)


def hessian_condnum(scene):
    """Condition number of the preconditioned Hessian HH at xi=0."""
    from position_retrieval import _braket_coupled, _block_from_pairs
    import scipy.sparse as sp

    s = scene
    reg = 1e-10
    st = taylor_shift_probe(s['dp'], np.zeros(s['nframes']), np.zeros(s['nframes']))
    QQinv = 1.0 / (Overlapc(np.abs(st['O'])**2, s['Nx'], s['Ny'], s['mapid']) + reg)
    psi_img = QQinv * (Overlapc(s['frames_clean'] * np.conj(st['O']),
                                 s['Nx'], s['Ny'], s['mapid']) + reg * s['truth'])
    psi = Splitc(psi_img, s['mapid'])
    qq  = Splitc(QQinv,   s['mapid'])

    plan = s['plan']
    col, row, dx, dy, bw = plan['col'], plan['row'], plan['dx'], plan['dy'], plan['bw']
    nf = s['nframes']

    ccx = np.sum(np.abs(st['x'] * psi)**2, axis=(1,2)).real
    ccy = np.sum(np.abs(st['y'] * psi)**2, axis=(1,2)).real
    cxy = np.sum(np.real(np.conj(st['x'] * psi) * (st['y'] * psi)), axis=(1,2))

    ab11, ba11 = _braket_coupled(s['frames_clean'], st['x'], st['x'], qq, col, row, dx, dy, bw)
    ab22, ba22 = _braket_coupled(s['frames_clean'], st['y'], st['y'], qq, col, row, dx, dy, bw)
    abx,  bax  = _braket_coupled(s['frames_clean'], st['x'], st['y'], qq, col, row, dx, dy, bw)

    H1 = _block_from_pairs(-ab11, -ba11, col, row, ccx, nf)
    H2 = _block_from_pairs(-ab22, -ba22, col, row, ccy, nf)
    Hx = _block_from_pairs(-abx,  -bax,  col, row, cxy, nf)

    H1 = H1.real; H2 = H2.real; Hx = Hx.real
    H = sp.bmat([[H1, Hx], [Hx.T, H2]], format='csr')
    H = (H + H.T) * 0.5

    d = 1.0 / np.sqrt(np.concatenate([ccx, ccy]) + 1e-30)
    D = sp.diags(d)
    HH = (D @ H @ D).tocsr()
    HH = (HH + HH.T) * 0.5

    n = HH.shape[0]
    # smallest and largest eigenvalues via ARPACK
    try:
        lam_min = spla.eigsh(HH, k=1, which='SA', return_eigenvectors=False, tol=1e-6)[0]
        lam_max = spla.eigsh(HH, k=1, which='LA', return_eigenvectors=False, tol=1e-6)[0]
    except Exception:
        lam_min, lam_max = np.nan, np.nan
    if np.isnan(lam_min) or np.isnan(lam_max):
        return np.nan, np.nan, np.inf
    cond = lam_max / max(abs(lam_min), 1e-30) if lam_min > 0 else np.inf
    return lam_min, lam_max, cond


print("=" * 70)
print("Part 1: autocorrelation ratio vs Hessian conditioning across r2")
print("        (r1 = 0.3 * r2, nx=32, step=4, nnx=12)")
print("=" * 70)
print(f"{'r2':>6} | {'r2*step':>7} | {'R_W(s)/R_W(0)':>14} | {'lam_min':>10} | {'cond(HH)':>12} | {'coupled ok?':>11}")
print("-" * 72)

r2_values = [0.06, 0.10, 0.15, 0.20, 0.255, 0.30, 0.40]
for r2 in r2_values:
    ac  = probe_deriv_autocorr(r2, step=4)
    sc  = build_scene(r2)
    lmin, lmax, cond = hessian_condnum(sc)
    ok = "YES" if lmin > 0 else "NO (indef)"
    print(f"{r2:>6.3f} | {r2*4:>7.3f} | {ac:>14.3f} | {lmin:>10.3e} | {cond:>12.2e} | {ok:>11}")

print()

# ---------------------------------------------------------------------------
# Part 2: lambda sweep at fixed r2=0.255 (original experiment)
# ---------------------------------------------------------------------------

print("=" * 70)
print("Part 2: lambda sweep on r2=0.255 probe (60 iterations)")
print("=" * 70)

scene = build_scene(0.255)
nframes = scene['nframes']

hx = np.zeros(nframes); hy = np.zeros(nframes)
for _ in range(30):
    hx, hy = position_solve_diag(scene['frames_clean'], scene['dp'], scene['truth'],
                                  scene['mapid'], scene['Nx'], scene['Ny'],
                                  hx, hy, max_step=0.5)
print(f"diag (30 it) reference: {shift_rmse(scene['xi_x'], scene['xi_y'], hx, hy):.3e}\n")

print(f"{'lam':>10} | {'dr @15it':>10} | {'dr @30it':>10} | {'dr @60it':>10}")
print('-' * 48)
for lam in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
    hx = np.zeros(nframes); hy = np.zeros(nframes)
    cp = {}
    for i in range(60):
        hx, hy = position_solve_coupled(scene['frames_clean'], scene['dp'],
                                         scene['truth'], scene['mapid'],
                                         scene['Nx'], scene['Ny'],
                                         hx, hy, scene['plan'],
                                         max_step=0.5, lam=lam)
        if i + 1 in (15, 30, 60):
            cp[i+1] = shift_rmse(scene['xi_x'], scene['xi_y'], hx, hy)
    print(f"{lam:>10.1e} | {cp[15]:>10.3e} | {cp[30]:>10.3e} | {cp[60]:>10.3e}")
