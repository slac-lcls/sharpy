"""
Dump H matrix from position_solve_coupled for comparison with MATLAB fit_shift.

Run this in Python, then run the equivalent MATLAB snippet below, and compare outputs.

MATLAB snippet to run after loading a matching synthetic example:
--------------------------------------------------------------------
nx=4; ny=4; nframes=3;
ixt=[0 3 0]; iyt=[0 0 3];   % scan positions
Nx=8; Ny=8;

probe_O = repmat(reshape(linspace(0.5,1.0,nx*ny),[nx,ny]),1,1) + 0j;
probe_x = probe_O * 0.1j;
probe_y = probe_O * 0.05;

% object pixel map
mapidx = reshape(1:Nx*Ny,[Nx Ny]);
mapidx_stack = zeros(nx,ny,nframes);
for k=1:nframes
  r = iyt(k)+1; c = ixt(k)+1;
  mapidx_stack(:,:,k) = mapidx(r:r+nx-1, c:c+ny-1);
end

% fake frames
z = zeros(nx,ny,nframes);
for k=1:nframes; z(:,:,k)=rand(nx,ny)+1j*rand(nx,ny); end

% QQinv
Qoverlap=@(f) reshape(accumarray(mapidx_stack(:),f(:),[Nx*Ny 1]),Nx,Ny);
norm2 = Qoverlap(abs(repmat(probe_O,[1,1,nframes])).^2);
QQinv = 1./(norm2+1e-10);

% Call fit_shift and inspect H:
% [you need fit_shift and framesmul4 in path]
% Pf = fit_shift(iyt, ixt, Nx, Ny, mapidx_stack);
% ... run Pfdotp_calc manually to extract H1, H2, Hx, H_sym
--------------------------------------------------------------------
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
config.GPU = False   # force CPU so we can inspect everything
import numpy as np

from Operators import map_frames, Splitc, Overlapc
from position_retrieval import (
    probe_derivatives, taylor_shift_probe, position_plan,
    _braket_coupled, _block_from_pairs,
)
import scipy.sparse as sp

# ---- tiny synthetic example ----
nx, ny = 4, 4
nframes = 3
Nx = Ny = 8
reg = 1e-10

# scan positions (integer, no sub-pixel shift)
tx = np.array([0, 3, 0], dtype=np.float64)
ty = np.array([0, 0, 3], dtype=np.float64)

# probe: smooth positive values (makes QQinv clearly non-uniform)
probe_O_flat = np.linspace(0.5, 1.0, nx * ny, dtype=np.complex128).reshape(nx, ny)
# fake derivatives
probe_x = probe_O_flat * 0.1j
probe_y = probe_O_flat * 0.05

# pack into a dp struct (no Taylor shift since xi=0)
class DotDict(dict):
    __getattr__ = dict.__getitem__

dp = DotDict(O=probe_O_flat, x=probe_x, y=probe_y,
             xx=np.zeros_like(probe_O_flat),
             xy=np.zeros_like(probe_O_flat),
             yy=np.zeros_like(probe_O_flat))

# xi=0 → taylor_shift returns dp unchanged
xi_x = np.zeros(nframes)
xi_y = np.zeros(nframes)

# dummy object
rng = np.random.default_rng(0)
obj = rng.standard_normal((Nx, Ny)) + 1j * rng.standard_normal((Nx, Ny))

# map frames
mapid = map_frames(tx, ty, nx, ny, Nx, Ny)

# frames: random (represents measured data after phase projection)
frames = rng.standard_normal((nframes, nx, ny)) + 1j * rng.standard_normal((nframes, nx, ny))

# probe stack (broadcast, xi=0)
probe_O_stack = np.broadcast_to(probe_O_flat[np.newaxis], (nframes, nx, ny))
probe_x_stack = np.broadcast_to(probe_x[np.newaxis], (nframes, nx, ny))
probe_y_stack = np.broadcast_to(probe_y[np.newaxis], (nframes, nx, ny))

QQinv = 1.0 / (Overlapc(np.abs(probe_O_stack) ** 2, Nx, Ny, mapid) + reg)
psi_img = QQinv * (Overlapc(frames * np.conj(probe_O_stack), Nx, Ny, mapid) + reg * obj)
psi = Splitc(psi_img, mapid)
QQinv_split = Splitc(QQinv, mapid)

zR1 = probe_x_stack * psi
zR2 = probe_y_stack * psi
zu  = frames - probe_O_stack * psi

def fsum(a): return np.sum(a, axis=(1, 2))

ccx = fsum(np.abs(zR1) ** 2).real
ccy = fsum(np.abs(zR2) ** 2).real
cxy = fsum(np.real(np.conj(zR1) * zR2))

rhs1 = fsum(np.real(np.conj(zu) * zR1))
rhs2 = fsum(np.real(np.conj(zu) * zR2))

plan = position_plan(tx, ty, nframes, nx, ny, Nx, Ny)
col, row = plan['col'], plan['row']
dx, dy, bw = plan['dx'], plan['dy'], plan['bw']

print(f"Pairs (col, row, dx, dy):")
for ii in range(len(col)):
    print(f"  ii={ii}: col={col[ii]}, row={row[ii]}, dx={dx[ii]}, dy={dy[ii]}")

ab11, ba11 = _braket_coupled(frames, probe_x_stack, probe_x_stack, QQinv_split, col, row, dx, dy, bw, backend='python')
ab22, ba22 = _braket_coupled(frames, probe_y_stack, probe_y_stack, QQinv_split, col, row, dx, dy, bw, backend='python')
abx,  bax  = _braket_coupled(frames, probe_x_stack, probe_y_stack, QQinv_split, col, row, dx, dy, bw, backend='python')

print(f"\nab11 = {ab11}")
print(f"ba11 = {ba11}")
print(f"real(ab11) = {ab11.real}")
print(f"real(ba11) = {ba11.real}")
print(f"ab11 == conj(ba11)? max diff = {np.max(np.abs(ab11 - np.conj(ba11))):.3e}")

print(f"\nOff-diagonal pairs where col != row:")
offd = col != row
for ii in np.where(offd)[0]:
    a, b = int(col[ii]), int(row[ii])
    print(f"  pair ({a},{b}): ab11={ab11[ii]:.4f+.4fj}, ba11={ba11[ii]:.4f+.4fj}")
    print(f"    real(ab11)={ab11[ii].real:.6f}, real(ba11)={ba11[ii].real:.6f}")
    print(f"    Python H11[{a},{b}] = {-(ab11[ii]+ba11[ii]).real:.6f}")
    print(f"    MATLAB H11[{a},{b}] would be = {-2*ba11[ii].real:.6f}  (using only ba)")
    print(f"    MATLAB H11[{a},{b}] would be = {-2*ab11[ii].real:.6f}  (using only ab)")

H1 = _block_from_pairs(-ab11, -ba11, col, row, ccx, nframes)
H2 = _block_from_pairs(-ab22, -ba22, col, row, ccy, nframes)
Hx = _block_from_pairs(-abx,  -bax,  col, row, cxy, nframes)

H1r = H1.real.toarray()
H2r = H2.real.toarray()
Hxr = Hx.real.toarray()

print(f"\n--- Python H1 (dense) ---")
print(np.array2string(H1r, precision=4, suppress_small=True))
print(f"\n--- Python H2 (dense) ---")
print(np.array2string(H2r, precision=4, suppress_small=True))
print(f"\n--- Python Hx (dense) ---")
print(np.array2string(Hxr, precision=4, suppress_small=True))
print(f"\nH1 symmetric? max|H1-H1.T| = {np.max(np.abs(H1r - H1r.T)):.3e}")
print(f"H2 symmetric? max|H2-H2.T| = {np.max(np.abs(H2r - H2r.T)):.3e}")
print(f"Hx symmetric? max|Hx-Hx.T| = {np.max(np.abs(Hxr - Hxr.T)):.3e}")

# Full system matrix
H = sp.bmat([[H1.real, Hx.real], [Hx.T.real, H2.real]], format='csr')
Hr = H.toarray()
print(f"\n--- Full H (6x6) ---")
print(np.array2string(Hr, precision=4, suppress_small=True))
print(f"\nFull H symmetric? max|H-H.T| = {np.max(np.abs(Hr - Hr.T)):.3e}")

eigvals = np.linalg.eigvalsh(Hr)
print(f"Eigenvalues of H: {eigvals}")
print(f"H positive semi-definite? min eigenvalue = {eigvals.min():.4e}")

print(f"\n--- RHS ---")
print(f"rhs1 = {2*rhs1}")
print(f"rhs2 = {2*rhs2}")
print(f"ccx  = {ccx}")
print(f"ccy  = {ccy}")
