"""Correctness check: _braket_coupled_cuda vs _braket_coupled_ref on a small scene."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# run with GPU=True
import config
assert config.GPU, "Set GPU=True in config.py before running this test"

import cupy as cp
import numpy as np
from Operators import make_probe, map_frames, Splitc, Overlapc
from position_retrieval import (
    probe_derivatives, taylor_shift_probe, position_plan,
    _braket_coupled_ref, _braket_coupled_cuda,
)

rng = np.random.default_rng(42)
nx = 32; nnx = 6; step = 4; Nx = Ny = nnx * step

probe, _ = make_probe(nx, nx, r1=0.03, r2=0.10)
probe = cp.asarray((probe / cp.abs(probe).max()).astype(cp.complex64))

a = rng.standard_normal((Nx,Ny)) + 1j*rng.standard_normal((Nx,Ny))
truth = cp.asarray(np.fft.ifft2(np.fft.fft2(a) *
    np.exp(-(np.add.outer(np.fft.fftfreq(Nx)**2,
                          np.fft.fftfreq(Ny)**2)) / (2*0.1**2))).astype(np.complex64))

tx, ty = np.meshgrid(np.arange(nnx)*step, np.arange(nnx)*step, indexing='ij')
tx = cp.asarray(tx.ravel().astype(np.float64))
ty = cp.asarray(ty.ravel().astype(np.float64))
nframes = int(tx.size)

xi_x = cp.asarray(rng.standard_normal(nframes) * 0.2)
xi_y = cp.asarray(rng.standard_normal(nframes) * 0.2)

dp = probe_derivatives(probe)
st = taylor_shift_probe(dp, xi_x, xi_y)
mapid = map_frames(tx, ty, nx, nx, Nx, Ny)
frames = Splitc(truth, mapid) * st['O']

reg = 1e-10
QQinv = 1.0 / (Overlapc(cp.abs(st['O'])**2, Nx, Ny, mapid) + reg)
qq = Splitc(QQinv, mapid)

plan = position_plan(tx, ty, nframes, nx, nx, Nx, Ny)
col, row, dx, dy, bw = plan['col'], plan['row'], plan['dx'], plan['dy'], plan['bw']

# GPU kernel result
ab_gpu, ba_gpu = _braket_coupled_cuda(frames, st['x'], st['y'], qq, col, row, dx, dy, bw)

# CPU reference result (bring arrays to CPU)
frames_cpu = cp.asnumpy(frames)
pL_cpu = cp.asnumpy(st['x'])
pR_cpu = cp.asnumpy(st['y'])
qq_cpu = cp.asnumpy(qq)
col_cpu = cp.asnumpy(col) if isinstance(col, cp.ndarray) else np.asarray(col)
row_cpu = cp.asnumpy(row) if isinstance(row, cp.ndarray) else np.asarray(row)
dx_cpu  = cp.asnumpy(dx)  if isinstance(dx,  cp.ndarray) else np.asarray(dx)
dy_cpu  = cp.asnumpy(dy)  if isinstance(dy,  cp.ndarray) else np.asarray(dy)

# temporarily switch xp to numpy for the ref
import position_retrieval as pr
_orig_xp = pr.xp
pr.xp = np
ab_ref, ba_ref = _braket_coupled_ref(frames_cpu, pL_cpu, pR_cpu, qq_cpu,
                                      col_cpu, row_cpu, dx_cpu, dy_cpu, bw)
pr.xp = _orig_xp

ab_gpu_np = cp.asnumpy(ab_gpu).astype(np.complex128)
ba_gpu_np = cp.asnumpy(ba_gpu).astype(np.complex128)

err_ab = np.abs(ab_gpu_np - ab_ref).max() / (np.abs(ab_ref).max() + 1e-30)
err_ba = np.abs(ba_gpu_np - ba_ref).max() / (np.abs(ba_ref).max() + 1e-30)

print(f"ab  max relative error: {err_ab:.2e}  {'PASS' if err_ab < 1e-3 else 'FAIL'}")
print(f"ba  max relative error: {err_ba:.2e}  {'PASS' if err_ba < 1e-3 else 'FAIL'}")
print(f"nnz={len(col_cpu)}, nframes={nframes}, bw={bw}")
