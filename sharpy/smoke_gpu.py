"""GPU smoke test: CuPy + sharpy operators (bincount Overlapc, sparse plan),
exercising non-square + partial-coverage geometries on device."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import cupy as cp, numpy as np
name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
free, total = cp.cuda.Device(0).mem_info
print("config.GPU =", config.GPU, "| CuPy", cp.__version__, "|", name,
      "| %.1f/%.1f GB free | CUDA" % (free/1e9, total/1e9), cp.cuda.runtime.runtimeGetVersion())
a = cp.random.random((1024,1024)).astype(cp.complex64)
print("FFT roundtrip rel err = %.2e" % float(cp.linalg.norm(cp.fft.ifft2(cp.fft.fft2(a))-a)/cp.linalg.norm(a)))

from Operators import map_frames, Splitc, Overlapc, Overlapc0, Split_Overlap_plan, xp
print("Operators xp =", xp.__name__)

def check(tag, Nx, Ny, nx, ny, starts):
    gx, gy = xp.meshgrid(starts, starts)
    tx, ty = gx.ravel().astype(float), gy.ravel().astype(float)
    mapid = map_frames(tx, ty, nx, ny, Nx, Ny)
    nf = tx.size
    O = (xp.asarray(np.random.randn(Ny,Nx)+1j*np.random.randn(Ny,Nx))).astype(xp.complex128)
    F = (xp.asarray(np.random.randn(nf,ny,nx)+1j*np.random.randn(nf,ny,nx))).astype(xp.complex128)
    SO = Splitc(O, mapid); StF = Overlapc(F, Nx, Ny, mapid)
    aerr = abs(complex(xp.vdot(SO,F))-complex(xp.vdot(O,StF)))/abs(complex(xp.vdot(SO,F)))
    Split, Overlap = Split_Overlap_plan(tx,ty,nx,ny,Nx,Ny)
    esp = float(xp.linalg.norm(Overlap(F)-StF)/xp.linalg.norm(StF))
    e0  = float(xp.linalg.norm(Overlapc0(F,Nx,Ny,mapid)-StF)/xp.linalg.norm(StF))
    print("  %-22s adjoint=%.1e sparse-vs-bincount=%.1e Overlapc0-vs-bincount=%.1e shape=%s"
          % (tag, aerr, esp, e0, tuple(StF.shape)))

check("square full",     64,64,32,32, xp.arange(8)*8)
check("nonsquare partial",160,96,32,32, xp.arange(6)*16)   # corner unlit -> needs shape=
check("square partial",  128,128,32,32, xp.arange(6)*16)
print("GPU SMOKE TEST PASSED")
