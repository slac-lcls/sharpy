"""A/B validation + benchmark for the fused eps_S residual kernel (_diffnorm).

The change (Operators.py): ||frames - frames_old|| -- the eps_S frame-step
residual in the AP solvers -- computed by ONE dependency-free RawKernel pass
over the two complex64 buffers (interleaved-float view, shared-mem block
reduce + atomicAdd(double)), instead of  xp.linalg.norm(a - b)  which
materializes the (a-b) temporary and then runs cuBLAS nrm2. HISTORY: a
ReductionKernel version was 30x SLOWER (see the Operators.py comment above
_diffnorm) -- this A/B settles whether that was the tool (ReductionKernel
deopt on the complex .real()/.imag() map) or the idea (fusing the residual).

Validates fused + plain against a float64 CPU reference and checks every
fallback trigger, then benchmarks with CUDA-event timing, ALTERNATING
plain/fused within each rep (harness style of proxd_e2e_test.py: GPU
clock-ramp / pool-state bias hits both arms equally; median over reps).

CPU: only exercises the safe fallback == plain path (config.GPU=False).
GPU: any Slurm cluster with an A100-class card and a cupy env, e.g.
  sbatch -p <gpu-partition> -A <account> -N1 -n1 --gpus 1 -t 15 \
    --wrap 'source <venv>/bin/activate; cd <sharpy>/sharpy; \
            python -u diffnorm_fused_test.py'
  env: REPS (default 5), ITERS (default 200)
  The sweep runs to 8e8 complex64 (~19 GB peak incl. the plain path's
  temporary); oversized points self-skip against free device memory.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from statistics import median

import numpy as np
import config  # noqa: F401
import Operators as O

xp = O.xp
GPU = O.GPU

REPS = int(os.environ.get("REPS", 5))
ITERS = int(os.environ.get("ITERS", 200))


def make(n, seed=0, shape=None):
    """a, and b near a: representative of a converging AP step (small eps_S),
    which is where float32 cancellation in (a-b) would show up -- both arms
    subtract in float32, the fused arm then accumulates the squares in double.
    Generated on-device when GPU (multi-GB sizes would otherwise need the same
    footprint in host RAM plus a slow H2D copy)."""
    shape = (n,) if shape is None else shape
    if GPU:
        rs = xp.random.RandomState(seed)
        a = (rs.standard_normal(shape, dtype=xp.float32) +
             1j * rs.standard_normal(shape, dtype=xp.float32)).astype(xp.complex64)
        b = (a + (0.03 * (rs.standard_normal(shape, dtype=xp.float32) +
                          1j * rs.standard_normal(shape, dtype=xp.float32))
                  ).astype(xp.complex64)).astype(xp.complex64)
        return a, b
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal(shape) +
         1j * rng.standard_normal(shape)).astype(np.complex64)
    b = (a + 0.03 * (rng.standard_normal(shape) +
                     1j * rng.standard_normal(shape))).astype(np.complex64)
    return a, b


def _f64_ref(a, b):
    a = a.get() if GPU else a
    b = b.get() if GPU else b
    return float(np.linalg.norm(a.astype(np.complex128) - b.astype(np.complex128)))


def validate(n):
    a, b = make(n, seed=1)
    O._FUSED_PROXD = True
    fused = float(O._diffnorm(a, b))
    plain = float(xp.linalg.norm(a - b))
    ref = _f64_ref(a, b)
    ef, ep = abs(fused - ref) / ref, abs(plain - ref) / ref
    # fused (double accumulation) must be at least as accurate as plain; on
    # GPU expect ~1e-7 while plain (float32 accumulation) degrades with n.
    # On CPU fused IS plain (fallback), so ef == ep exactly.
    ok = ef <= max(1e-5, 2.0 * ep)
    print(f"  validate n={n:>10,}: fused relerr={ef:.2e}  plain relerr={ep:.2e}"
          f"  (ref {ref:.6g})  ok={ok}")
    # flag off -> exact historical path (same expression, bit-identical)
    O._FUSED_PROXD = False
    ok_off = float(O._diffnorm(a, b)) == plain
    O._FUSED_PROXD = True
    # off-spec inputs must fall back, not misfire: dtype / layout / shape
    c128 = abs(float(O._diffnorm(a.astype(xp.complex128), b.astype(xp.complex128)))
               - ref) <= 1e-9 * ref
    if a.ndim == 1 and n > 4:
        av, bv = a[::2], b[::2]  # non-contiguous view
        ok_nc = abs(float(O._diffnorm(av, bv)) -
                    float(xp.linalg.norm(av - bv))) <= 1e-6 * ref
    else:
        ok_nc = True
    print(f"  validate n={n:>10,}: flag-off==plain={ok_off}  "
          f"c128-fallback={c128}  noncontig-fallback={ok_nc}")
    return ok and ok_off and c128 and ok_nc


def time_events(fn, iters=ITERS, warmup=30):
    for _ in range(warmup):
        fn()
    xp.cuda.Device().synchronize()
    s, e = xp.cuda.Event(), xp.cuda.Event()
    s.record(); [fn() for _ in range(iters)]; e.record(); e.synchronize()
    return xp.cuda.get_elapsed_time(s, e) / iters


def _peak_bytes(fn):
    """Extra device memory the call needs beyond its inputs: plain materializes
    the full (a-b) temporary (8 B/elem), fused allocates nothing. Measured as
    the growth of the pool's total_bytes from a clean pool."""
    pool = xp.get_default_memory_pool()
    fn(); xp.cuda.Device().synchronize()      # let the pool reach steady state
    pool.free_all_blocks()
    before = pool.total_bytes()
    fn(); xp.cuda.Device().synchronize()
    return pool.total_bytes() - before


def bench(n, shape=None, label=""):
    a, b = make(n, shape=shape)
    O._FUSED_PROXD = True
    plain = lambda: xp.linalg.norm(a - b)   # noqa: E731  the historical path
    fused = lambda: O._diffnorm(a, b)       # noqa: E731
    ts = {"plain": [], "fused": []}
    for _ in range(REPS):                    # ALTERNATE arms within each rep
        for name, fn in (("plain", plain), ("fused", fused)):
            ts[name].append(time_events(fn))
    p, f = median(ts["plain"]), median(ts["fused"])
    # fused reads 16 B per complex elem (two c64 streams), writes ~nothing
    bw = 16 * a.size / (f * 1e-3) / 1e9
    mp, mf = _peak_bytes(plain), _peak_bytes(fused)
    print(f"  n={a.size:>12,}{label:<14}  plain={1e3*p:9.2f} us  "
          f"fused={1e3*f:9.2f} us  speedup={p/f:5.2f}x  BW ~{bw:4.0f} GB/s  "
          f"temp: plain={mp/1e9:6.3f} GB fused={mf/1e9:.3f} GB")
    # drop the buffers (the lambdas' closure cells are these same names) so the
    # next, larger size starts from a clean pool
    a = b = plain = fused = None
    xp.get_default_memory_pool().free_all_blocks()
    return p / f


if __name__ == "__main__":
    if GPU:
        import cupy as cp
        print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    print(f"GPU={GPU}  REPS={REPS}  ITERS={ITERS}")
    ok = validate(20_000) and validate(10_000_000)
    if not GPU:
        print("CPU: _diffnorm falls back to xp.linalg.norm. Run on the A100 for the A/B.")
        sys.exit(0 if ok else 1)
    print(f"\nmedian of {REPS} alternating reps, {ITERS} event-timed calls each")
    # Sweep well past 1e7: the repo's own memory_limit_harness pushes nx=256 with
    # hundreds of frames/side, so the interesting regime is 1e8+, where cuBLAS is
    # no longer latency-bound and the (a-b) temporary is GBs of headroom.
    free, total = xp.cuda.runtime.memGetInfo()
    print(f"  device memory: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
    sizes = [20_000, 100_000, 1_000_000, 10_000_000, 100_000_000,
             400_000_000, 800_000_000]
    for n in sizes:
        # peak need: two c64 inputs (16 B/elem) + plain's (a-b) temp (8 B/elem)
        if 24 * n > 0.80 * free:
            print(f"  n={n:>12,}  SKIPPED (needs ~{24*n/1e9:.1f} GB, "
                  f"{free/1e9:.1f} GB free)")
            continue
        bench(n)
    # the e2e harness shape: (K*K, nx, nx) frames from proxd_e2e_test defaults
    bench(None, shape=(576, 128, 128), label=" (576,128,128)")
    sys.exit(0 if ok else 1)
