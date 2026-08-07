"""A/B validation + benchmark for the fused ProxD+residual kernel.

The change (Operators.py): SHARPY_FUSED_PROXD=1 routes Project_data's middle block
(the data residual + ProxD) through ONE RawKernel that computes  z <- ProxD(z)  AND
sum|z-ProxD(z)|^2  in a single in-place pass (zero temporaries), instead of the
two-pass  |z| + sqrt(data) + ProxD + norm.

This script isolates that middle block (the FFTs are identical either way) and:
  1. validates the fused kernel bit-close to the two-pass path (output + mse), and
  2. benchmarks both, with event timing AND a captured-CUDA-graph timing when
     available -- the fused path is alloc-free so it captures cleanly; the two-pass
     path allocates (abs/sqrt/norm/ProxD-out) so it may refuse capture (a second win).

CPU (Mac): only exercises the safe fallback == plain path.
GPU (Perlmutter):
  srun -A lcls_g -C gpu -q interactive -N1 -n1 --gpus 1 -t 10 \
    bash -c 'source ~/sharpy-venv/bin/activate; cd $SCRATCH/sharpy/sharpy; \
             python -u proxd_fused_test.py 256 256'
  (args: nframes nx ; default 256 256 -> ~16.8M complex64 elements)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config  # noqa: F401
import Operators as O

xp = O.xp
GPU = O.GPU
eps = O.eps


def make(nframes, nx, seed=0):
    rng = np.random.default_rng(seed)
    f = (rng.standard_normal((nframes, nx, nx)) +
         1j * rng.standard_normal((nframes, nx, nx))).astype(np.complex64)
    d = (rng.random((nframes, nx, nx)).astype(np.float32)) * 10.0
    if GPU:
        f = xp.asarray(f); d = xp.asarray(d)
    return f, d


def plain_middle(f, d, resid):
    """The current two-pass path on Fourier-space frames (out-of-place ProxD)."""
    mse = xp.linalg.norm(xp.abs(f) - xp.sqrt(d)) if resid else eps
    out = O.ProxD(f, d, eps)
    return out, mse


def validate_weighted(nframes, nx):
    """A/B the WEIGHTED fused kernel (proxd_resid_w) against the plain weighted
    path (ProxD w= + masked norm), incl. the W=0 'data can't leak' invariant."""
    f, d = make(nframes, nx, seed=1)
    rng = np.random.default_rng(2)
    W = (rng.random((nx, nx)) < 0.5).astype(np.float32)   # (N,N) mask, broadcast
    if GPU:
        W = xp.asarray(W)
    mp_ = xp.sqrt(W) * (xp.abs(f) - xp.sqrt(d))
    msep = xp.linalg.norm(mp_)
    outp = O.ProxD(f.copy(), d, eps, w=W)
    outf, msef = O._proxd_resid_apply(f.copy(), d, True, weights=W)
    ok_out = bool(xp.allclose(outp, outf, rtol=1e-4, atol=1e-5))
    ok_mse = bool(xp.allclose(msep, msef, rtol=1e-4))
    print(f"  validate wgt    : out match={ok_out}  mse match={ok_mse}  "
          f"(plain={float(msep):.6g}  fused={float(msef):.6g})")
    # masked-pixel data must be inert: perturb d at W=0, nothing may change
    d2 = d + 37.5 * (1.0 - W)
    outf2, msef2 = O._proxd_resid_apply(f.copy(), d2, True, weights=W)
    ok_inert = bool(xp.all(outf == outf2)) and float(msef) == float(msef2)
    # W=ones must reproduce the unweighted kernel bitwise
    ones = xp.ones((nx, nx), dtype=xp.float32)
    out1, mse1 = O._proxd_resid_apply(f.copy(), d, True, weights=ones)
    out0, mse0 = O._proxd_resid_apply(f.copy(), d, True)
    ok_ones = bool(xp.all(out0 == out1)) and float(mse0) == float(mse1)
    print(f"  validate wgt    : W=0 inert={ok_inert}  W=ones==None bitwise={ok_ones}")
    return ok_out and ok_mse and ok_inert and ok_ones


def validate(nframes, nx):
    f, d = make(nframes, nx)
    outp, msep = plain_middle(f.copy(), d, True)
    if GPU:
        outf, msef = O._proxd_resid_apply(f.copy(), d, True)
    else:  # CPU: _proxd_resid_apply intentionally falls back to plain
        outf, msef = O._proxd_resid_apply(f.copy(), d, True)
    ok_out = bool(xp.allclose(outp, outf, rtol=1e-4, atol=1e-5))
    ok_mse = bool(xp.allclose(msep, msef, rtol=1e-4))
    print(f"  validate resid : out match={ok_out}  mse match={ok_mse}  "
          f"(plain={float(msep):.6g}  fused={float(msef):.6g})")
    # compute_residuals=False: outputs still match, mse is the eps sentinel
    o0p, m0p = plain_middle(f.copy(), d, False)
    o0f, m0f = O._proxd_resid_apply(f.copy(), d, False)
    print(f"  validate no-res : out match={bool(xp.allclose(o0p, o0f, rtol=1e-4, atol=1e-5))}  "
          f"mse sentinel eq={float(m0p) == float(m0f)}")
    return ok_out and ok_mse


def time_events(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    xp.cuda.Device().synchronize()
    s, e = xp.cuda.Event(), xp.cuda.Event()
    s.record(); [fn() for _ in range(iters)]; e.record(); e.synchronize()
    return xp.cuda.get_elapsed_time(s, e) / iters


def time_graph(fn, iters=200, warmup=30):
    """Captured-graph timing; returns None if capture is refused (e.g. allocations)."""
    st = xp.cuda.Stream(non_blocking=True)
    try:
        with st:
            for _ in range(warmup):
                fn()
            st.synchronize()
            st.begin_capture()
            for _ in range(iters):
                fn()
            g = st.end_capture()
    except Exception as ex:
        print(f"    (graph capture refused: {type(ex).__name__}: {str(ex)[:60]})")
        return None
    g.launch(stream=st); st.synchronize()          # warm the graph
    s, e = xp.cuda.Event(), xp.cuda.Event()
    s.record(st); g.launch(stream=st); e.record(st); e.synchronize()
    return xp.cuda.get_elapsed_time(s, e) / 1.0     # one graph = `iters` launches


def bench(nframes, nx):
    f, d = make(nframes, nx)
    n = f.size
    fbuf = f.copy()          # fused is in-place; ProxD fixed point (|z|=sqrt d) => stable to re-run
    # plain path: ProxD is out-of-place, norm/abs/sqrt allocate -> the honest two-pass cost
    plain = lambda: plain_middle(f, d, True)         # noqa: E731  (f not mutated; ProxD out-of-place)
    fused = lambda: O._proxd_resid_apply(fbuf, d, True)  # noqa: E731

    pe = time_events(plain)
    fe = time_events(fused)
    print(f"  event  ms/call : plain={pe:.4f}  fused={fe:.4f}  speedup={pe/fe:.2f}x")
    # ~bytes moved: fused 20 B/elem (z r+w 16, data r 4); illustrative achieved BW
    print(f"  fused DRAM BW  : ~{20*n/(fe*1e-3)/1e9:.0f} GB/s (20 B/elem)")

    iters = 100
    pg = time_graph(plain, iters=iters)
    fg = time_graph(fused, iters=iters)
    if fg is not None:
        fg_pc = fg / iters
        if pg is not None:
            print(f"  graph  ms/call : plain={pg/iters:.4f}  fused={fg_pc:.4f}  speedup={(pg/iters)/fg_pc:.2f}x")
        else:
            print(f"  graph  ms/call : plain=CAPTURE-REFUSED (allocs)  fused={fg_pc:.4f}  "
                  f"(fused is graph-capturable; plain is not -> extra win in the AP loop)")


if __name__ == "__main__":
    nframes = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    nx = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    print(f"GPU={GPU}  size={nframes}x{nx}x{nx}  ({nframes*nx*nx/1e6:.1f}M elems)")
    okv = validate(nframes, nx)
    okw = validate_weighted(nframes, nx)
    okv = okv and okw
    if not GPU:
        print("CPU: fused path falls back to plain (no kernel). Run on Perlmutter for the A/B.")
        sys.exit(0 if okv else 1)
    bench(nframes, nx)
