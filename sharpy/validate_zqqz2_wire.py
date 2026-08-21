"""Validation gate for the zQQz2 coupled-GPU wiring (Yuan's e3ecb5e): _braket_coupled's CUDA path
(dotp2, complex64 + CUB reduce) vs the complex128 numba twin on identical seeded inputs, both
orientations, mixed-sign displacements, bw sweep. Also times cuda vs numba vs the old pure-python ref
(the 7.4 s/36-frame baseline that motivated the wiring). Run from a sharpy/sharpy dir on a GPU node."""
import time, numpy as np
import config; config.GPU = True
import cupy as cp
import position_retrieval as pr

rng = np.random.default_rng(7)

def make_case(g, nxf, step, bw):
    ty, tx = np.meshgrid(np.arange(g) * step, np.arange(g) * step, indexing="ij")
    tx = tx.ravel().astype(float); ty = ty.ravel().astype(float)
    nf = tx.size
    pl = pr.pair_list(tx, ty, nxf, bw=bw) if hasattr(pr, "pair_list") else None
    if pl is None:                                    # build pairs the way the solver does
        col, row, dx, dy = [], [], [], []
        for a in range(nf):
            for b in range(a + 1, nf):
                ddx = int(tx[b] - tx[a]); ddy = int(ty[b] - ty[a])
                if abs(ddx) < nxf - 2 * bw and abs(ddy) < nxf - 2 * bw:
                    col.append(a); row.append(b); dx.append(ddx); dy.append(ddy)
        col = np.array(col); row = np.array(row); dx = np.array(dx); dy = np.array(dy)
    else:
        col, row, dx, dy = pl["col"], pl["row"], pl["dx"], pl["dy"]
    frames = (rng.standard_normal((nf, nxf, nxf)) + 1j * rng.standard_normal((nf, nxf, nxf))).astype(np.complex64)
    pL = (rng.standard_normal((nf, nxf, nxf)) + 1j * rng.standard_normal((nf, nxf, nxf))).astype(np.complex64)
    pR = (rng.standard_normal((nf, nxf, nxf)) + 1j * rng.standard_normal((nf, nxf, nxf))).astype(np.complex64)
    qq = (rng.random((nf, nxf, nxf)) + 0.3).astype(np.complex64)
    return frames, pL, pR, qq, col, row, dx, dy

for (g, nxf, step, bw, label) in [(6, 64, 24, 0, "36f/64px bw0"),
                                  (6, 64, 24, 2, "36f/64px bw2"),
                                  (12, 128, 40, 3, "144f/128px bw3")]:
    frames, pL, pR, qq, col, row, dx, dy = make_case(g, nxf, step, bw)
    nnz = len(col)
    # CPU complex128 reference (numba twin, direct call)
    ab_ref = np.empty(nnz, np.complex128); ba_ref = np.empty(nnz, np.complex128)
    t0 = time.perf_counter()
    pr._braket_coupled_numba(frames.astype(np.complex128), pL.astype(np.complex128),
                             pR.astype(np.complex128), qq.astype(np.complex128),
                             col, row, dx, dy, bw, ab_ref, ba_ref)
    t_nb = time.perf_counter() - t0
    # GPU path through the real dispatch
    fg, plg, prg, qg = (cp.asarray(x) for x in (frames, pL, pR, qq))
    cg, rg = cp.asarray(col).astype(cp.uint64), cp.asarray(row).astype(cp.uint64)
    dxg, dyg = cp.asarray(dx).astype(cp.int64), cp.asarray(dy).astype(cp.int64)
    ab_g, ba_g = pr._braket_coupled(fg, plg, prg, qg, cg, rg, dxg, dyg, bw)
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        ab_g, ba_g = pr._braket_coupled(fg, plg, prg, qg, cg, rg, dxg, dyg, bw)
    cp.cuda.runtime.deviceSynchronize()
    t_gpu = (time.perf_counter() - t0) / 10
    ab_h, ba_h = cp.asnumpy(ab_g).ravel(), cp.asnumpy(ba_g).ravel()
    sc = np.abs(ab_ref).mean() + 1e-30
    rel_ab = float(np.max(np.abs(ab_h - ab_ref)) / sc)
    rel_ba = float(np.max(np.abs(ba_h - ba_ref)) / sc)
    print("CASE %-16s nnz %5d | rel_ab %.3g rel_ba %.3g %s | numba %.3f s -> cuda %.4f s (%.0fx)" % (
        label, nnz, rel_ab, rel_ba,
        "OK" if max(rel_ab, rel_ba) < 1e-4 else "FAIL",
        t_nb, t_gpu, t_nb / max(t_gpu, 1e-9)), flush=True)

# the old pure-python ref timing on the 36-frame case (the 7.4 s baseline)
frames, pL, pR, qq, col, row, dx, dy = make_case(6, 64, 24, 0)
t0 = time.perf_counter()
ab_p, ba_p = pr._braket_coupled_ref(frames, pL, pR, qq, col, row, dx, dy, 0)
t_py = time.perf_counter() - t0
print("python ref 36f/64px: %.2f s (the pre-wiring GPU fallback cost)" % t_py, flush=True)
print("ZQQZ2 WIRE VERDICT: see OK/FAIL above", flush=True)
print("DONE", flush=True)
