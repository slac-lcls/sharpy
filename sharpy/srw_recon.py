"""First sharpy-style reconstruction of Himanshu's SRW XPP ptychography data.

Blind AP (object + probe) on REAL imported frames, reusing sharpy's split/overlap GPU kernels.
Handles the three real-data conventions Project_data/refine_illumination_function don't:
  (1) frames are direct-beam-CENTERED -> ifftshift so DC sits at the corner (sharpy's
      data = |fft2(exit)|^2 convention);
  (2) DETECTOR MASK (module gaps): masked Fourier projection keeps the model magnitude where
      mask==0 instead of forcing it to sqrt(data);
  (3) NO truth probe -> ePIE-style blind object+probe updates, probe init by back-propagating
      the mean diffraction (the ZP donut -> a focused-spot guess).

Object update  O   = Overlap(psi'·conj(P)) / Overlap(|P|^2)
Probe update   P   = sum_k conj(O_k)·psi'_k / sum_k |O_k|^2     (O_k = Split(O) at pos k)
Optional Gramian sync every SE iters (off by default for this first validation).

Run on a GPU node: srun ... python -u srw_recon.py
env: NFRAMES CROP(512) REBIN(1) MAXIT(250) PU_WARMUP(15) SE(0=no sync) SYNC_EVERY(5)
     RUN(config.json) DET(at_detector dir)
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cupy as cp
import srw_import as imp
from wrap_ops import split_cuda, overlap_cuda

CROP = int(os.environ.get("CROP", 512)); REBIN = int(os.environ.get("REBIN", 1))
NF = int(os.environ["NFRAMES"]) if "NFRAMES" in os.environ else None
MAXIT = int(os.environ.get("MAXIT", 250)); PU_WARMUP = int(os.environ.get("PU_WARMUP", 15))
SE = int(os.environ.get("SE", 0)); SYNC_EVERY = int(os.environ.get("SYNC_EVERY", 5))
REG = 1e-3


def masked_project(exit_w, sdata, mask):
    """Fourier-magnitude projection with a detector mask (DC at corner). Keeps the model
    magnitude where mask==0. sdata = sqrt(data), both ifftshifted to DC-corner."""
    Z = cp.fft.fft2(exit_w)
    mag = cp.abs(Z)
    Znew = mask * (sdata * Z / (mag + 1e-12)) + (1 - mask) * Z
    return cp.fft.ifft2(Znew)


def data_misfit(exit_w, sdata, mask):
    mag = cp.abs(cp.fft.fft2(exit_w))
    return float(cp.linalg.norm((mag - sdata) * mask) / (cp.linalg.norm(sdata * mask) + 1e-12))


def main():
    R = os.environ.get("RUN", "/pscratch/sd/h/hgoel97/simulation_work/runs/"
                        "lcls_xpp_sim/base/9e3c6acc4678/config.json")
    DET = os.environ.get("DET", "/pscratch/sd/h/hgoel97/simulation_work/runs/apply_detector/"
                         "lcls_xpp_default_s1/da693a5b5ca9/stages/apply_detector/outputs/at_detector")
    fr, tx, ty, mask, meta = imp.import_run(R, DET, NF, CROP, REBIN)
    nf, nx, ny = fr.shape
    print("META:", {k: meta[k] for k in ("nframes", "dx_rec_nm", "N", "obj_px", "span_px")})
    # DC-to-corner (sharpy convention) + to device
    sdata = cp.asarray(np.fft.ifftshift(np.sqrt(np.maximum(fr, 0)), axes=(-2, -1)).astype(np.float32))
    dmask = cp.asarray(np.fft.ifftshift(mask, axes=(-2, -1)).astype(np.float32))[None]  # (1,nx,ny)
    tr = cp.asarray((tx + 1j * ty).astype(np.complex64))
    Nx = int(np.ceil(tx.max())) + nx + 2; Ny = int(np.ceil(ty.max())) + ny + 2
    ones_p = cp.ones((nx, ny), cp.complex64)
    # probe init: back-propagate the mean diffraction (donut -> focused spot)
    P = cp.fft.ifft2(cp.mean(sdata, axis=0).astype(cp.complex64))
    P /= cp.max(cp.abs(P))
    O = cp.ones((Ny, Nx), cp.complex64)
    Ok = cp.empty((nf, nx, ny), cp.complex64)
    exit_w = cp.empty((nf, nx, ny), cp.complex64)

    def normalization(Pc):
        n = cp.zeros((Ny, Nx), cp.complex64); overlap_cuda(n, 0, tr, Pc); return n

    normP = normalization(P)
    cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
    for it in range(MAXIT):
        split_cuda(O, exit_w, tr, P)                       # psi = Split(O)*P (model)
        rep = it % 25 == 0 or it == MAXIT - 1
        mf = data_misfit(exit_w, sdata, dmask) if rep else 0.0   # misfit on the MODEL (pre-proj!)
        exit_w = masked_project(exit_w, sdata, dmask)      # -> psi'
        if SE and it % SYNC_EVERY == 0 and it >= PU_WARMUP:
            om = sync_omega(exit_w, P, tr, Nx, Ny, nx)     # optional Gramian phase sync
            exit_w = exit_w * om
        Onum = cp.zeros((Ny, Nx), cp.complex64)
        overlap_cuda(Onum, exit_w, tr, P)                  # Overlap(psi'*conj(P))
        O = Onum / (normP + REG * float(cp.max(cp.abs(normP))))
        if it >= PU_WARMUP:                                # blind probe update
            split_cuda(O, Ok, tr, ones_p)                  # O_k = Split(O)
            Pnum = cp.sum(cp.conj(Ok) * exit_w, axis=0)
            Pden = cp.sum(cp.abs(Ok) ** 2, axis=0)
            P = Pnum / (Pden + REG * float(cp.max(Pden)))
            P /= cp.max(cp.abs(P)); normP = normalization(P)
        if rep:
            print(f"  it {it:4d}  data-misfit {mf:.4f}")
    cp.cuda.Stream.null.synchronize()
    print(f"recon {MAXIT} iters in {time.perf_counter()-t0:.1f}s  (sync={'on' if SE else 'off'})")

    Oc = cp.asnumpy(O); Pc = cp.asnumpy(P)
    np.savez("srw_recon.npz", obj=Oc, probe=Pc, tx=tx, ty=ty, meta=meta)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
        a = np.abs(Oc); ph = np.angle(Oc)
        # crop to the covered region for display
        x0, x1 = int(tx.min()), int(tx.max()) + nx; y0, y1 = int(ty.min()), int(ty.max()) + ny
        ax[0].imshow(a[y0:y1, x0:x1], cmap="gray"); ax[0].set_title("object |O| (Cu bright/Si dark)")
        ax[1].imshow(ph[y0:y1, x0:x1], cmap="twilight"); ax[1].set_title("object phase")
        ax[2].imshow(np.abs(np.fft.fftshift(Pc)), cmap="magma"); ax[2].set_title("probe |P| (focus)")
        ax[3].imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(Pc))) + 1), cmap="magma")
        ax[3].set_title("probe far-field (donut check)")
        plt.tight_layout(); plt.savefig("srw_recon.png", dpi=95)
        print("saved srw_recon.png + srw_recon.npz")
    except Exception as e:
        print("plot skipped:", str(e)[:80])


def sync_omega(exit_w, P, tr, Nx, Ny, nx):
    """Gramian phase sync (opt-in) reusing the production plan + eigensolver."""
    from Operators import Gramiam_plan, Gramiam_calc_cuda, Eigensolver, Precondition_calc, Splitc, xp
    global _GP
    if "_GP" not in globals():
        _GP = Gramiam_plan(tr.real, tr.imag, exit_w.shape[0], nx, nx, Nx, Ny, bw=0)
    inorm = cp.ones_like(exit_w)                            # unit inorm (sync only needs relative phase)
    fn = Precondition_calc(exit_w, bw=0)
    H = Gramiam_calc_cuda(exit_w, _GP, P, inorm, fn)
    return Eigensolver(H, 10).reshape(exit_w.shape[0], 1, 1)


if __name__ == "__main__":
    main()
