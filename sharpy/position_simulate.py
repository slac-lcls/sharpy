"""
Simulate a ptychography dataset to HDF5 and load it back -- mimics the
experimental workflow (ptycho_simulate.py -> .h5 -> ptycho_reconstruct.py),
for the position-retrieval tests.

Two realism points over the inline test builders:
  * Realistic, weakly-absorbing object: transmission = exp(c*(-1+i*ph)*density),
    so |object| ~ 1 (Beer-Lambert), as in ptycho_simulate.py line 84 -- not a
    zero-mean random complex field.
  * Data goes through an .h5 file: only the measured intensity is stored, and
    the pixel-unit translations are recovered via the
    resolution = wavelength*detector_distance/(nx*detector_pixel_size)
    round-trip, exactly as ptycho_reconstruct.py does.

The per-frame position errors xi are baked into the data via the probe Taylor
shift; the true xi is stored too so recovery error can be measured.
"""

import os

import numpy as np
import h5py

import config
if config.GPU:
    import cupy as cp
    _xp = cp
    def _tonp(a): return cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a)
    def _toxp(a): return cp.asarray(a)
else:
    _xp = np
    def _tonp(a): return np.asarray(a)
    def _toxp(a): return np.asarray(a)

from Operators import make_probe, map_frames, Splitc, cropmat
from position_retrieval import shift_probe_fourier, taylor_shift_probe, apodize_probe, probe_derivatives

_HERE = os.path.dirname(os.path.abspath(__file__))


def transmission_object(Nx, Ny, contrast=0.69, phase_frac=0.5, gold_png=None):
    """Weakly-absorbing transmission object exp(c(-1+i*ph)*density), |.| <= 1.

    Same mapping as ptycho_simulate.py: gold-balls density in [0,1] -> a
    transmission ~1 with weak amplitude/phase contrast (the realistic regime).
    """
    path = gold_png or os.path.join(_HERE, "..", "data", "gold_balls.png")
    from PIL import Image
    density = np.array(Image.open(path), np.float32) / 63.0          # [0,1]
    obj = np.exp(contrast * (-1.0 + phase_frac * 1j) * density)      # ~1
    obj = np.asarray(obj, dtype=np.complex64)
    r0 = (obj.shape[0] - Nx) // 2
    r1 = (obj.shape[1] - Ny) // 2
    return obj[r0:r0 + Nx, r1:r1 + Ny]


def simulate_to_h5(fname, xi_x, xi_y, nx=32, nnx=16, step=3.5,
                   r1=0.075, r2=0.255, contrast=0.69, apodize=True,
                   wavelength=0.1, detector_pixel_size=100.0, resolution=1.0,
                   int_jitter_px=0, int_jitter_rng=None,
                   shift_model="fourier"):
    """Write a dataset (transmission object + baked-in position errors) to
    `fname` in the ptycho_simulate.py schema, plus the true xi.

    xi_x, xi_y      : (nframes,) unknown per-frame sub-pixel position errors to
                      bake in and later recover (nframes = nnx*nnx).
    contrast        : Beer-Lambert exponent; ~0.69 is realistic weak contrast
                      (|obj|~0.5-1), larger -> stronger contrast (easier position
                      retrieval). NOTE: position retrieval needs adequate contrast
                      AND small shift errors; weak contrast makes it much harder.
    int_jitter_px   : if > 0, add known integer noise uniform in
                      [-int_jitter_px, int_jitter_px] to tx, ty (baked into the
                      stored translations so the solver sees the correct integer
                      map, matching the paper's "known ±1 px perturbations").
    int_jitter_rng  : numpy RNG for the integer jitter (default: fixed seed 42).
    shift_model     : "fourier" (default) — exact band-limited shift via FFT phase
                      ramp; honest test since the solver inverts a Taylor model.
                      "taylor" — 2nd-order Taylor shift matching the MATLAB
                      Poverlap_branch2 convention; model-consistent with the solver
                      → machine-precision convergence, replicates the paper exactly.
    """
    ny = nx
    nny = nnx
    Nx = Ny = int(round(nnx * step))    # periodic image (full overlap coverage)

    truth = transmission_object(Nx, Ny, contrast=contrast)

    probe = make_probe(nx, ny, r1=r1, r2=r2)
    if isinstance(probe, tuple):
        probe = probe[0]
    probe = _toxp(probe)
    probe = (probe / _xp.abs(probe).max()).astype(_xp.complex64)
    if apodize:
        probe = apodize_probe(probe)

    # hexagonal periodic scan; integer base positions (sub-pixel goes in xi)
    ix1 = np.arange(nnx) * step
    iy1 = np.arange(nny) * step
    ix, iy = np.meshgrid(ix1, iy1, indexing="ij")
    ix = ix + np.floor(step / 2) * (np.arange(1, nnx + 1) % 2)[:, None]
    tx = np.round(ix).ravel().astype(np.float64)
    ty = np.round(iy).ravel().astype(np.float64)
    nframes = tx.size

    if int_jitter_px > 0:
        jrng = int_jitter_rng if int_jitter_rng is not None else np.random.default_rng(42)
        tx = tx + jrng.integers(-int_jitter_px, int_jitter_px + 1, nframes).astype(float)
        ty = ty + jrng.integers(-int_jitter_px, int_jitter_px + 1, nframes).astype(float)

    xi_x = np.asarray(xi_x, dtype=np.float64)
    xi_y = np.asarray(xi_y, dtype=np.float64)

    truth_xp = _toxp(truth)
    mapid = map_frames(_toxp(tx), _toxp(ty), nx, ny, Nx, Ny)
    if shift_model == "taylor":
        # Model-consistent with the solver (matches MATLAB Poverlap_branch2):
        # data generated with the same 2nd-order Taylor shift the solver inverts,
        # so the residual can reach machine precision at convergence.
        dp = probe_derivatives(probe)
        probe_shifted = taylor_shift_probe(dp, _toxp(xi_x), _toxp(xi_y))["O"]
    else:
        # Exact band-limited shift via FFT phase ramp — honest test since the
        # solver inverts only a Taylor/linearized model.
        probe_shifted = shift_probe_fourier(probe, _toxp(xi_x), _toxp(xi_y))
    frames = Splitc(truth_xp, mapid) * probe_shifted
    data = _tonp(_xp.abs(_xp.fft.fft2(frames)) ** 2)   # measured intensity

    detector_distance = nx * detector_pixel_size * resolution / wavelength
    translations = (tx + 1j * ty) * resolution        # physical-unit, like sim

    with h5py.File(fname, "w") as f:
        f["data"] = data.astype(np.float32)
        f["probe"] = _tonp(probe)
        f["truth"] = _tonp(truth_xp)
        f["translations"] = translations
        f["translations_x"] = tx
        f["translations_y"] = ty
        f["Nx"] = Nx
        f["Ny"] = Ny
        f["wavelength"] = wavelength
        f["detector_pixel_size"] = detector_pixel_size
        f["detector_distance"] = detector_distance
        f["xi_x_truth"] = xi_x
        f["xi_y_truth"] = xi_y
    return fname


def load_h5(fname):
    """Read a dataset back, recovering pixel-unit translations via the
    resolution round-trip (as ptycho_reconstruct.py does). Returns a dict."""
    with h5py.File(fname, "r") as f:
        data = np.array(f["data"], dtype=np.float32)
        probe = np.array(f["probe"], dtype=np.complex64)
        truth = np.array(f["truth"], dtype=np.complex64)
        translations = np.array(f["translations"])
        nframes, nx, ny = data.shape
        resolution = (float(f["wavelength"][()]) * float(f["detector_distance"][()])
                      / (nx * float(f["detector_pixel_size"][()])))
        translations = translations / resolution          # back to pixels
        tx = np.array(f["translations_x"], dtype=np.float64)
        ty = np.array(f["translations_y"], dtype=np.float64)
        Nx = int(f["Nx"][()])
        Ny = int(f["Ny"][()])
        xi_x_truth = np.array(f["xi_x_truth"], dtype=np.float64)
        xi_y_truth = np.array(f["xi_y_truth"], dtype=np.float64)

    return dict(
        frames_data=data, probe=probe, truth=truth,
        translations=translations, translations_x=tx, translations_y=ty,
        nx=nx, ny=ny, Nx=Nx, Ny=Ny, resolution=resolution,
        xi_x_truth=xi_x_truth, xi_y_truth=xi_y_truth,
    )


if __name__ == "__main__":
    import tempfile
    nf = 16 * 16
    rng = np.random.default_rng(0)
    xi = rng.standard_normal(nf) * 0.5, rng.standard_normal(nf) * 0.5
    fn = os.path.join(tempfile.gettempdir(), "pos_sim_demo.h5")
    simulate_to_h5(fn, xi[0], xi[1])
    p = load_h5(fn)
    print("wrote/read", fn)
    print("object |.| range: %.3f .. %.3f  (transmission ~1)"
          % (float(np.abs(p["truth"]).min()), float(np.abs(p["truth"]).max())))
    print("frames:", p["frames_data"].shape, "image:", (p["Nx"], p["Ny"]),
          "resolution:", p["resolution"])
