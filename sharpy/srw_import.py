"""Import Himanshu Goel's SRW XPP ptychography sim output -> sharpy reconstruction inputs.

Reads the detector-plane diffraction frames (at_detector/intensity_N.h5, dataset 'intensity',
flat (ny*nx,) with x fastest) + mask.npy, and reproduces the spiral scan positions from the
run's config.json via a concentric-ring spiral scan matching the sim (deterministic, no RNG). Centers
each frame on the DIRECT BEAM (from the SRW mesh xStart/yStart, JungFrau is off-center), and
converts positions (m) -> object-grid pixels via the Fraunhofer real-space pixel
dx = lambda*z/(N*det_px). Far-field is valid here (Fresnel number ~0.01). Returns
(frames_data, translations_x, translations_y, mask, meta) ready for sharpy.

Geometry (SLAC UserMeeting'25 deck + h5 meshes): 8.8 keV (lambda 1.409 A), det_px 75 um,
sample->det 4.147 m, frame 1030(x) x 1064(y), Poisson-only noise. Object = Holler 2019 IC
nanolaminography (Zenodo 2657340), Cu bright / Si dark, low absorption.

This is OUR reader for his output layout -- NOT Himanshu's sim code (that stays private).
Run the self-check on Perlmutter: python -u srw_import.py   (saves srw_import_preview.png)
env: RUN(base cfg dir) DET(at_detector dir) NFRAMES CROP(512) REBIN(1)
"""
import os
import re
import json
import glob
import math
import numpy as np
import h5py

HC = 12398.419843320026e-10       # eV*m
ENERGY = 8800.0                   # eV
LAMBDA = HC / ENERGY              # ~1.409e-10 m
DET_PX = 75e-6                    # m (JungFrau)
Z_SD = 4.147                      # m (XPP: Sample 229.999042 -> Detector 234.146042)
SAMPLE_RES = 6.5e-9              # m (spiral pixelSize; result is pixelSize-independent)
DEF_SCAN = (5e-6, 5e-6, 0.234e-6)


def spiral_pattern(width, height, step, pixel_size):
    """Concentric-ring scan (rings at radius i*step, ~step arc spacing, clipped to the
    width x height field) reproducing the SRW sim's scan pattern -- deterministic, so shot N
    <-> position N; pixel_size only sets the ring count. Returns [(x, y), ...] in metres."""
    rings = max(math.ceil(math.ceil(width / pixel_size) / math.ceil(step / pixel_size)),
                math.ceil(math.ceil(height / pixel_size) / math.ceil(step / pixel_size)))
    pts = []
    for i in range(rings):
        r = i * step
        if r == 0:
            pts.append((0.0, 0.0)); continue
        m = int(2 * math.pi * r / step)
        for j in range(m):
            x, y = r * math.cos(2 * math.pi * j / m), r * math.sin(2 * math.pi * j / m)
            if -width / 2 <= x <= width / 2 and -height / 2 <= y <= height / 2:
                pts.append((x, y))
    return pts


def _idx(p):
    return int(re.search(r'intensity_(\d+)\.h5', os.path.basename(p)).group(1))


def _load_frame(p):
    with h5py.File(p, 'r') as f:
        a = dict(f.attrs); nx, ny = int(a['nx']), int(a['ny'])
        d = f['intensity'][:].reshape(ny, nx)            # SRW: x fastest -> [y, x]
        # direct-beam pixel from the mesh (x=0,y=0), JungFrau off-center
        cx = -a['xStart'] / ((a['xFin'] - a['xStart']) / nx)
        cy = -a['yStart'] / ((a['yFin'] - a['yStart']) / ny)
    return d, nx, ny, float(cx), float(cy)


def _scan_from_config(run_cfg):
    try:
        c = json.load(open(run_cfg)).get('scan', {})
        return (c['width'], c['height'], c['step'])
    except Exception:
        return DEF_SCAN


def import_run(run_cfg, detector_dir, nframes=None, crop=512, rebin=1):
    sw, sh, st = _scan_from_config(run_cfg)
    pts = spiral_pattern(sw, sh, st, SAMPLE_RES)
    files = sorted(glob.glob(os.path.join(detector_dir, 'intensity_*.h5')), key=_idx)
    if len(files) != len(pts):
        print(f"  WARN: {len(files)} frames vs {len(pts)} spiral points")
    n = min(len(files), len(pts)); n = nframes or n
    files, pts = files[:n], pts[:n]
    _, nx, ny, cx, cy = _load_frame(files[0])
    cxi, cyi = int(round(cx)), int(round(cy)); h = crop // 2
    assert cxi - h >= 0 and cxi + h <= nx and cyi - h >= 0 and cyi + h <= ny, \
        f"crop {crop} centered ({cyi},{cxi}) out of frame ({ny},{nx})"
    fr = np.empty((n, crop, crop), np.float32)
    for k, fp in enumerate(files):
        d = _load_frame(fp)[0]
        fr[k] = d[cyi - h:cyi + h, cxi - h:cxi + h]
    mp = os.path.join(detector_dir, 'mask.npy')
    if not os.path.exists(mp):                            # mask sits in the outputs/ parent
        mp = os.path.join(os.path.dirname(detector_dir.rstrip('/')), 'mask.npy')
    mask = np.load(mp).astype(np.float32)
    mask = mask[cyi - h:cyi + h, cxi - h:cxi + h]
    N = crop; det_px = DET_PX
    if rebin > 1:
        fr = fr.reshape(n, crop // rebin, rebin, crop // rebin, rebin).sum((2, 4))
        mask = mask.reshape(crop // rebin, rebin, crop // rebin, rebin).min((1, 3))
        N = crop // rebin; det_px = DET_PX * rebin
    dx_rec = LAMBDA * Z_SD / (N * det_px)                 # Fraunhofer real-space pixel
    tx = np.array([p[0] for p in pts]) / dx_rec
    ty = np.array([p[1] for p in pts]) / dx_rec
    tx -= tx.min(); ty -= ty.min()
    meta = dict(nframes=n, frame=fr.shape[1:], dx_rec_nm=dx_rec * 1e9, N=N,
                det_px_um=det_px * 1e6, scan_um=(sw * 1e6, sh * 1e6, st * 1e6),
                beam_ctr=(cy, cx), span_px=(float(tx.max()), float(ty.max())),
                obj_px=(int(ty.max()) + N, int(tx.max()) + N))
    return fr, tx.astype(np.float32), ty.astype(np.float32), mask, meta


if __name__ == "__main__":
    R = os.environ.get("RUN", "/pscratch/sd/h/hgoel97/simulation_work/runs/"
                        "lcls_xpp_sim/base/9e3c6acc4678/config.json")
    DET = os.environ.get("DET", "/pscratch/sd/h/hgoel97/simulation_work/runs/apply_detector/"
                         "lcls_xpp_default_s1/da693a5b5ca9/stages/apply_detector/outputs/at_detector")
    NF = int(os.environ["NFRAMES"]) if "NFRAMES" in os.environ else None
    CROP = int(os.environ.get("CROP", 512)); REBIN = int(os.environ.get("REBIN", 1))
    fr, tx, ty, mask, meta = import_run(R, DET, NF, CROP, REBIN)
    print("META:", json.dumps({k: (v if not isinstance(v, tuple) else list(v))
                               for k, v in meta.items()}, default=float, indent=0))
    print(f"frames {fr.shape} dtype {fr.dtype}  min/mean/max {fr.min():.1f}/{fr.mean():.1f}/{fr.max():.1f}")
    print(f"mask {mask.shape} valid-frac {mask.mean():.3f}  (masked pixels = module gaps)")
    print(f"tx px [{tx.min():.1f},{tx.max():.1f}]  ty px [{ty.min():.1f},{ty.max():.1f}]  -> object ~{meta['obj_px']}")
    tot = fr.sum((1, 2))
    print(f"per-frame total counts: min/med/max {tot.min():.2e}/{np.median(tot):.2e}/{tot.max():.2e} "
          f"(dI/I rms {tot.std()/tot.mean():.3f} = the intensity jitter)")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        mid = len(fr) // 2
        ax[0].imshow(np.log10(fr[mid] + 1), cmap="magma"); ax[0].set_title(f"frame {mid} log10")
        ax[1].imshow(mask, cmap="gray"); ax[1].set_title("mask (module gaps)")
        sc = ax[2].scatter(tx, ty, c=np.arange(len(tx)), s=8, cmap="viridis")
        ax[2].set_aspect("equal"); ax[2].set_title("spiral positions (px, colored by shot)")
        plt.colorbar(sc, ax=ax[2]); plt.tight_layout(); plt.savefig("srw_import_preview.png", dpi=90)
        print("saved srw_import_preview.png")
    except Exception as e:
        print("preview skipped:", str(e)[:80])
