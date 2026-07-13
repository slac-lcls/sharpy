import os
import numpy as np
from PIL import Image
from Operators import make_probe, make_translations, cropmat
from wrap_ops import split_cuda
import config

if config.GPU:
    import cupy as xp
else:
    xp = np

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def simulate(nnx, nx=16, Dx=5, crop_zeros=True):  # B2: added nx, Dx, crop_zeros params — revert: def simulate(nnx): / nx, Dx = 16, 5
    ny, nny, Dy = nx, nnx, Dx
    nframes = nnx * nny

    illumination, _ = make_probe(nx, ny, r1=0.025*3, r2=0.085*3, fx=+20, fy=-20)

    np.random.seed(42)
    noise = xp.asarray(np.clip(np.random.normal(0, 0.01, (nx, ny)), 0, 1).astype(np.float32))
    illumination = xp.array(illumination, dtype=xp.complex64) + noise

    translations_x, translations_y = make_translations(Dx, Dy, nnx, nny, nnx*Dx, nny*Dy)

    # img0 = np.array(Image.open(os.path.join(_DATA_DIR, 'gold_balls.png')), np.float32) / 63.0  # original 1024×1024 — revert after large-scale run
    _raw = np.array(Image.open(os.path.join(_DATA_DIR, 'gold_balls.png')), np.float32)
    if crop_zeros:                          # B2: revert: remove if/else, keep only the tile line below
        _rows = np.any(_raw > 0, axis=1)
        _raw = _raw[_rows]                  # strip 192 zero rows top+bottom (640×1024)
        img0 = np.tile(_raw, (7, 4)) / 63.0  # 4480×4096 — covers 4080×4080 scan
    else:
        img0 = np.tile(_raw, (4, 4)) / 63.0  # 4096×4096 — uncropped, has zero border strips
    img0 = np.exp(0.69 * (-1 + 0.5j) * img0).astype(np.complex64)

    Nx = int(xp.ceil(xp.max(translations_x) - xp.min(translations_x)))
    Ny = Nx
    truth = xp.ascontiguousarray(cropmat(xp.array(img0, dtype=xp.complex64), [Nx, Ny]))

    frames_out = xp.zeros((nframes, nx, ny), dtype=xp.complex64)
    split_cuda(truth, frames_out, (translations_x + 1j*translations_y).astype(xp.complex64), illumination)
    data = xp.abs(xp.fft.fft2(frames_out))**2

    return data, illumination, truth, translations_x, translations_y, nframes, nx, ny, Nx, Ny
