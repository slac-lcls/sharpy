"""GPU validation for the per-pixel detector-weights (mask) path.

The CPU suite (sharpy/tests/test_proxd_weights.py) pins the weights semantics
on numpy; this script validates the pieces that only exist (or only matter) on
device, none of which have ever executed on a GPU:

  1. KERNELS  -- the ProxDW cp.fuse kernel and the fused proxd_resid_w
     RawKernel: fused==plain, W=ones==None bitwise, W=0 data inert
     (delegates to proxd_fused_test.validate / validate_weighted).
  2. AP_c     -- Alternating_projections_c(weights=) on a padded-detector
     synthetic (EXACT mirror of the CPU test geometry: 32px frames on a 64px
     object, 8x8 scan, central-disk detector, ~27% of |F| energy in the
     padding): masked-region |F| survives with W, collapses without; ones
     matches None.  Run under BOTH Project_data paths (plain + fused).
  3. BATCHED  -- Alternating_projections_batched_c(weights=), device tier,
     B<nframes: matches AP_c, and a per-frame (nf,nx,nx) weights stack
     (sliced per batch) matches the broadcast (nx,nx) mask.
  4. AP+LS    -- the CPU-style Alternating_projections with line_search=True
     and weights= on device (cupy reductions in ap_accel.line_search).

Expected ratios (from the CPU test, same geometry/seeds): weighted ~1.0
(>0.99), unweighted ~0.07 (<0.3).

CPU (Mac): runs only the kernel-fallback checks, then exits (the real CPU
coverage is the pytest suite).

GPU (Perlmutter; config.GPU=True locally as usual):
  srun -A lcls_g -C gpu -q interactive -N1 -n1 --gpus 1 -t 10 \
    bash -c 'source ~/sharpy-venv/bin/activate; cd $SCRATCH/sharpy/sharpy; \
             python -u proxd_weights_gpu_test.py'
The script flips Operators._FUSED_PROXD itself, so one run covers both the
plain and the SHARPY_FUSED_PROXD=1 production paths.  Exit code 0 = all pass.
"""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")   # Solvers imports pyplot; stay headless
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config  # noqa: F401
import Operators as O
import proxd_fused_test as PF

xp = O.xp
GPU = O.GPU

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILURES.append(name)


def rel(a, b):
    return float(xp.linalg.norm(a - b) / (1.0 + xp.linalg.norm(a)))


# ── 1. kernel level (CPU-safe: falls back to the plain path off-GPU) ─────────
print("== 1. kernel A/B (proxd_fused_test) ==")
check("fused==plain (unweighted)", PF.validate(64, 128))
check("fused==plain (weighted) + inert + ones", PF.validate_weighted(64, 128))

if not GPU:
    if os.environ.get("SLURM_JOB_ID"):
        # on an allocated node with config.GPU=False the run is a mistake --
        # fail loudly rather than let a batch pipeline record a false pass
        print("ERROR: running under Slurm but config.GPU is False -- "
              "set GPU = True in sharpy/config.py on this machine.")
        sys.exit(1)
    print("CPU: solver-level weights coverage lives in tests/test_proxd_weights.py;"
          " run this on a GPU node for the kernel+solver device validation.")
    sys.exit(0 if not FAILURES else 1)

import cupy as cp
import Solvers
from wrap_ops import split_cuda, overlap_cuda
from Operators import make_probe, make_translations, Split_Overlap_plan

name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
print(f"device: {name} | CuPy {cp.__version__}")

# ── padded-detector synthetic: EXACT mirror of the CPU test geometry ─────────
# (tests/test_proxd_weights.py::_padded_synthetic -- same sizes, seeds, probe)
nx, Nx = 32, 64
tx, ty = make_translations(8, 8, 8, 8, Nx, Nx)
probe, _ = make_probe(nx, nx)
probe = cp.asarray(probe, dtype=cp.complex64)
translations = (tx + 1j * ty).astype(cp.complex64)
nf = int(translations.size)

rng = np.random.default_rng(0)
# same seed/values as the CPU test; the final complex64 cast is REQUIRED --
# the cuda kernels index the image as float2, complex128 would corrupt
truth = cp.asarray(((rng.standard_normal((Nx, Nx))
                     + 1j * rng.standard_normal((Nx, Nx))).astype(np.complex64)
                    + 2.0).astype(np.complex64))

frames_true = cp.zeros((nf, nx, nx), dtype=cp.complex64)
split_cuda(truth, frames_true, translations, probe)
F_true = cp.fft.fft2(frames_true)
data_full = (cp.abs(F_true) ** 2).astype(cp.float32)

xi = np.fft.ifftshift(np.arange(nx) - nx / 2).reshape(nx, 1)
W = cp.asarray((np.sqrt(xi ** 2 + xi.T ** 2) <= 8).astype(np.float32))
data = (data_full * W).astype(cp.float32)          # padding holds 0, not data

M = 1.0 - W
denomM = float((cp.abs(F_true) * M).sum())
pad_frac = denomM / float(cp.abs(F_true).sum())
print(f"\ngeometry: {nf} frames {nx}x{nx} on {Nx}x{Nx}; padding |F| fraction = {pad_frac:.2f}")


def masked_ratio(img):
    """Fraction of the true padded-region |F| retained by split(img)*probe."""
    fb = cp.zeros((nf, nx, nx), dtype=cp.complex64)
    split_cuda(img, fb, translations, probe)
    F = cp.fft.fft2(fb)
    return float((cp.abs(F) * M).sum() / denomM)


def run_apc(weights, maxiter=8):
    img, _, _, res = Solvers.Alternating_projections_c(
        False, truth + 0, None, probe, tx, ty, overlap_cuda, split_cuda,
        data, False, maxiter, None, truth, 1, weights=weights)
    return img, res


# ── 2. Alternating_projections_c, plain and fused Project_data paths ─────────
ones = cp.ones((nx, nx), dtype=cp.float32)
img_w_ref = None
for fused in (False, True):
    O._FUSED_PROXD = fused
    tag = "fused" if fused else "plain"
    print(f"\n== 2. Alternating_projections_c weights= ({tag} ProxD path) ==")
    img_w, _ = run_apc(W)
    img_u, _ = run_apc(None)
    img_1, _ = run_apc(ones)
    r_w, r_u = masked_ratio(img_w), masked_ratio(img_u)
    print(f"  masked-|F| ratio: weighted={r_w:.3f}  unweighted={r_u:.3f}")
    check(f"APc/{tag}: masked region survives with W", r_w > 0.99, f"ratio={r_w:.3f}")
    check(f"APc/{tag}: masked region collapses without W", r_u < 0.3, f"ratio={r_u:.3f}")
    d = rel(img_u, img_1)
    check(f"APc/{tag}: weights=ones matches None", d < 1e-6, f"rel={d:.2e}")
    if fused is False:
        img_w_ref = img_w
d = rel(img_w_ref, img_w)
check("APc: fused vs plain weighted recon agree", d < 1e-4, f"rel={d:.2e}")
O._FUSED_PROXD = False

# ── 3. batched solver, device tier, B < nframes ──────────────────────────────
print("\n== 3. Alternating_projections_batched_c weights= (device tier, B=16) ==")


def run_batched(weights):
    img, _, _, _ = Solvers.Alternating_projections_batched_c(
        False, truth + 0, None, probe, tx, ty, overlap_cuda, split_cuda,
        data, False, 8, residuals_interval=1, frame_batch=16,
        frame_store="device", weights=weights)
    return img


img_bw = run_batched(W)
d = rel(img_w_ref, img_bw)
check("batched: weighted matches APc weighted", d < 1e-4, f"rel={d:.2e}")
check("batched: masked region survives", masked_ratio(img_bw) > 0.99,
      f"ratio={masked_ratio(img_bw):.3f}")
Wstack = cp.ascontiguousarray(cp.broadcast_to(W, (nf, nx, nx)))
d = rel(img_bw, run_batched(Wstack))
check("batched: per-frame (nf,nx,nx) weights stack == broadcast mask", d < 1e-6,
      f"rel={d:.2e}")
O._FUSED_PROXD = True                       # batched x fused x weights coverage
d = rel(img_bw, run_batched(W))
check("batched: fused ProxD path matches plain (weighted)", d < 1e-4, f"rel={d:.2e}")
O._FUSED_PROXD = False

# ── 4. CPU-style AP on device with the weighted line search ──────────────────
print("\n== 4. Alternating_projections + line_search weights= (device) ==")
Split, Overlap = Split_Overlap_plan(tx.ravel().astype(float), ty.ravel().astype(float),
                                    nx, nx, Nx, Nx)
frames_true_ls = O.Illuminate_frames(Split(truth), probe).astype(cp.complex64)
data_ls = (cp.abs(cp.fft.fft2(frames_true_ls)) ** 2).astype(cp.float32) * W
# floored normalization, as in the CPU test (the internally computed one has
# no zero-floor and the probe has near-zero pixels)
norm_ls = Overlap(O.Replicate_frame(cp.abs(probe) ** 2, nf)).real.astype(cp.float32) + 1e-8


def run_ap_ls(weights):
    img, _, _, _ = Solvers.Alternating_projections(
        False, truth + 0, None, probe, Overlap, Split, data_ls, False, 8,
        norm_ls, None, np.inf, line_search=True, weights=weights)
    return img


img_lw = run_ap_ls(W)
r_lw = masked_ratio(img_lw)
check("AP+line_search: masked region survives with W", r_lw > 0.99, f"ratio={r_lw:.3f}")
d = rel(run_ap_ls(None), run_ap_ls(ones))
check("AP+line_search: weights=ones matches None", d < 1e-5, f"rel={d:.2e}")

# ── verdict ──────────────────────────────────────────────────────────────────
print("\n" + ("ALL GPU WEIGHTS CHECKS PASSED" if not FAILURES
              else f"FAILED: {len(FAILURES)} check(s): {FAILURES}"))
sys.exit(0 if not FAILURES else 1)
