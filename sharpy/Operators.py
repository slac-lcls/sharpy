"""
Ptycho operators

Naming / vocabulary
-------------------
Three names in this codebase refer to the SAME underlying operation -- the
illumination-weighted inner product of two frames over their region of
overlap -- at different levels and on different backends:

  * Gramian  (the MATRIX)      -- the Hermitian matrix H of all pairwise
        overlap inner products between overlapping frames. Functions:
        Gramiam_plan / Gramiam_calc (CPU) / Gramiam_calc_cuda (GPU).
        ("Gramiam" is a long-standing misspelling of "Gramian", kept for
        compatibility -- renaming would churn many scripts/notebooks.)

  * braket   (one ENTRY, CPU)  -- <bra|ket>, a single element of H computed
        from the overlap of two frames. bra() / ket() / braket_i() are the
        readable reference; _braket_val_numba() is the parallel CPU version.

  * zQQz     (entries, GPU)    -- z*QQz, the same element computed inside the
        CUDA kernel (src/zQQz.cu, "dotp"), illumination + normalization
        applied per pixel. src/zQQz2.cu generalizes it to separate
        left/right illuminations (the coupled position-retrieval blocks).

The position-retrieval solver's _braket_coupled_numba is the same overlap
inner product generalized to derivative probes (the O11/O22/Ox terms) --
the CPU analog of zQQz2.cu.

So: Gramian = the matrix; braket / zQQz / pair-overlap = the element
computation (CPU reference / GPU / generalized). One assembly path
(plan["val2H"]) turns the per-pair entries into H for both CPU and GPU.

Key operators
-------------
Geometry / setup:
  make_probe(nx, ny, ...)        synthesize the (zone-plate) illumination
  make_translations(...)         close-packed scan positions
  map_frames(tx, ty, ...)        frame<->image index map `mapid` (integer)

Forward & adjoint  (image <-> frames):
  Splitc(img, mapid)             image -> frames   (gather overlapping windows)
  Overlapc(frames, Nx, Ny, mapid)  frames -> image (overlap-add; adjoint of Split)
  Illuminate_frames(frames, w)   multiply frames by the probe w

Propagation  (real space <-> detector / far field):
  Propagate / IPropagate         FFT2 / IFFT2

Data constraint:
  Project_data(frames, data)     Fourier-magnitude projection: replace the
        modeled magnitude with sqrt(data), keep the phase (ProxD inside).
        Optional weights= (per-pixel detector mask): W=1 pixels are projected,
        W=0 pixels (dead / zero-padded detector) keep their Fourier value
        unconstrained -- without it the AGM prox reads y=0 as "magnitude must
        be 0" and clamps |F| at every unmeasured pixel.

Gramian & phase synchronization:
  Gramiam_plan(...)              precompute overlap geometry + plan["val2H"]
  Gramiam_calc / _cuda           build the Gramian H (CPU Numba / GPU kernel)
  bra / ket / braket_i           one overlap inner product (readable reference)
  Precondition(_calc)            Jacobi (D H D) preconditioning of H
  Eigensolver                    dominant eigenvector (power iteration; eigsh
        is unreliable in single precision -- see synchronize path)
  synchronize_frames_c(...)      global per-frame phase fix from H's top eigvec

Metric:
  mse_calc(a, b)                 phase-corrected normalized MSE

The alternating-projections loop is, in these terms:
  Split -> Illuminate -> Propagate -> Project_data -> IPropagate
        -> [synchronize_frames_c] -> Overlap  (-> next iterate)
"""
#!/cds/home/y/yn754/anaconda3/envs/sharpy-env/bin/python

import numpy as np
import scipy as sp
import multiprocessing as mp
import matplotlib.pyplot as plt

import math

try:
    from numba import njit as _njit, prange as _prange
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

import os
import config

GPU = config.GPU

if GPU:
    import cupy as cp
    xp = cp
    import cupyx.scipy.sparse as sparse
    from fft_plan import fft2, ifft2
    import cupyx as cpx
    import cupy as cp
    from wrap_ops import gram_raw_kernel

else:
    xp = np
    import scipy.sparse as sparse
    from scipy.fftpack import fft2, ifft2


# import multiprocessing as mp

# timers: keep track of timing for different operators
from timeit import default_timer as timer

timers = {
    "Overlap": 0,
    "Split": 0,
    "Prox_data": 0,
    "Data_prox_tot": 0,
    "Propagate": 0,
    "mse_data": 0,
    "Gramiam": 0,
    "Gramiam_completion": 0,
    "Precondition": 0,
    "Eigensolver": 0,
    "Sync_setup": 0,
    "fd": 0,
}


def get_times():
    return timers


def reset_times():
    for keys in timers:
        timers[keys] = 0


def normalize_times():
    tot = 0
    for keys in timers:
        tot += timers[keys]
    if tot != 0:
        for keys in timers:
            timers[keys] /= tot
    return tot


def Propagate(frames):
    # simple propagation
    #print('frames shape',frames.shape)
    #print('frames type', type(frames))
    return fft2(frames)


def IPropagate(frames):
    # simple inverse propagation
    # return xp.fft.ifft2(frames)
    return ifft2(frames)


eps = xp.float32(1e-16)


if GPU:

    @cp.fuse(kernel_name="ProxD")
    def _ProxD_hard(x, y, eps):
        # AGM (amplitude / sqrt-Gaussian) HARD magnitude projection: |z| <- sqrt(data),
        # phase kept. Single sqrt-of-ratio (was two sqrt + a divide). GPU-fast reciprocal
        # form: x * cpx.rsqrt((|x|^2+eps)/(y+eps)) (as in Prox_data_r) -- the
        # SHARPY_FUSED_PROXD RawKernel below already uses rsqrtf. Noise-model hook
        # (KL/Anscombe): see ProxD_noise + the note above Project_data.
        return x * xp.sqrt((y + eps) / ((xp.real(x) ** 2 + xp.imag(x) ** 2) + eps))

    @cp.fuse(kernel_name="ProxDW")
    def _ProxD_hard_w(x, y, w, eps):
        # weighted hard projection: convex combination of the projected and the
        # untouched value, x*(w*s + (1-w)). w=1 -> x*s bitwise (1*s+0 is exact),
        # w=0 -> x untouched (masked/padded detector pixels stay unconstrained).
        s = xp.sqrt((y + eps) / ((xp.real(x) ** 2 + xp.imag(x) ** 2) + eps))
        return x * (w * s + (1.0 - w))


else:

    def _ProxD_hard(x, y, eps):
        # AGM hard magnitude projection (numpy has no rsqrt ufunc -> sqrt-of-ratio).
        return x * xp.sqrt((y + eps) / ((xp.real(x) ** 2 + xp.imag(x) ** 2) + eps))

    def _ProxD_hard_w(x, y, w, eps):
        # weighted variant: see the GPU twin above.
        s = xp.sqrt((y + eps) / ((xp.real(x) ** 2 + xp.imag(x) ** 2) + eps))
        return x * (w * s + (1.0 - w))


def ProxD(x, y, eps, w=None):
    """AGM hard magnitude projection |x| <- sqrt(y), phase kept.

    w: optional per-pixel detector weights in {0,1} (or [0,1]), broadcastable
    against x (e.g. one (N,N) detector mask for (V,N,N) frames). W=1 applies
    the projection, W=0 keeps x unchanged -- dead / zero-padded pixels must NOT
    be clamped to |x|=0 (y=0 there means "unmeasured", not "dark"); leaving
    them free is what allows extrapolation beyond the physical detector.
    w=None reproduces the unweighted result bit-for-bit. Masked-pixel y values
    are never used (any finite garbage there is multiplied by w=0)."""
    if w is None:
        return _ProxD_hard(x, y, eps)
    return _ProxD_hard_w(x, y, w, eps)


# ---------------------------------------------------------------------------
# Noise-model-selectable data prox (REFERENCE; NOT wired in -> default stays AGM-hard).
# KEY: at the HARD projection limit (plain AP, tau=None) EVERY model collapses to
# |z| <- sqrt(data), so the metric only matters once sharpy runs a RELAXED / proximal
# data step (finite tau: RAAR-with-prox / ADMM). Anscombe additionally needs the photon
# GAIN ("what 1 photon means" -- y must be COUNTS, not ADU). Formulas validated in
# scratchpad/proxd_noise.py; memory bpr-survey-proxd (arXiv:2211.06619 eqs 14-16).
# ---------------------------------------------------------------------------
def ProxD_noise(x, y, tau=None, metric="amplitude", eps=eps, w=None):
    """Magnitude data prox. tau=None -> hard (== ProxD). metric='amplitude' (AGM /
    sqrt-Gaussian) or 'poisson' (IPM / KL). y = intensity (COUNTS for poisson).

    w: optional per-pixel detector weights in {0,1} (or [0,1]), broadcastable
    against x -- W=0 (dead/padded) pixels keep x unchanged, so their y values
    never influence the result (must be finite; they are 0 for padded data).
    w=None (default) is bit-identical to the historical unweighted path."""
    r = xp.sqrt(xp.real(x) ** 2 + xp.imag(x) ** 2 + eps)
    if tau is None:
        m = xp.sqrt(y + eps)                              # hard limit == ProxD
    elif metric == "amplitude":
        m = (tau * xp.sqrt(y + eps) + r) / (1.0 + tau)    # relaxed AGM (convex combo)
    else:  # 'poisson'/KL (IPM): positive root of (2+1/tau) m^2 - (r/tau) m - 2y = 0
        A = 2.0 + 1.0 / tau
        m = (r / tau + xp.sqrt((r / tau) ** 2 + 8.0 * y * A)) / (2.0 * A)
    if w is None:
        return x * (m / r)
    return x * (w * (m / r) + (1.0 - w))                  # w=1 exact, w=0 keeps x


# ---------------------------------------------------------------------------
# Fused ProxD + data-residual: ONE kernel computes  z <- ProxD(z)  AND
# ssq = sum |z - ProxD(z)|^2 = sum (|z| - sqrt(data))^2  in a single memory pass,
# in place (zero temporaries).  Replaces the two-pass path (|z| alloc + sqrt(data)
# alloc + ProxD alloc + a separate norm reduction).  Opt-in via SHARPY_FUSED_PROXD=1
# (default off -> byte-identical to the cp.fuse ProxD path below).  Supersedes the
# dead `dotnorm2` ReductionKernel further down: a ReductionKernel can emit only the
# scalar, not also write back the projected array; a RawKernel does both.
# Requires complex64/float32, C-contiguous (production GPU dtypes); off-spec inputs
# fall back to the plain path so we never silently corrupt.  atomicAdd(double) => sm_60+.
# ---------------------------------------------------------------------------
_FUSED_PROXD = GPU and os.environ.get("SHARPY_FUSED_PROXD", "0") == "1"

if GPU:
    # Dependency-free: treat complex64 as interleaved float* (re,im), built-in
    # intrinsics only (rsqrtf/atomicAdd) + a hand-rolled shared-mem reduction.
    # NO thrust/cub/cupy-complex includes -> no jitify (those headers break NVRTC
    # here). Launch 256 threads (matches sh[256] + the power-of-two reduction).
    _proxd_resid_src = r"""
    extern "C" __global__ void proxd_resid(
            float* z, const float* data, double* g_ssq,
            const float eps, const int do_resid, const long long n) {
        // n = number of COMPLEX elements; z holds 2n floats (re,im interleaved).
        __shared__ double sh[256];
        double local = 0.0;
        const long long gstride = (long long)gridDim.x * blockDim.x;
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
             i < n; i += gstride) {
            float re = z[2*i], im = z[2*i + 1];
            float di = data[i];
            float r2 = re*re + im*im;
            // AGM scale s = |Pz|/|z| = sqrt((data+eps)/(r2+eps)); rsqrtf = one SFU op
            // (kernel is DRAM-bound: the win is the fusion, not the rsqrt).
            // Noise-model hook: swap s for the KL/IPM or Anscombe target -- see ProxD_noise.
            float s  = rsqrtf((r2 + eps) / (di + eps));
            z[2*i] = re * s; z[2*i + 1] = im * s;     // ProxD, in place
            if (do_resid) {
                // |z-Pz|^2 = r2(1-s)^2 == (|z|-sqrt(data))^2 (to float eps); no extra sqrt.
                float t = 1.0f - s;
                local += (double)(r2 * t * t);
            }
        }
        if (do_resid) {                              // block reduce -> atomicAdd
            int tid = threadIdx.x;
            sh[tid] = local; __syncthreads();
            for (int w = blockDim.x >> 1; w > 0; w >>= 1) {
                if (tid < w) sh[tid] += sh[tid + w];
                __syncthreads();
            }
            if (tid == 0) atomicAdd(g_ssq, sh[0]);
        }
    }
    """
    _proxd_resid = cp.RawKernel(_proxd_resid_src, "proxd_resid")
    _proxd_ssq = cp.zeros(1, dtype=cp.float64)        # persistent accumulator (zeroed per call)

    # Weighted twin of proxd_resid: per-pixel detector weights w (typically one
    # (N,N) mask broadcast over frames: index i % nw). Scale becomes
    # w*s + (1-w)  (w=0 keeps z: dead/padded pixels unconstrained), residual
    # becomes  sum w*(|z|-sqrt(data))^2  (only measured pixels counted).
    # Separate kernel (not a flag in proxd_resid) so the default path stays
    # byte-identical and pays neither the load nor the modulo.
    _proxd_resid_w_src = r"""
    extern "C" __global__ void proxd_resid_w(
            float* z, const float* data, const float* wgt, double* g_ssq,
            const float eps, const int do_resid,
            const long long n, const long long nw) {
        // n complex elements; wgt has nw entries, broadcast as wgt[i % nw].
        __shared__ double sh[256];
        double local = 0.0;
        const long long gstride = (long long)gridDim.x * blockDim.x;
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
             i < n; i += gstride) {
            float re = z[2*i], im = z[2*i + 1];
            float di = data[i];
            float w  = wgt[i % nw];
            float r2 = re*re + im*im;
            float s  = rsqrtf((r2 + eps) / (di + eps));
            float se = w * s + (1.0f - w);            // w=1 -> s exactly; w=0 -> 1
            z[2*i] = re * se; z[2*i + 1] = im * se;
            if (do_resid) {
                float t = 1.0f - s;
                local += (double)(w * (r2 * t * t));  // w * (|z|-sqrt(data))^2
            }
        }
        if (do_resid) {
            int tid = threadIdx.x;
            sh[tid] = local; __syncthreads();
            for (int wd = blockDim.x >> 1; wd > 0; wd >>= 1) {
                if (tid < wd) sh[tid] += sh[tid + wd];
                __syncthreads();
            }
            if (tid == 0) atomicAdd(g_ssq, sh[0]);
        }
    }
    """
    _proxd_resid_w = cp.RawKernel(_proxd_resid_w_src, "proxd_resid_w")


def _proxd_resid_apply(frames, frames_data, compute_residuals, weights=None):
    """Fused ProxD + residual on Fourier-space `frames`, in place. Returns (frames, mse).
    Falls back to the plain cp.fuse path for any off-spec dtype/layout.
    weights: optional per-pixel detector weights (see Project_data); the fused
    weighted kernel additionally requires float32 C-contiguous weights whose
    shape matches the trailing dims of frames (broadcast over leading dims)."""
    on_spec = (GPU and frames.dtype == xp.complex64 and frames_data.dtype == xp.float32
               and frames.flags.c_contiguous and frames_data.flags.c_contiguous)
    if weights is not None:
        on_spec = (on_spec and weights.dtype == xp.float32 and weights.flags.c_contiguous
                   and frames.shape[frames.ndim - weights.ndim:] == weights.shape)
    if not on_spec:
        if compute_residuals:
            if weights is None:
                mse = xp.linalg.norm(xp.abs(frames) - xp.sqrt(frames_data))
            else:  # sqrt(w) keeps this an L2 norm; w=1 multiplies by exactly 1.0
                mse = xp.linalg.norm(xp.sqrt(weights) * (xp.abs(frames) - xp.sqrt(frames_data)))
        else:
            mse = eps
        return ProxD(frames, frames_data, eps, w=weights), mse
    n = frames.size
    threads = 256                                     # must match cub::BlockReduce<..,256>
    blocks = min(65535, (n + threads - 1) // threads) # grid-stride covers the rest
    do_resid = 1 if compute_residuals else 0
    if do_resid:
        _proxd_ssq.fill(0)
    if weights is None:
        _proxd_resid((blocks,), (threads,),
                     (frames, frames_data, _proxd_ssq, eps, np.int32(do_resid), np.int64(n)))
    else:
        _proxd_resid_w((blocks,), (threads,),
                       (frames, frames_data, weights, _proxd_ssq, eps,
                        np.int32(do_resid), np.int64(n), np.int64(weights.size)))
    mse = xp.sqrt(_proxd_ssq[0]) if do_resid else eps
    return frames, mse


# NOTE: a fused ReductionKernel for the eps_S residual ||frames-frames_old|| was
# tried and REVERTED -- it ran 30x SLOWER than xp.linalg.norm (0.35 ms) on A100
# (10.4 ms/call; the complex .real()/.imag() map deoptimizes the reduction).
# xp.linalg.norm(a-b) is already cuBLAS-nrm2-fast, so the residual is NOT fused;
# only the frames_old COPY is dropped (reference trick in Solvers, memory win).


def Project_data(frames, frames_data, compute_residuals=False, weights=None):
    """Fourier-magnitude projection (+ optional data residual).

    weights: optional per-pixel detector weights in {0,1} (or [0,1]),
    broadcastable against the frames -- typically ONE (N,N) detector mask
    shared by all (V,N,N) frames. W=1 pixels get the magnitude projection,
    W=0 pixels (dead / zero-padded detector regions, where frames_data holds
    0, not a measurement) keep their current Fourier value, so the iterate is
    free to extrapolate |F| beyond the physical detector instead of having it
    clamped to 0. The residual likewise counts only weighted pixels:
    mse = sqrt(sum W*(|F|-sqrt(data))^2). weights=None (default) is
    bit-identical to the historical unweighted behaviour."""
    if weights is not None:
        # accept bool/int masks and lists; float32 also satisfies the fused kernel
        weights = xp.asarray(weights)
        if weights.dtype != xp.float32:
            weights = weights.astype(xp.float32)
    time00 = timer()
    time0 = time00
    # apply Fourier magnitude projections
    frames = Propagate(frames)
    timers["Propagate"] += timer() - time0

    time0 = timer()
    if _FUSED_PROXD:
        # one kernel: ProxD in place + data residual, zero temporaries
        frames, mse = _proxd_resid_apply(frames, frames_data, compute_residuals, weights)
        timers["Prox_data"] += timer() - time0
    else:
        # compute mse
        if compute_residuals:
            if weights is None:
                mse = xp.linalg.norm(xp.abs(frames) - xp.sqrt(frames_data))
            else:  # sum only weighted pixels; sqrt(w)=1.0 exactly where w=1
                mse = xp.linalg.norm(xp.sqrt(weights) * (xp.abs(frames) - xp.sqrt(frames_data)))
        else:
            mse = eps

        timers["mse_data"] += timer() - time0

        time0 = timer()
        frames = ProxD(frames, frames_data, eps, w=weights)

        timers["Prox_data"] += timer() - time0

    time0 = timer()
    frames = IPropagate(frames)
    timers["Propagate"] += timer() - time0
    timers["Data_prox_tot"] += timer() - time00

    return frames, mse


"""
if GPU:
    dotnorm2 = xp.ReductionKernel(
                'T x ,T x1, T y, T y1, Z zz', 'Z z',
                '(x-x1)*(y-y1)+zz*((y-y1)*(y-y1))',#'(x-y)* conj(x-y)+zz*(x*conj(x))',
                'a + b','z = a','0')
else:
    mse = xp.linalg.norm(xp.abs(frames)-xp.sqrt(frames_data))
"""


def Prox_data_r(frames, frames_data_r, compute_residuals=False):
    time00 = timer()
    time0 = time00
    # apply Fourier magnitude projections
    frames = Propagate(frames)
    timers["Propagate"] += timer() - time0

    time0 = timer()

    # compute mse
    if compute_residuals:
        mse = xp.linalg.norm(frames - cpx.rsqrt(frames_data_r))

    else:
        mse = eps

    timers["mse_data"] += timer() - time0

    time0 = timer()
    frames *= cpx.rsqrt(((frames * frames.conj()).real + eps) * (frames_data_r))

    # frames *= cpx.rsqrt(((frames.real**2+frames.imag**2).real+eps)*(frames_data_r))

    # frames *= xp.sqrt((frames_data+eps)/(xp.abs(frames)**2+eps))
    # frames *= xp.sqrt((frames_data+eps)/((frames.real*frames.real + frames.imag*frames.imag)+eps))
    # frames *= xp.sqrt((frames_data+eps)/((frames*frames.conj()).real+eps))

    timers["Prox_data"] += timer() - time0

    time0 = timer()
    frames = IPropagate(frames)
    timers["Propagate"] += timer() - time0
    timers["Data_prox_tot"] += timer() - time00

    return frames, mse


# def prox_data_plan(frames_data):
#     """
#     Parameters
#     ----------
#     frames_data : diffraction frames
#         3d matrix

#     Returns
#     -------
#     prox_data : function(frames, compute_residuals = False)
#         compute the proximal operator to frames and returns projected frames

#     """
#     if GPU:
#         if True:

#             def prox_data(frames, compute_residuals=False):
#                 frames, mse = Project_data(
#                     frames, frames_data, compute_residuals=compute_residuals
#                 )
#                 return frames, mse

#         else:
#             fdr = xp.float32(1) / (frames_data + eps)
#             #  print('type fdr:', type(fdr), 'dtype:', fdr.dtype)
#             def prox_data(frames, compute_residuals=False):
#                 frames, mse = Prox_data_r(
#                     frames, fdr, compute_residuals=compute_residuals
#                 )
#                 return frames, mse

#     else:

#         def prox_data(frames, compute_residuals=False):
#             frames, mse = Project_data(
#                 frames, frames_data, compute_residuals=compute_residuals
#             )
#             return frames, mse

#         # rox_data = lambda frames, compute : Project_data(frames, frames_data, compute_residuals = False)

#     return prox_data


def make_probe(nx, ny, r1=0.03, r2=0.06, fx=0.0, fy=0.0):
    """
    make an illumination (probe) in a (nx, ny) frame shape
    r1,r2 fractions of of the frame width
    fx,fy:  x-y quadradic fase (focus)

    """

    xi = xp.reshape(xp.arange(0, nx) - nx / 2, (nx, 1)) 

    xi = xp.fft.ifftshift(xi)

    rr = xp.sqrt(xi**2 + (xi.T) ** 2)
    r1 = r1 * nx  # define zone plate circles
    r2 = r2 * nx

    lens_mask = (rr >= r1) & (rr <= r2)

    phase = xp.exp(1j * fx * xp.pi * ((xi / nx) ** 2)) * xp.exp(
        1j * fy * xp.pi * ((xi.T / nx) ** 2)
    )

    Fprobe = lens_mask * phase

    probe = xp.fft.fftshift(xp.fft.ifft2(Fprobe))
    probe = probe / max(abs(probe).flatten())
    return probe,lens_mask


# close packing translations
def make_translations(Dx, Dy, nnx, nny, Nx, Ny):
    """
    make scan positions using spacing Dx,Dy, number of steps nnx, nny,
    image width Nx,Ny. The lattice is periodic with close-packing arrangement

    """
    #ix, iy = xp.meshgrid(
    #    xp.arange(0, Dx * nnx, Dx) + Nx / 2 - Dx * nnx / 2 + 1,
    #    xp.arange(0, Dy * nny, Dy) + Ny / 2 - Dy * nny / 2 + 1,
    #)
    
    ix, iy = xp.meshgrid(
        xp.arange(0, Dx * nnx, Dx) + Nx // 2 - Dx * nnx // 2 + 1,
        xp.arange(0, Dy * nny, Dy) + Ny // 2 - Dy * nny // 2 + 1,
    )
    
    xshift = math.floor(Dx / 2) * xp.mod(xp.arange(1, xp.size(ix, 1) + 1), 2)
    
    # adding shift in the x-direction to make close-packing lattice
    ix = xp.transpose(xp.add(xp.transpose(ix), xshift))
    ix = ix - xp.min(ix)
    iy = iy - xp.min(iy)

    ix = xp.reshape(ix, (nnx * nny, 1, 1))
    iy = xp.reshape(iy, (nnx * nny, 1, 1))
    ix = xp.asarray(ix)
    iy = xp.asarray(iy)
    return ix, iy


def map_frames(translations_x, translations_y, nx, ny, Nx, Ny):
    """
    return frame mapping: frames = image[mapid]
    """

    # map frames to image indices
    translations_x = xp.reshape(
        xp.transpose(translations_x), (xp.size(translations_x), 1, 1)
    )
    translations_y = xp.reshape(
        xp.transpose(translations_y), (xp.size(translations_y), 1, 1)
    )

    xframeidx, yframeidx = xp.meshgrid(xp.arange(nx), xp.arange(ny))
    # print('translations shapes:',xp.shape(translations_x),'frameidx',xp.shape(xframeidx))

    spv_x = xp.add(xframeidx, translations_x)
    spv_y = xp.add(yframeidx, translations_y)

    # enforce periodic boundaries
    mapidx = xp.mod(spv_x, Nx)
    mapidy = xp.mod(spv_y, Ny)

    mapid = xp.add(mapidx, mapidy * Nx)
    # mapid=xp.add(mapidx*Nx,mapidy)
    mapid = mapid.astype(np.uint32)

    return mapid


def Splitc(img, mapid):
    # Split an image into frames given mapping
    time0 = timer()
    frames_out = (img.ravel())[mapid]
    timers["Split"] += timer() - time0
    return frames_out


def Overlapc(frames, Nx, Ny, mapid):  # check
    # overlap frames onto an image by scatter-add (adjoint of Splitc).
    # Uses xp.bincount so it runs on BOTH numpy and cupy -- numpy_groupies
    # is CPU-only and raises on cupy arrays. minlength=Nx*Ny zero-fills
    # pixels no frame covers, so partial-coverage (padded) geometries work.
    # mapid = mapidx + mapidy*Nx (row-stride Nx) => the image is [y, x] with
    # width Nx, i.e. shape (Ny, Nx); reshaping to (Nx, Ny) only happens to be
    # correct when Nx == Ny.
    time0 = timer()
    g = mapid.ravel()
    f = frames.ravel()
    if xp.iscomplexobj(f):
        accum = xp.bincount(g, weights=f.real, minlength=Nx * Ny) + 1j * xp.bincount(
            g, weights=f.imag, minlength=Nx * Ny
        )
    else:
        accum = xp.bincount(g, weights=f, minlength=Nx * Ny)
    accum = xp.reshape(accum, (Ny, Nx))
    timers["Overlap"] += timer() - time0
    return accum


def Overlapd(frames, SS, shape):  # check
    # overlap frames onto an image using SPmV and reshape

    time0 = timer()
    output = SS * frames.ravel()
    output.shape = shape
    timers["Overlap"] += timer() - time0
    return output


#    accum = xp.reshape(numpy_groupies.aggregate(mapid.ravel(),frames.ravel()),(Nx,Ny))


def Split_Overlap_plan(translations_x, translations_y, nx, ny, Nx, Ny):
    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)

    # for cupy we need a sparse matrix. shape=(Nx*Ny, mapid.size) is REQUIRED:
    # without it coo_matrix infers rows = max(mapid)+1, so on partial-coverage
    # geometries (object larger than the scan -> far corner unlit) SS has fewer
    # than Nx*Ny rows and Overlapd's reshape fails. (Ny, Nx) matches the
    # [y, x] layout of mapid (see Overlapc).
    col = xp.arange(mapid.size)
    val = xp.ones((mapid.size), dtype=np.float32)
    SS = sparse.coo_matrix(
        (val.ravel(), (mapid.ravel(), col.ravel())), shape=(Nx * Ny, mapid.size)
    )
    SS = sparse.csr_matrix(SS)

    Split = lambda img: Splitc(img, mapid)
    Overlap = lambda frames: Overlapd(frames, SS, (Ny, Nx))
    return Split, Overlap


def Split_plan(translations_x, translations_y, nx, ny, Nx, Ny):
    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)
    Split = lambda img: Splitc(img, mapid)
    return Split


def Overlap_plan(translations_x, translations_y, nx, ny, Nx, Ny):
    mapid = map_frames(translations_x, translations_y, nx, ny, Nx, Ny)
    Overlap = lambda frames: Overlapc(frames, Nx, Ny, mapid)
    return Overlap


def crop_center(img, cropx, cropy):
    # crop an image
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]


def cropmat(img, size):
    # crop an image to a given size
    left0 = math.floor((xp.size(img, 0) - size[0]) / 2)
    right0 = size[0] + math.floor((xp.size(img, 0) - size[0]) / 2)
    left1 = math.floor((xp.size(img, 1) - size[1]) / 2)
    right1 = size[1] + math.floor((xp.size(img, 1) - size[1]) / 2)
    crop_img = img[left0:right0, left1:right1]
    return crop_img


def Overlapc0(frames, Nx, Ny, mapid):

    # minlength=Nx*Ny zero-fills uncovered pixels (see Overlapc); the imag
    # part must be scaled by 1j to reconstruct the complex sum.
    ret = xp.bincount(
        mapid.ravel(), weights=(frames.ravel()).real, minlength=Nx * Ny
    ) + 1j * xp.bincount(
        mapid.ravel(), weights=(frames.ravel()).imag, minlength=Nx * Ny
    )
    ret.shape = (Ny, Nx)
    return ret


# broadcast
def Illuminate_frames(frames, Illumination):
    # frames =frames*xp.reshape(Illumination,(1,xp.shape(Illumination)[0],xp.shape(Illumination)[1]))
    Illuminated = frames * xp.reshape(
        Illumination, (1, xp.shape(Illumination)[0], xp.shape(Illumination)[1])
    )
    return Illuminated


def Replicate_frame(frame, nframes):
    # replicate a frame along the first dimension
    Replicated = xp.repeat(frame[xp.newaxis, :, :], nframes, axis=0)
    return Replicated


def Sum_frames(frames):
    Summed = xp.add(frames, axis=0)
    return Summed


def Stack_frames(frames, omega):
    # multiply frames by a vector in the first dimension
    omega = omega.reshape([len(omega), 1, 1])
    # stv=xp.multiply(frames,omega)
    stv = frames * omega
    return stv


def ket(ystackr, dx, dy, bw=0):
    # extracts the portion of the left frame that overlaps
    # dxi=dx[ii,jj].astype(int)
    # dyi=dy[ii,jj].astype(int)
    nx, ny = ystackr.shape
    dxi = dx.astype(int)
    dyi = dy.astype(int)
    #dxi = dx
    #dyi = dy
    ket = ystackr[
        max([0, dyi]) + bw : min([nx, nx + dyi]) - bw,
        max([0, dxi]) + bw : min([nx, nx + dxi]) - bw,
    ]
    #print('range frames left then right',max([0, dyi]) + bw, min([nx, nx + dyi]) - bw,max([0, dxi]) + bw,min([nx, nx + dxi]) - bw)
    # ket=ystackr[max([0,dxi])+bw:min([nx,nx+dxi])-bw,
    #             max([0,dyi])+bw:min([nx,nx+dyi])-bw]

    return ket


def bra(ystackl, dx, dy, bw=0):
    # calculates the portion of the right frame that overlaps
    bra = ket(ystackl, dx, dy, bw)
    return bra


#def braket(ystackl, ystackr, dd, bw):
def braket(ystackl, ystackr, dx,dy, bw):
    # calculates inner products between the overlapping portion
    #    dxi=dx[ii,jj]
    #    dyi=dy[ii,jj]
    #dxi = dd.real
    #dyi = dd.imag

    # bracket=xp.sum(xp.multiply(bra(ystackl[jj],nx,ny,-dxi,-dyi),ket(ystackr[ii],nx,ny,dxi,dyi)))
    # bracket=xp.vdot(bra(ystackl[jj],nx,ny,-dxi,-dyi),ket(ystackr[ii],nx,ny,dxi,dyi))
    #bket = xp.vdot(bra(ystackl, -dxi, -dyi, bw), ket(ystackr, dxi, dyi, bw))
    bket = xp.vdot(bra(ystackl, -dx, -dy, bw), ket(ystackr, dx, dy, bw))

    return bket

def braket_i(ii,framesl,framesr,col,row,dx,dy,bw):
#def braket_i(ystackl,ystackr,dx,dy,bw):
    #val = braket(framesl[col[ii]], framesr[row[ii]], dx[ii],dy[ii], bw).get()
    val = xp.vdot(bra(framesl[col[ii]], -dx[ii], -dy[ii], bw), ket(framesr[row[ii]], dx[ii], dy[ii], bw))
   #    val[ii] = braket(framesl[col[ii]], framesr[row[ii]], dd[ii], bw)
    return val
#braket_i = cp.fuse(kernel_name='braket_i')(braket_i)


if _HAVE_NUMBA:

    @_njit(parallel=True, cache=True, fastmath=True)
    def _braket_val_numba(framesl, framesr, col, row, dx, dy, bw, frames_norm, out):
        """Parallel CPU <bra|ket> over all overlapping pairs (CPU twin of zQQz.cu).

        Same strategy as Gramiam_calc_cuda: one parallel iteration per pair
        (the CUDA block) with a serial inner sum over the overlap (the CUDA
        threads + BlockReduce collapse to this on CPU). For each pair ii,

            out[ii] = <bra | ket> / (||frames_norm[col]|| ||frames_norm[row]||)

        where, exactly as in bra()/ket(),
            bra = overlap window of the LEFT frame  framesl[col]  shifted (-dx,-dy)
            ket = overlap window of the RIGHT frame framesr[row]  shifted (+dx,+dy)
        and <bra|ket> = sum conj(bra) * ket over the overlap.
        """
        nnz = col.shape[0]
        nx = framesl.shape[1]  # square frames (ket uses nx for both axes)
        for ii in _prange(nnz):
            c = col[ii]
            r = row[ii]
            dxi = dx[ii]
            dyi = dy[ii]
            # bra window (left frame, -dx,-dy);  ket window (right frame, +dx,+dy)
            bra_r0 = max(0, -dyi) + bw
            bra_c0 = max(0, -dxi) + bw
            ket_r0 = max(0, dyi) + bw
            ket_c0 = max(0, dxi) + bw
            hgt = nx - abs(dyi) - 2 * bw
            wid = nx - abs(dxi) - 2 * bw
            acc = 0.0 + 0.0j
            for a in range(hgt):           # <bra|ket> = sum conj(bra) * ket
                for b in range(wid):
                    acc += np.conj(framesl[c, bra_r0 + a, bra_c0 + b]) \
                        * framesr[r, ket_r0 + a, ket_c0 + b]
            out[ii] = acc / (frames_norm[c] * frames_norm[r])
    

def Gramiam_calc(framesl, framesr, plan,frames_norm):
    # computes all the inner products between overlaping frames
    col = plan["col"]
    row = plan["row"]
    #dd = plan["dd"]
    dx = plan["dx"]
    dy = plan["dy"]
    bw = plan["bw"]
    val = plan["val"]
    #print(type(col.dtype),type(dd.dtype))
    
    nframes = framesl.shape[0]
    nnz = len(col)
    # val=xp.empty((nnz,1),dtype=framesl.dtype)
    # val = shared_array(shape=(nnz),dtype=xp.complex128)
    
    #col=np.array([np.argwhere(col[i]==np.unique(col)) for i in range(np.size(col))]).ravel()
    #row=np.array([np.argwhere(row[i]==np.unique(row)) for i in range(np.size(row))]).ravel()
    
   
    # def proc1(ii):
    #    return braket(framesl[col[ii]],framesr[row[ii]],dd[ii],bw)

            
    #@cp.fuse(kernel_name='braket_i')
    
    time0 = timer()
    #print(nnz)
    if (not GPU) and _HAVE_NUMBA:
        # fast CPU path: parallel <bra|ket> kernel (same strategy as the GPU
        # Gramiam_calc_cuda). Fills `val` with the preconditioned (D@H@D)
        # inner products in one threaded pass instead of a Python loop.
        val = xp.empty(nnz, dtype=xp.complex128)
        _braket_val_numba(
            xp.ascontiguousarray(framesl),
            xp.ascontiguousarray(framesr),
            xp.ascontiguousarray(col).astype(np.int64),
            xp.ascontiguousarray(row).astype(np.int64),
            xp.ascontiguousarray(dx).astype(np.int64),
            xp.ascontiguousarray(dy).astype(np.int64),
            int(bw),
            xp.ascontiguousarray(frames_norm),
            val,
        )
    else:
        # reference path: explicit <bra|ket> per overlapping pair
        for ii in range(nnz):
            #braket_i(ii)
            val[ii] = braket_i(ii,framesl,framesr,col,row,dx,dy,bw)
            val[ii] /= frames_norm[col[ii]]*frames_norm[row[ii]] #calculate D @ H @ D

    #print('true value',val)


    timers["Gramiam"] += timer() - time0
    time0 = timer()

    # Assemble the Hermitian H from the triu values. Use the SAME precomputed
    # sparse structure as Gramiam_calc_cuda (plan["val2H"], built once in
    # mapu2all): it just refills H.data and fills the lower triangle by
    # conjugation, instead of rebuilding coo->csr every call. One assembly
    # path now serves both the CPU and GPU Gramian.
    H = plan["val2H"](val.ravel())
    timers["Gramiam_completion"] += timer() - time0

    return H


_src = os.path.join(os.path.dirname(__file__), 'src')

with open(os.path.join(_src, 'zQQz.cu'), 'r') as f:
    zQQz_raw_kernel = f.read()


def Gramiam_calc_cuda(frames,plan,illumination,normalization,frames_norm,timers=timers):

    t0 = timer()

    # The zQQz.cu RawKernel hardcodes thrust::complex<float> for every array, so
    # it reinterprets the raw bytes as complex64. If any input is complex128
    # (e.g. a dtype promotion upstream -- a float64 in the image division can lift
    # frames to complex128), the kernel misreads the 16-byte elements as 8-byte
    # and returns garbage/NaN. Cast to complex64 (no-op/no-copy when already so).
    frames        = frames.astype(xp.complex64, copy=False)
    illumination  = illumination.astype(xp.complex64, copy=False)
    normalization = normalization.astype(xp.complex64, copy=False)
    frames_norm   = frames_norm.astype(xp.complex64, copy=False)

    value = plan["gram_calc"](frames,frames_norm, illumination, normalization)

    timers['Gramiam'] = timer()-t0

    t0 = timer()
    H = plan["val2H"](value.ravel())
    timers['Gramiam_completion'] = timer()-t0
    #H0 = plan["val2H"](xp.ones_like(value.ravel()))
      

    '''
        
    col = plan['col']
    row = plan['row']
    dx = plan['dx']
    dy = plan['dy']
    bw = plan['bw']
    value = plan['val']
    frame_height = frames.shape[1]
    #print('height',frame_height)
    frame_width = frames.shape[2]   

    nthreads = 128
    nnz = len(col)
    nblocks = nnz
    cp.RawKernel(zQQz_raw_kernel,"dotp",jitify=True,options=("--std=c++17",))\
    ((int(nblocks),),(int(nthreads),), \
    (value,frames,frames_norm,illumination,normalization,col,row,dx,dy,bw,nnz, frame_height, frame_width))
        
    print('same_kernel', xp.linalg.norm(value0-value))
    timers['Gramiam'] = timer() - t0
    nframes = frames.shape[0]
    print(type(col),col.dtype,type(row),row.dtype,type(value),value.dtype)
    H = sparse.coo_matrix((value.ravel(), (col, row)), shape=(nframes, nframes))
    #H = sparse.coo_matrix((xp.ones_like(value.ravel()), (col, row)), shape=(nframes, nframes))
    H += sparse.triu(H, k=1).conj().T
    H = H.tocsr()
    timers['Gramiam_completion']=timer() - t0
    import matplotlib.pyplot as plt
    plt.imshow(abs((H-H0).todense()).get())
    print("same?",xp.linalg.norm((H-H0).todense()))
    '''
        
    #print('initialize', value)
    #print('col',col,type(col),col.dtype, col.shape)
    #print('row',row,row.dtype,row.shape)
    #print('dx',type(dx),dx.dtype, dx.shape)
    #print('out',value)
    #print('value by Cuda',value)
    
    return H

def Gramiam_plan(translations_x, translations_y, nframes, nx, ny, Nx, Ny, bw=0):
    from scipy.spatial import KDTree

    # Overlap thresholds — match original: abs(dx)<ny-2bw, abs(dy)<nx-2bw
    thresh_dx = ny - 2 * bw
    thresh_dy = nx - 2 * bw

    # ── KD-tree neighbor search: O(N·k) memory, replaces the O(N²) dx/dy matrix ─
    # Pull translations to CPU numpy for the KD-tree (translations are tiny).
    tx = cp.asnumpy(translations_x.ravel()) if GPU else np.asarray(translations_x.ravel(), dtype=float)
    ty = cp.asnumpy(translations_y.ravel()) if GPU else np.asarray(translations_y.ravel(), dtype=float)
    points = np.column_stack([tx, ty])          # (nframes, 2)

    # 3×3 periodic image copies handle the toroidal (Nx × Ny) wrap-around.
    offsets = np.array([(sx * Nx, sy * Ny) for sx in (-1, 0, 1) for sy in (-1, 0, 1)])
    tiled       = np.vstack([points + off for off in offsets])   # (9·nframes, 2)
    tile_frame  = np.tile(np.arange(nframes), 9)                 # original frame index per tiled pt

    # Bounding-box (l∞) query to get candidate pairs, then exact rectangle filter.
    r_box = max(thresh_dx, thresh_dy)
    hits  = KDTree(points).query_ball_tree(KDTree(tiled), r=r_box, p=np.inf)

    # Vectorised flatten of hit lists → (all_i, all_k) index arrays.
    counts = np.array([len(h) for h in hits], dtype=np.intp)
    all_i  = np.repeat(np.arange(nframes), counts)
    all_k  = np.array([k for h in hits for k in h], dtype=np.intp)

    # dx_ij = tx[j]_wrap - tx[i]  (sign convention: tiled copy minus query point)
    dx_all = tiled[all_k, 0] - points[all_i, 0]
    dy_all = tiled[all_k, 1] - points[all_i, 1]

    keep   = (np.abs(dx_all) < thresh_dx) & (np.abs(dy_all) < thresh_dy)
    # row_np = matrix row index (standard), col_np = matrix col index (standard)
    row_np = all_i[keep].astype(np.int32)
    col_np = tile_frame[all_k[keep]].astype(np.int32)
    dx_np  = dx_all[keep].astype(np.int64)
    dy_np  = dy_all[keep].astype(np.int64)

    # The 3×3 periodic tiling can return the SAME (row, col) pair via several
    # image copies. The original dense-matrix path kept exactly ONE entry per
    # pair (the minimal/wrapped image, via wrap_boundary), so coo→csr never
    # merged entries. Left un-deduplicated, the duplicates (a) make H.data
    # (merged by tocsr) smaller than val2H's index bookkeeping expects →
    # IndexError, and (b) would double-count overlaps. De-duplicate per
    # (row, col), keeping the closest image (smallest l∞ |dx|,|dy|), to match
    # the pre-KD-tree Gramian exactly.
    _key   = row_np.astype(np.int64) * nframes + col_np.astype(np.int64)
    _linf  = np.maximum(np.abs(dx_np), np.abs(dy_np))
    _order = np.lexsort((_linf, _key))          # group by pair, closest image first
    _ks    = _key[_order]
    _first = np.empty(_order.size, dtype=bool)
    _first[0] = True
    _first[1:] = _ks[1:] != _ks[:-1]
    _sel   = _order[_first]
    row_np = row_np[_sel]; col_np = col_np[_sel]
    dx_np  = dx_np[_sel];  dy_np  = dy_np[_sel]

    # ── val2H needs the FULL symmetric (row, col) list ────────────────────────
    val2H = mapu2all(xp.array(row_np), xp.array(col_np), nframes)

    # ── TRIU pairs for gram_calc ───────────────────────────────────────────────
    # Original naming convention (from xp.where(xp.triu(...))):
    #   plan["col"] = matrix row index  (first xp.where output, ≤ plan["row"])
    #   plan["row"] = matrix col index  (second xp.where output)
    # dx_kernel = dx_matrix[plan["row"], plan["col"]]
    #           = tx[plan["col"]] - tx[plan["row"]]   (no-wrap sign)
    # From KD-tree: dx_np = tx[j]_wrap - tx[i], so dx_kernel = -dx_np.
    triu   = row_np <= col_np
    col_t  = xp.array(row_np[triu])           # plan["col"] = matrix row idx
    row_t  = xp.array(col_np[triu])           # plan["row"] = matrix col idx
    dx_t   = xp.array(-dx_np[triu], dtype=xp.int64)   # kernel expects long long int
    dy_t   = xp.array(-dy_np[triu], dtype=xp.int64)

    nnz = int(col_t.size)
    val = xp.zeros((nnz, 1), dtype=xp.complex64)

    plan = {
        "col": col_t.astype(int), "row": row_t.astype(int),
        "dx": dx_t, "dy": dy_t, "val": val, "bw": bw,
        "val2H": val2H, "gram_calc": None,
    }

    if GPU:
        nthreads = 128
        nblocks  = nnz
        def gram_calc(frames, frames_norm, illumination, normalization, value=val+0):
            cp.RawKernel(zQQz_raw_kernel, "dotp", jitify=True, options=("--std=c++17",))(
                (int(nblocks),), (int(nthreads),),
                (value, frames, frames_norm, illumination, normalization,
                 col_t.astype(int), row_t.astype(int), dx_t, dy_t, bw, nnz, nx, ny))
            return value
        plan["gram_calc"] = gram_calc

    return plan
    
def Precondition_calc(frames, bw=0):
    fw, fh = frames.shape[1:]
    t0 = timer()
    frames_norm = xp.linalg.norm(frames[:, bw : fw - bw, bw : fh - bw], axis=(1, 2)).astype(xp.complex64)

    return frames_norm


def Precondition(H, frames, bw=0):
    time0 = timer()
    fw, fh = frames.shape[1:]
    t0 = timer()
    frames_norm = xp.linalg.norm(frames[:, bw : fw - bw, bw : fh - bw], axis=(1, 2)).astype(xp.complex64)

    if GPU == False:
        D = sp.sparse.diags(1 / frames_norm)
    else: 
        t0 = timer()
        D = sparse.diags(1 / frames_norm , format='csr') #slow
        print('2',timer()-t0)
    t0 = timer()    
    H1 = D @ H @ D #slow

    timers["Precondition"] += timer() - time0 
    return H1, D

if GPU:
    from cupyx.scipy.sparse.linalg import eigsh
else:
    from scipy.sparse.linalg import eigsh

import os
# Sync eigensolver selection (env-overridable; default = committed power iteration).
#   SHARPY_SYNC=invit -> inverse iteration on the connection Laplacian
#   (Marchesini, arXiv:1209.4924 App. A, eq. A4): solves the BOTTOM of L=D-H
#   instead of power-iterating the near-degenerate TOP of the adjacency H.
SYNC_METHOD = os.environ.get("SHARPY_SYNC", "power").lower()
SYNC_EPS    = float(os.environ.get("SHARPY_SYNC_EPS", "1e-4"))    # must be < Fiedler gap (1-lambda2)
SYNC_STEPS  = int(os.environ.get("SHARPY_SYNC_STEPS", "1"))
SYNC_MODE   = os.environ.get("SHARPY_SYNC_MODE", "cg").lower()    # "cg" matrix-free inverse iter | "si" matrix-free shift-invert Lanczos | "direct" splu/spsolve
SYNC_TOL    = float(os.environ.get("SHARPY_SYNC_TOL", "1e-8"))
SYNC_SEED   = int(os.environ.get("SHARPY_SYNC_SEED", "1"))        # invit_seed: #cold syncs via invit before warm-power
SYNC_EIGTOL = float(os.environ.get("SHARPY_SYNC_EIGTOL", "1e-7")) # power-iteration GAP-AWARE stop tol (distance-to-eigenvector; 0 = full num_iter)

#######
####Eigensolver is causing problems, need implementation
#######
_eig_v0 = None  # cached dominant eigenvector, for power-iteration warm start
_sync_state = {"n": 0}  # invit_seed: count of sync calls so far (reset by eig_reset)
_invit_cache = {}  # invit: cached CSR row-index array (overlap-graph sparsity is scan-fixed)


def eig_reset():
    """Drop the cached eigenvector + seed counter (call between unrelated reconstructions)."""
    global _eig_v0
    _eig_v0 = None
    _sync_state["n"] = 0
    _invit_cache.clear()


def Eigensolver(H, num_iter, v0=None, tol=1e-7):
    # tol is the GAP-AWARE stopping tolerance (estimated distance-to-eigenvector,
    # NOT the raw step). Default 1e-7 reaches the deep sync floor at every frame
    # count tested (256/1024/4096) ~9x faster than running the full num_iter; a
    # looser 1e-6 floors a bit high at >=4096, 1e-8 costs ~full time. tol=0 disables
    # the early-out entirely. See the loop below + sharpy_delight_fig1_reproduction.
    global _eig_v0
    time0 = timer()

    nframes = xp.shape(H)[0]
    # print('nframes',nframes)
    # This eigenvector solve runs on BOTH backends: the power iteration's `H @ v`
    # and xp reductions work for cupyx and scipy sparse alike. It was previously
    # gated by `if GPU:`, which left `eigenvectors` unbound on CPU (UnboundLocalError)
    # and broke synchronize_frames_c there.
    if True:
        
        #print('IS H herm', H - H.transpose().conj())
        #use sparsity, and hermitian, use only triu
        #v0 = xp.ones((nframes),xp.complex64)
        #eigenvalues, eigenvectors = eigsh(H, k=1, ncv=3, maxiter=5, v0=v0, which="LM", tol=1e-3)
        #eigenvalues, eigenvectors = eigsh(H, k=1, ncv = 1, v0=v0,  which="LM", tol=0)
        #eigenvalues,eigenvectors = xp.linalg.eigh(H.todense())
        #print(xp.argmax(eigenvalues))
        #v0 = np.ones((nframes,1),np.complex64)
        #eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(H.get(), k=10, which="LM", v0=v0, tol=0)
        #eigenvalues, eigenvectors = sp.sparse.linalg.eigs(H.get(), k=64, which="LM", v0=v0, tol=0)
    
        
        '''
        eigenvectors = xp.ones((nframes,1),xp.complex64)
        for _ in range(num_iter):
            eigenvectors = H * eigenvectors
        '''
        
        ###complex 128
        #v0 = xp.ones((nframes,),xp.complex128)
        #v0 /= xp.linalg.norm(v0)
        #H1 = H.astype(xp.complex128)
        #eigenvalues, eigenvectors = eigsh(H1, k =2, which="LM", v0= v0, ncv = 6, tol=1e-6,return_eigenvectors = True) #
        #eigenvectors = eigenvectors.astype(xp.complex64)
        
        
        if H.size>20:
            '''
            v0 = xp.ones((nframes,),xp.complex64)
            eigenvalues, eigenvectors = eigsh(H, k =3, ncv = 9, v0 = v0, which="LM", return_eigenvectors = True) #
            '''
            #####
            ## When refining illuminations, fist few H are very small (maybe due to initialization 
            ## of illumination =0). Either change initialization or only sync after a stable 
            ## estimation of illumination found.
            ####
            # Power iteration (single-precision robust -- no Lanczos loss of
            # orthogonality, so no phase jumps in omega/|omega|). WARM-START from
            # the previous call's eigenvector (_eig_v0) when available, else ones:
            # the consensus eigenvector is ~constant-phase, so ones is already close
            # and the previous AP step is closer.
            if v0 is not None and v0.shape[0] == nframes:
                eigenvectors = xp.asarray(v0, dtype=xp.complex64).reshape(nframes, 1) + 0.0
            elif _eig_v0 is not None and _eig_v0.shape[0] == nframes:
                eigenvectors = _eig_v0 + 0.0
            else:
                eigenvectors = xp.ones((nframes, 1), xp.complex64)
            eigenvectors /= xp.linalg.norm(eigenvectors)
            # GAP-AWARE early-out. The naive step test (|v_n - v_{n-1}| < tol) stops
            # FAR too early on a near-degenerate overlap graph: the Fiedler gap
            # shrinks ~1/N, so consecutive iterates are a tiny step apart while the
            # eigenvector is still Fiedler-contaminated -> wrong per-frame gauge ->
            # the reconstruction FLOORS at large N (poster Fig 1.1). Power iteration
            # contracts geometrically at rate rho ~= lambda2/lambda1, so the step
            # shrinks like (1-rho)*distance_to_limit; the TRUE distance is therefore
            # ~step*rho/(1-rho). Estimate rho from the ratio of successive steps and
            # stop on that distance, NOT the raw step: rho->1 (tiny gap) keeps
            # iterating until the gauge is actually resolved, rho<<1 (well separated)
            # still stops in ~1-2 matvecs. tol=0 disables the early-out (full num_iter).
            prev_step = None
            for _ in range(num_iter):
                vn = H @ eigenvectors
                vn /= xp.linalg.norm(vn)
                step = float(xp.linalg.norm(vn - eigenvectors))
                eigenvectors = vn
                if step < 1e-12:                      # machine-exact (perfect warm start)
                    break
                if prev_step is not None and step < prev_step:
                    rho = step / prev_step            # observed contraction ~ lambda2/lambda1
                    if step * rho / (1.0 - rho) < tol:  # gap-aware distance-to-limit
                        break
                prev_step = step
            _eig_v0 = eigenvectors + 0.0          # cache for next AP iteration's warm start

        else:
            eigenvalues,eigenvectors = np.linalg.eigh((H.get() if GPU else H).todense())
            eigenvectors = xp.array(eigenvectors)
            _eig_v0 = xp.asarray(eigenvectors[:, -1]).reshape(nframes, 1) + 0.0
        
        #eigenvalues, eigenvectors = eigsh(H1 , k=2,ncv = 6, v0 = v0, maxiter = 10,which="LM", tol=1e-6,return_eigenvectors = True) # if dont specify starting point v0, converges to another eigenvector
        #eigenvalues,eigenvectors = np.linalg.eigh(H.get().todense()) #working
        #eigenvalues,eigenvectors = xp.linalg.eigh(H.todense()) #not working
        
         
    '''    
    else:
        print(type(H))
        v0 = xp.ones((nframes, 1),xp.complex64)
        eigenvalues, eigenvectors = eigsh(H, k=3, which="LM", tol=1e-9)
    
    #result by the eigsh
    #omega0 = xp.array(eigenvectors[:,-1])
    #so = xp.sign(omega1)[0] #substract the common phase for eigsh. The angle tends to jump between theta and -theta. so force the angle between [0,\pi]
    #so = xp.sign(xp.sum(omega1)) #common phase
    #omega1 /= so

    #result by power it
    #omega0 = eigenvectors[:, 0]
    #omega0 /= xp.linalg.norm(omega0) #normalize
    
    #correct for the blow up the difference for small magnitude
    #mask = xp.abs(omega1)< (1e-3 /nframes) #amp
    #omega1[mask] = 1
    #omega11 = omega1 / xp.abs(omega1) #blow up the difference for small magnitude
    #mask2 = (xp.abs(xp.angle(omega11)) < xp.pi/8)*1 #angle
    #omega11 *= -mask  + 1
    #omega11[mask] = 1
    
    
    #gradient decent for 10 steps
    w1 = eigenvectors[:,-1]
    so = xp.sign(xp.sum(w1)) #common phase
    w1 /= so
    w2 = eigenvectors[:,-2]
    so = xp.sign(xp.sum(w2)) #common phase
    w2 /= so
    w3 = eigenvectors[:,-3]
    so = xp.sign(xp.sum(w3))
    w3 /= so
    a1 = xp.sqrt(w1.size)/2 
    a2 = 0.01*a1* 1j
    ss = 0.01
    
    
    import matplotlib.pyplot as plt
    plt.imshow((xp.reshape(xp.abs(w1),(64,64))).get())
    plt.colorbar()
    plt.show()
    plt.imshow((xp.reshape(xp.abs(w2),(64,64))).get()) #in the hope that when abs(w1(i))~= 0, abs(w2(i))~= 1
    plt.colorbar()
    plt.show()
    plt.imshow((xp.reshape(xp.abs(w3),(64,64))).get()) #in the hope that when abs(w1(i))~= 0, abs(w2(i))~= 1
    plt.colorbar()
    plt.show()
    
    print(xp.linalg.norm(w1),xp.linalg.norm(w2))
    omega0 = a1 * w1 + a2 * w2
    print(xp.linalg.norm(omega0))
    print('decent?', evalf(a1,a2,w1,w2),a1,a2)
    
    for _ in range(5000):
        grad1,grad2 = gradientf(a1,a2,w1,w2)
        a1 = a1 - ss *grad1
        a2 = a2 - ss * grad2
    
    print('decent?', evalf(a1,a2,w1,w2),a1,a2)
    
    print(xp.linalg.norm(w1),xp.linalg.norm(w2))
    omega0 = a1 * w1 + a2 * w2
    print(xp.linalg.norm(omega0))
    
    plt.imshow((xp.reshape(xp.abs(omega0),(64,64))).get()) #in the hope that when abs(w1(i))~= 0, abs(w2(i))~= 1
    plt.colorbar()
    plt.show()
    
    plt.imshow((xp.reshape(xp.angle(omega0),(64,64))).get()) #in the hope that when abs(w1(i))~= 0, abs(w2(i))~= 1
    plt.clim(-0.02,0.06)
    plt.show()
    '''
    
    omega0 = eigenvectors[:,-1]
    #omega0 = eigenvectors[:,0]
    omega0 /= xp.linalg.norm(omega0) #normalize
  
    #random sign problem by eigsh
    '''
    so = xp.sign(xp.sum(omega0)) 
    omega0 /= so
    print(omega0)
    '''

    omega0 /= xp.abs(omega0)

    #common phase
    so = xp.conj(xp.sum(omega0)) 
    so /= xp.abs(so)
    omega0 *= so


    
    timers["Eigensolver"] += timer() - time0
    
    '''
    plt.imshow(abs(np.reshape(omega11.get()-omega0.get(),(64,64))))
    plt.colorbar()
    plt.show()
    plt.imshow(np.reshape(np.angle(omega11.get()),(64,64)))
    plt.colorbar()
    plt.show()
    plt.imshow(np.reshape(np.angle(omega0.get()),(64,64)))
    plt.colorbar()
    plt.show()
    '''
    # subtract the average phase
    #so = xp.conj(xp.sum(omega))
    #so /= abs(so)
    #omega *= so
    ########

    #omega = xp.reshape(omega11, (nframes, 1, 1))
    omega = xp.reshape(omega0, (nframes, 1, 1))
    return omega


def Eigensolver_invit(H, eps=1e-4, steps=1, tol=1e-8, mode="cg"):
    """Inverse-iteration sync eigensolver (Marchesini, arXiv:1209.4924 App. A, eq. A4).

    Drop-in for Eigensolver: returns the per-frame unit-modulus sync phase, shape
    (nframes,1,1). Instead of power-iterating the TOP of the normalized adjacency H
    (slow when the overlap-graph Fiedler gap is tiny / the true phase is far from
    constant), it solves the BOTTOM of the connection Laplacian
        L = diag(d) - H,   d_i = sum_j |H_ij|     (H is zero-diagonal by construction)
    by inverse iteration from ones, on the symmetric-normalized
        L_sym = I - D^-1/2 H D^-1/2  (eigenvalues in [0,2], sync vector at ~0):
        x <- (L_sym + eps*I)^{-1} x.
    The near-degenerate TOP gap of H is a near-ZERO BOTTOM eigenvalue of L, so one
    shifted solve lands the sync vector regardless of <ones,phi> -- exactly the
    regime where power iteration plateaus. eps shifts off the (singular) null
    direction (= Tikhonov / shift-invert at sigma=-eps); steps>=2 -> ~machine.
    Cost moves into one sparse solve: direct (splu/cuDSS) or AMG-preconditioned is
    the real win; an unpreconditioned iterative solve just relocates 1/gap.
    """
    global _eig_v0
    time0 = timer()
    nframes = H.shape[0]
    Hc = H.tocsr()
    # magnitude row-sum = connection-graph degree (H has zero diagonal: mapu2all
    # excludes the diagonal when assembling). The overlap-graph SPARSITY is fixed
    # by the scan, so cache the CSR row-index array once and refill the degree by
    # bincount each call (no |H| copy, no sparse allocation).
    key = (nframes, Hc.nnz)
    if _invit_cache.get("key") != key:
        _invit_cache["key"] = key
        # COO row indices of the fixed sparsity (cupy.repeat rejects array repeats)
        _invit_cache["rows"] = Hc.tocoo().row.astype(xp.int64)
    d = xp.bincount(_invit_cache["rows"], weights=xp.abs(Hc.data), minlength=nframes).real
    d = xp.maximum(d, 1e-30)
    s = (1.0 / xp.sqrt(d)).astype(H.dtype)

    if mode == "cg_asm" or mode == "direct":
        # assembled (Lsym + eps I): reference path / needed for the factorization
        dm12 = sparse.diags(s)
        Id = sparse.identity(nframes, dtype=H.dtype, format="csr")
        Lsym = (Id - (dm12 @ Hc @ dm12)).tocsr()
        M = (Lsym + eps * Id).tocsr()
    else:
        # mode="cg" (default): MATRIX-FREE. CG only needs matvecs, and
        #   (Lsym + eps I) x = (1+eps) x - s .* (H @ (s .* x)),
        # so skip the sparse triple product + CSR merges entirely -- the per-call
        # assembly WAS the dominant cost (100x larger eps cut CG matvecs ~10x but
        # wall-time only 1.5x). Each CG step now costs ONE H@v, same as a power step.
        if GPU:
            from cupyx.scipy.sparse.linalg import LinearOperator as _LO
        else:
            from scipy.sparse.linalg import LinearOperator as _LO
        one_eps = H.dtype.type(1.0 + eps)

        def _mv(x):
            x = x.ravel()
            return one_eps * x - s * (Hc @ (s * x))

        M = _LO((nframes, nframes), matvec=_mv, dtype=H.dtype)

    # M = Lsym + eps*I is Hermitian POSITIVE-DEFINITE. Solve by:
    #   mode="cg"     -> conjugate gradient (iterative, O(nnz)/matvec, NO fill-in;
    #                    ~sqrt(kappa)=~sqrt(1/eps) matvecs -> faster than power AND than a
    #                    direct factorization at scale -- benchmarked 169 mv / 47 ms vs
    #                    power's 2000 mv / 320 ms and splu's 467 ms at 4096 frames).
    #                    NOTE: scipy/cupyx MINRES rejects COMPLEX-Hermitian ("non-symmetric"
    #                    matrix"); CG handles complex Hermitian-PD -- use CG, not minres.
    #   mode="direct" -> splu (CPU) / spsolve (GPU): robust reference, but fill-in => super-linear.
    # eps must be SMALLER than the Fiedler gap (1-lambda2) to isolate the consensus from the
    # near-degenerate cluster (eps > gap -> under-amplified / wrong vector).
    # WARM-START from the previous call's vector: the sync target is non-stationary
    # (it develops with the image), but it DRIFTS slowly once formed, so seeding x0 from
    # _eig_v0 makes each per-AP-iter solve cheap (few CG iters) while still re-solving
    # accurately every iter. First call is cold (ones).
    if _eig_v0 is not None and _eig_v0.shape[0] == nframes:
        x = _eig_v0.ravel().astype(H.dtype) + 0.0
    else:
        x = xp.ones(nframes, dtype=H.dtype)
    x0 = x + 0.0
    if mode == "si":
        # MATRIX-FREE SHIFT-INVERT LANCZOS. Run eigsh over Minv = (Lsym+eps I)^{-1},
        # applied by CG (no factorization). The consensus mode -- the SMALLEST eigenvalue
        # of Lsym+eps I -- becomes the LARGEST of Minv, hence well separated, so Lanczos
        # targets it without the near-degenerate-cluster ambiguity of a direct eigsh(H).
        # This is the scipy sigma~0 / OPinv path made matrix-free, and it supplies BOTH
        # of cupyx eigsh's missing scipy features at once: it IS the shift-invert, and by
        # making the target dominant it removes the need for a v0 anchor (measured on the
        # A100: plain cupyx eigsh(H) align 0.18 -> shift-invert eigsh(Minv) align 1.000).
        # More robust than the CG inverse iteration when ones is a poor anchor (a strongly
        # varying phase far from consensus), at a few x the cost (Lanczos runs several
        # inner CG solves); use it as the robust fallback, not the cheap in-loop default.
        if GPU:
            from cupyx.scipy.sparse.linalg import cg as _cg, eigsh as _eigsh
            from cupyx.scipy.sparse.linalg import LinearOperator as _LOi
        else:
            from scipy.sparse.linalg import cg as _cg, eigsh as _eigsh
            from scipy.sparse.linalg import LinearOperator as _LOi

        def _solve(b):
            b = b.ravel()
            try:
                y, _ = _cg(M, b, rtol=tol, maxiter=20000)
            except TypeError:                     # older scipy/cupyx use tol= not rtol=
                y, _ = _cg(M, b, tol=tol, maxiter=20000)
            return y

        Minv = _LOi((nframes, nframes), matvec=_solve, dtype=H.dtype)
        ncv = int(min(nframes - 1, max(8, 2 * steps + 6)))
        if GPU:                                    # cupyx eigsh has no v0 (target is dominant)
            _lam, V = _eigsh(Minv, k=1, which="LM", ncv=ncv, maxiter=300)
        else:                                      # anchor scipy ARPACK at ones for determinism
            _lam, V = _eigsh(Minv, k=1, which="LM", ncv=ncv, maxiter=300,
                             v0=xp.ones(nframes, dtype=H.dtype))
        x = V[:, 0]
    elif mode == "direct":
        # inverse-power iteration via a REUSABLE factorization (cupyx/scipy splu): factor
        # (Lsym+epsI) ONCE, then a few inverse-iteration solves. This is the "shift-invert /
        # inverse-power" route -- the per-call cost is the factorization (cuSOLVER on GPU),
        # NOT the solves. (For matrix-free shift-invert Lanczos without a factorization, see
        # mode="si" above.)
        if GPU:
            from cupyx.scipy.sparse.linalg import splu as _splu
        else:
            from scipy.sparse.linalg import splu as _splu
        lu = _splu(M.tocsc())
        for _ in range(steps):
            x = lu.solve(x); x /= xp.linalg.norm(x)
    else:
        if GPU:
            from cupyx.scipy.sparse.linalg import cg as _cg
        else:
            from scipy.sparse.linalg import cg as _cg
        for _ in range(steps):
            try:
                x, _info = _cg(M, x, x0=x0, rtol=tol, maxiter=20000)
            except TypeError:                 # older scipy/cupyx use tol= not rtol=
                x, _info = _cg(M, x, x0=x0, tol=tol, maxiter=20000)
            x /= xp.linalg.norm(x); x0 = x + 0.0

    omega0 = x / xp.abs(x)                     # per-frame unit modulus = the gauge sync uses
    so = xp.conj(xp.sum(omega0)); so = so / xp.abs(so)   # fix global phase (match Eigensolver)
    omega0 = omega0 * so
    _eig_v0 = xp.reshape(omega0, (nframes, 1)) + 0.0   # seed power warm-start (for invit_seed mode)
    timers["Eigensolver"] += timer() - time0
    return xp.reshape(omega0, (nframes, 1, 1))


#def Eigensolver(H,eigsh_tol, eigsh_maxiter,power_it,power_iterations):
def Eigensolver_c(H,num_iter=5):
    time0 = timer()

    nframes = H.shape[1]
    
    eigenvectors = xp.random.rand(nframes,1,dtype = xp.float32) + 1j * xp.random.rand(nframes,1,dtype = xp.float32)
    for _ in range(num_iter):
        eigenvectors = H @ eigenvectors
        eigenvectors /= xp.linalg.norm(eigenvectors) #this is somehow required for numerical stability
    ########
    omega = xp.array(eigenvectors[:, 0])

    
    omega /= xp.linalg.norm(omega)
   
    timers["Eigensolver"] += timer() - time0

    # subtract the average phase
    omega0 = omega + 0 
    omega0 /= xp.abs(omega0)
    so = xp.conj(xp.sum(omega0))
    so /= abs(so)
    
    omega *= so
  
    ########
    omega = xp.reshape(omega0, (nframes, 1, 1))
    return omega

def mapu2all(row, col , nframes):

    # initialize sparse array. complex64 on GPU (matches the RawKernel output);
    # complex128 on CPU so the reference Gramian keeps full precision (eigsh is
    # unreliable in single precision -- power iteration is used on GPU instead).
    Hdtype = xp.complex64 if GPU else xp.complex128
    val0=xp.empty(col.size, dtype = Hdtype)
    Soo=sparse.coo_matrix((val0,(row,col)))
    H=Soo.tocsr()
    # The sync's H @ v matvecs (power iteration, invit CG, mode="si") are CSR
    # SpMV and hit the fast cuSPARSE / SciPy path only when H is in CANONICAL
    # format (sorted indices, no duplicates). tocsr() guarantees this (cupy #3430:
    # sum_duplicates + _has_canonical_format=True), and val2H below only rewrites
    # H.data in place -- it never touches indices/indptr, so the flag is preserved.
    # Assert it so a future change to the assembly can't silently revert to the
    # slow SpMV path.
    assert getattr(H, "_has_canonical_format", True), \
        "Gramian H must be canonical CSR for the fast H@v SpMV path (cupy #3430)"

    # split up upper and lower matrix indices
    iiu = xp.where(row <= col)[0]
    iil1 = xp.where(row > col)[0] # excluding diag
   
    # mapping index from upper to lower triangle
    idx = xp.arange(iiu.size)
    # exclude the diagonal  
    nd = xp.where(row[iiu] != col[iiu])
   
    # transpose the ordering
    ii=col[iiu[nd]]*nframes+row[iiu[nd]]
    u2l=idx[nd][xp.argsort(ii)]
    
    # combined index for assignment
    ii_fill=xp.concatenate((iiu,iil1))
    
 
    def val2H(valu):
        H.data[ii_fill] = xp.concatenate((valu, xp.conj(valu[u2l])))
     
        return H
    
    return   val2H


def synchronize_frames_c(frames, illumination, frames_norm, normalization, plan, bw=0,num_iter=5):
    # col,row,dx,dy=frames_overlap(translations_x,translations_y,nframes,nx,ny,Nx,Ny)
    # Gramiam = Gramiam_plan(translations_x,translations_y,nframes,nx,ny,Nx,Ny)
    
    time0 = timer()
    timers["Sync_setup"] += timer() - time0
    if GPU:     
        H = Gramiam_calc_cuda(frames,plan,illumination,normalization,frames_norm)

        #H = (H + H.transpose().conj())
        #print('Here is H', H.todense()) 
      
    else:
        framesl = Illuminate_frames(frames, xp.conj(illumination))
        framesr = framesl * normalization
        H = Gramiam_calc(framesl, framesr, plan,frames_norm)
    
    '''incorporated in the kernel
    if "Preconditioner" in plan:
        time0 = timer()
        # print('hello')
        D = plan["Preconditioner"]
        H1 = D @ H @ D
        timers["Precondition"] = timer() - time0
    else:
        H1, D = Precondition(H, frames, bw)
    '''
    
    # compute the largest eigenvalue of H1
    
    #omega = Eigensolver(H)
    #if type(eig_plan) == type(None):
        #omega = Eigensolver(H,eigsh_tol = 1e-6, eigsh_maxiter = None,power_it = False,power_iterations = 5)
 
    if SYNC_METHOD == "invit":
        omega = Eigensolver_invit(H, eps=SYNC_EPS, steps=SYNC_STEPS, tol=SYNC_TOL, mode=SYNC_MODE)
    elif SYNC_METHOD == "invit_seed":
        # invit for the first SYNC_SEED cold sync(s) (seeds the power warm-start cache),
        # then warm power (cheap drift tracking) thereafter.
        if _sync_state["n"] < SYNC_SEED:
            omega = Eigensolver_invit(H, eps=SYNC_EPS, steps=SYNC_STEPS, tol=SYNC_TOL, mode=SYNC_MODE)
        else:
            omega = Eigensolver(H, num_iter, tol=SYNC_EIGTOL)
        _sync_state["n"] += 1
    else:
        omega = Eigensolver(H, num_iter, tol=SYNC_EIGTOL)
    #omega = Eigensolver_c(H,num_iter)
    
    '''
    eig = sp.sparse.linalg.eigs(H.get())
    eigs = eig[0][0]
    eigv = eig[1][:,0]
    print(eigv.shape)
    print('HELLOOOOO', xp.linalg.norm( H.get() * eigv - eigs * eigv))
    print('11',xp.linalg.norm(eigv))
    '''
    '''
    else:
        tol = eig_plan['tol'] #tol for eig solver
        maxiter = eig_plan['maxiter'] #maxiter for eig solver
        power_it = eig_plan['power_it'] #boolean power iteration
        num_iterations = eig_plan['num_iterations'] #number of power iteration step
        print(tol,maxiter,power_it,num_iterations)
        #omega = Eigensolver(H,tol, maxiter,power_it,num_iterations)
        omega = Eigensolver(H)
    '''
    return omega

if GPU:
    from wrap_ops import refine_illumination_cuda
def synchronize_illum_c(nrm_illumination, frames,normalization, plan, num_iter=5):
    
    if GPU:     
        A = refine_illumination_cuda(frames,normalization, plan)

    omega = Eigensolver_c(A,num_iter)
    omega = xp.reshape(omega,(frames.shape[1],frames.shape[2]))
    omega *= nrm_illumination
    
    import matplotlib.pyplot as plt
    plt.imshow(abs(omega.get()))
    plt.show()
    return omega

# def synchronize_frames_plan(inormalization_split,Gramiam):
#    omega=lambda frames synchronize_frames_c(frames, illumination, inormalization_split, Gramiam)
#    Gramiam = lambda framesl,framesr: Gramiam_calc(framesl,framesr,nframes,col,row,nx,ny,dx,dy)
#    return Gramiam

#old version of refine_illumination
def refine_illumination_function(
    img, illumination,illumination_truth, frames,translations,Split, Overlap, GPU,lens_mask,i
):
    """
    refine_illumination based on

    Parameters
    ----------
    img : TYPE
        input image.
    illumination : TYPE
        initial illumination.
    frames : TYPE
        frames estimate.
    Split : TYPE
        Split operator.
    Overlap : TYPE
        overlap operator.
    lens_mask : TYPE, optional
        lens mask in F-space to remove grid pathology. The default is None.

    Returns
    -------
    illumination : TYPE
        refined illumination.
    normalization : TYPE
        refined normalization.

    """
    eps_illum = None #need implementation
    eps0 = xp.float32(1e-2)
    
    illumination0 = illumination + 0
    #global eps_illum
    if GPU:
        frames_split = xp.zeros(frames.shape,dtype = xp.complex64)
        Split(img + 0.0,frames_split,translations,0)
    else:    
        frames_split = Split(img)
    #norm_frames = xp.mean(xp.abs(frames_split) ** 2, 0) 
    if GPU:
        norm_frames = xp.zeros(frames.shape,dtype = xp.complex64) #dtype?
        Split(img * xp.conj(img),norm_frames,translations,0)
        norm_frames = xp.sum(norm_frames, 0) 

    if type(eps_illum) == type(None):
            eps_illum = xp.max(xp.abs(norm_frames)) * eps0 * 2**(-i)
            
    '''
    illumination = xp.sum(
        frames * xp.conj(Split(img)) + eps_illum * illumination, 0
    ) / (norm_frames + eps_illum)
    '''

    #illumination =  (xp.sum(frames * xp.conj(frames_split), 0) +  eps_illum * illumination0) / (norm_frames + eps_illum * xp.eye(frames.shape[1],frames.shape[2],dtype = frames.dtype))

    illumination =  (xp.sum(frames * xp.conj(frames_split), 0) +  eps_illum * illumination0) / (norm_frames + eps_illum * xp.eye(norm_frames.shape[0],norm_frames.shape[1], dtype = norm_frames.dtype))
 
    # apply mask to illumination
    if type(lens_mask) != type(None):
        illumination = xp.fft.fft2(illumination)
        illumination *= lens_mask
        illumination = xp.fft.ifft2(illumination)

    '''
    normalization = Overlap(
        Replicate_frame(xp.abs(illumination) ** 2, frames_split.shape[0])
    )  # check
    '''
    
    #normalize. Orthwise goes to inf
    #illumination /= xp.linalg.norm(illumination) #have same norm
    #illumination *= xp.linalg.norm(illumination_truth)

    #common phase
    #phase=xp.dot(illumination.conj().ravel(),illumination_truth.ravel())
    #phase /= xp.abs(phase)
    #illumination *= phase
    
    #return illumination, normalization
    return illumination


def refine_illumination_deflated(img, illumination, frames, mapid,
                                 beta=0.0, eps=1e-2, lens_mask=None,
                                 residual=True):
    """Probe (illumination) update with optional average-transparency deflation
    (eq.16 of Marchesini & Wu, arXiv:1408.1922; MATLAB FPoverlap_x.m,
    Fix_probe_intrnl*), in two variants.

    The plain update (beta=0) is the regularized least-squares probe step
    (same as refine_illumination_function):

        illum = (sum_n frames_n conj(O_n) + eps*illum0) / (sum_n |O_n|^2 + eps)

    where O_n = Split(img) are the current object frames. Deflation subtracts the
    scalar average transparency  mm * illum  before the update.

    residual=True: mm is taken from the RESIDUAL z - O*P:
        mm = <illum, frames - O*illum> / (nframes * ||illum||^2)
      The residual -> 0 at the solution, so the deflation SELF-CANCELS at the
      fixed point -> NO bias/plateau (reaches the same floor as plain eq.7). BUT
      it also provides essentially NO acceleration: empirically beta>0 is
      identical to beta=0 here -- the leftover term lies along the probe's own
      scale (the scalar gauge / object update already absorb it). KEY LESSON: the
      bias and the speedup of z-deflation are the SAME mechanism (subtracting a
      transparency that is NONZERO at the solution); remove the bias and you
      remove the speedup. So residual=True ~= plain eq.7. Default for safety.

    residual=False (the original eq.16): mm from z itself:
        mm = <illum, frames> / (nframes * ||illum||^2)
      mm*illum is NONZERO at the solution -> biases the probe fixed point -> the
      reconstruction PLATEAUS above the plain-eq.7 floor (and a fixed beta
      re-grows the mode and diverges; only an ANNEALED beta is usable, and it
      still plateaus). Gives an early-iteration head start (useful only if you
      stop early). Kept for reference / reproducing the paper's behaviour.

    CPU/GPU agnostic.

    Parameters
    ----------
    img : (Nx, Ny) complex          current image (object) estimate
    illumination : (nx, ny) complex current probe estimate
    frames : (nframes, nx, ny)      current exit-wave estimate (post data proj.)
    mapid : frame<->image index map (from map_frames)
    beta : float                    deflation strength this call (0 = plain eq.7)
    eps : float                     LS regularization (relative to max |O|^2)
    lens_mask : (nx, ny) or None    optional Fourier aperture constraint
    residual : bool                 deflate the residual (self-cancelling) vs z
    """
    object_frames = Splitc(img, mapid)
    norm_frames = xp.sum(xp.abs(object_frames) ** 2, 0)
    eps_illum = xp.max(xp.abs(norm_frames)) * eps

    z = frames
    if beta:
        nframes = frames.shape[0]
        # average transparency: from the residual (self-cancelling) or from z
        src = z - object_frames * illumination[xp.newaxis] if residual else z
        mm = xp.sum(xp.conj(illumination)[xp.newaxis] * src) / (
            nframes * xp.sum(xp.abs(illumination) ** 2) + 1e-30
        )
        z = z - beta * mm * illumination[xp.newaxis]

    illum = (xp.sum(z * xp.conj(object_frames), 0) + eps_illum * illumination) / (
        norm_frames + eps_illum
    )

    if lens_mask is not None:
        illum = xp.fft.ifft2(xp.fft.fft2(illum) * lens_mask)
    return illum


if GPU:
    from wrap_ops import overlap_cuda,split_cuda
#refine illumination based on pairwise relationship between frames
def refine_illumination_pairwise(
    img, illumination_start, illumination_truth, frames, translations, split_cuda, overlap_cuda, lens_mask=None
):
    """
    refine_illumination based on

    Parameters
    ----------
    img : TYPE
        input image.
    illumination : TYPE
        initial illumination.
    frames : TYPE
        frames estimate.
    Split : TYPE
        Split operator.
    Overlap : TYPE
        overlap operator.
    lens_mask : TYPE, optional
        lens mask in F-space to remove grid pathology. The default is None.

    Returns
    -------
    illumination : TYPE
        refined illumination.
    normalization : TYPE
        refined normalization.

    """

    # eps_illum = None
    global eps_illum
    img0 = img * 0 
    frames0 = frames * 0 
    nframes = frames.shape[0]
    
    #matrix D^(-1/2)
    frames_norm = (xp.abs(frames) ** 2).astype(xp.complex64)
    nl = xp.sum(split_cuda(overlap_cuda(img0, frames_norm, translations, illumination_truth * 0 + 1), frames0, translations,0),axis = 0)
    D_inv = 1 / (xp.sqrt(nl) + 1e-8)
    
    
    #Define D^(-1/2)HD^(-1/2) as an operator
    #@xp.fuse(kernel_name="DHD")
    def DHD(a,img,frames):
        img0 = img * 0
        frames0 = frames * 0
        overlap_cuda(img0, frames.conj()* Replicate_frame(D_inv * a, nframes),translations,illumination_truth *0 + 1)
        split_cuda(img0, frames0,translations,0)
        b = D_inv * xp.sum(frames * frames0,axis = 0)
        return b
    
    tolerance = 1e-6
    #solve illumination by power iteration    
    w = illumination_start + 0
    for _ in range(1000):
        w_new = DHD(w,img,frames)
    
        # Check for convergence
        if xp.linalg.norm(w_new - w) < tolerance:
            print("Convergence reached.")
            break
    
        w = w_new
        
        
    #transform back
    illumination = D_inv * w
    #normalize. Orthwise goes to inf

    
    #illumination /= xp.linalg.norm(illumination) #have same norm
    #illumination *= xp.linalg.norm(illumination_truth)

    
    #common phase
    #phase=xp.dot(illumination.conj().ravel(),illumination_truth.ravel())
    #phase /= xp.abs(phase)
    #illumination *= phase
    #not necessary
    
    # apply mask to illumination
    if type(lens_mask) != type(None):
        illumination = xp.fft.fft2(illumination)
        illumination *= lens_mask
        illumination = xp.fft.ifft2(illumination)

    return illumination


def mse_calc(img0, img1):
    # calculate the MSE between two images after global phase correction
    
    nnz = xp.size(img0)
    # compute the best phase
    phase = xp.dot(xp.reshape(xp.conj(img1), (1, nnz)), xp.reshape(img0, (nnz, 1)))[
        0, 0
    ]
    phase = phase / xp.abs(phase)
    # compute norm after correcting the phase
    mse = xp.linalg.norm(img0 - img1 * phase)
    #compute the best phase and scalar
    #phase = xp.conj(phase) / xp.linalg.norm(img0)**2
    #mse = xp.linalg.norm(img0 - img1 / phase)
    return mse

def common_scale(img0,img1):
    scale = xp.dot(img0.ravel(),img1.ravel()) / xp.dot(img0.ravel(),img0.ravel())   
    return scale

import ctypes
from multiprocessing import sharedctypes


def shared_array(shape=(1,), dtype=np.float32):
    np_type_to_ctype = {
        np.float32: ctypes.c_float,
        np.float64: ctypes.c_double,
        np.bool: ctypes.c_bool,
        np.uint8: ctypes.c_ubyte,
        np.uint64: ctypes.c_ulonglong,
        np.complex128: ctypes.c_double,
        np.complex64: ctypes.c_float,
    }

    numel = np.int(np.prod(shape))
    iscomplex = dtype == np.complex128 or dtype == xp.complex64
    # numel *=
    arr_ctypes = sharedctypes.RawArray(np_type_to_ctype[dtype], numel * (1 + iscomplex))
    np_arr = np.frombuffer(arr_ctypes, dtype=dtype, count=numel)
    np_arr.shape = shape

    return np_arr


def circular_aperture(radius,img):
    aperture = xp.zeros_like(img)
    center = (img.shape[0]//2,img.shape[1]//2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if xp.sqrt((i - center[0])**2 + (j - center[1])**2) <= radius:
                aperture[i,j] = 1              
    return aperture

def gradientf(a1,a2,w1,w2):
    #this function calculate the gradient of f(a1,a2,w1,w2)=\| abs(a1w1+a2w2) - ones \|^2
    #a1,a2 are two complex scalrs, w1 w2 are top two orthonormal complex vectors
    
    denom = xp.abs(a1*w1 +a2*w2)
    alpha11 = xp.sum(xp.abs(w1)**2/denom)
    alpha12 = xp.sum(xp.conj(w1) * w2/denom)
    
    alpha21 = xp.sum(xp.abs(w2)**2/denom)
    alpha22 = xp.sum(xp.conj(w2) * w1/denom)
    
    #gradient w.r.t a1
    grad1 = a1 - a1*alpha11  - a2 * alpha12
    grad2 = a2 - a2*alpha21  - a1 * alpha22
    
    return grad1, grad2

def evalf(a1,a2,w1,w2):
    #evaluate f(a1,a2,w1,w2)=\| abs(a1w1+a2w2) - ones \|^2
    return xp.linalg.norm(xp.abs(a1*w1+a2*w2) - xp.ones(w1.shape))**2