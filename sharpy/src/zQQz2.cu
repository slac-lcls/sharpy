#include "cupy/complex.cuh"
// Portable bundled-CUB include -- see the long note in zQQz.cu. cupy 13.x moved the vendored
// CCCL to cupy/_cccl/cub/...; the earlier CUDA-12 fix here pinned the <=12.x path, which now
// fails on cupy 13.6.0.
#if defined(__has_include)
#  if __has_include(<cupy/_cccl/cub/cub/block/block_reduce.cuh>)
#    include <cupy/_cccl/cub/cub/block/block_reduce.cuh>
#  elif __has_include(<cupy/cub/cub/block/block_reduce.cuh>)
#    include <cupy/cub/cub/block/block_reduce.cuh>
#  else
#    include <cub/block/block_reduce.cuh>
#  endif
#else
#  include <cupy/cub/cub/block/block_reduce.cuh>
#endif

/*
 * Generalized zQQz: left/right illumination.
 *
 * Same neighbor-overlap inner product as zQQz.cu's dotp(), but the left and
 * right sides of the product carry DIFFERENT illuminations.  This computes
 * the coupled position-retrieval blocks O11/O22/Ox of Eq. (27) of
 * arXiv:1209.4924 without materializing the frame-sized products
 * z*conj(px), z*conj(py), (...)*QQinv -- the probes are applied per pixel.
 *
 * Call with (illum_left, illum_right):
 *     O11 <- (px, px)
 *     O22 <- (py, py)
 *     Ox  <- (px, py)
 *
 * Unlike zQQz.cu (single 2D probe indexed within-frame), illum_left and
 * illum_right are FULL frame-sized stacks (same shape as `frames`), because
 * the position-retrieval probe is shifted per frame.  They are indexed by
 * the global frame-stack index (ii1 / ii2), exactly like `frames`.
 * `normalization` is the split QQinv (frame-sized), indexed at ii2.
 *
 * frames_norm divides the result (D@H@D preconditioning); pass ones to get
 * the raw entries and precondition in Python.
 *
 * NOTE: written to mirror _pair_overlaps_fused in position_retrieval.py
 * (the validated CPU reference); needs testing on GPU hardware.
 */
extern "C" __global__ void
dotp2(
    thrust::complex<float> * value,
    thrust::complex<float> * frames,
    thrust::complex<float> * frames_norm,
    thrust::complex<float> * illum_left,
    thrust::complex<float> * illum_right,
    thrust::complex<float> * normalization,
    size_t * col,
    size_t * row,
    long long int * dx,
    long long int * dy,
    int bw,
    int nnz,
    int frame_height, int frame_width) {

        typedef cub::BlockReduce<thrust::complex<float>, 128> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;

        int ii = blockIdx.x;
        if (ii >= nnz) return;

        int col00 = col[ii];
        int row00 = row[ii];

        thrust::complex<float> dd1 = frames_norm[col[ii]];
        thrust::complex<float> dd2 = frames_norm[row[ii]];

        long long int Dx = frame_width  - abs(dx[ii]) - 2*bw; /*integration width */
        long long int Dy = frame_height - abs(dy[ii]) - 2*bw; /*integration height*/

        long long int DD = col00 * frame_height * frame_width
            + (-dx[ii] + abs(dx[ii])) / 2
            + (-dy[ii] + abs(dy[ii])) / 2 * frame_width
            + bw * (1 + frame_width);                          /*row-wise*/
        long long int Dij = dx[ii] + dy[ii] * frame_width
            + (row00 - col00) * frame_height * frame_width;    /*row-wise*/

        thrust::complex<float> Sum0 = 0;
        size_t ii1, ii2;

        for (size_t pos = threadIdx.x; pos < (Dx * Dy); pos += blockDim.x) {
            ii1 = pos / Dx * frame_width + pos % Dx + DD;      /*left  pixel*/
            ii2 = ii1 + Dij;                                    /*right pixel*/

            /* conj(frames_l) * illum_left_l * frames_r * conj(illum_right_r) * norm_r
             * == conj(frames_l * conj(illum_left_l)) * (frames_r * conj(illum_right_r) * norm_r) */
            Sum0 += thrust::conj(frames[ii1]) * illum_left[ii1]
                  * frames[ii2] * thrust::conj(illum_right[ii2])
                  * normalization[ii2];
        }

        thrust::complex<float> Sum1 = BlockReduce(temp_storage).Sum(Sum0);

        if (threadIdx.x == 0)
            value[ii] = Sum1 / (dd1 * dd2);
        }
