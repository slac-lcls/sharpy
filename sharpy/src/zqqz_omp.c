/*=================================================================
 * zQQz on CPU with OpenMP.
 *
 * CPU twin of src/zQQz.cu: computes the Gramian entries -- the
 * illumination-weighted <bra|ket> inner product of overlapping frame
 * pairs. The CUDA "block per pair + threads per pixel + BlockReduce"
 * becomes here "omp parallel for over pairs + a serial inner sum over the
 * overlap". Line-for-line port of position_retrieval / Operators
 * _braket_val_numba (the validated Numba reference).
 *
 * Build (parallel, Perlmutter / Linux gcc):
 *   gcc -O3 -fopenmp -shared -fPIC -o zqqz_omp.so zqqz_omp.c
 * Build (serial fallback, e.g. Apple clang without libomp):
 *   cc  -O3 -shared -fPIC -Wno-unknown-pragmas -o zqqz_omp.so zqqz_omp.c
 *
 * NumPy complex128 is laid out as interleaved (re, im) doubles, i.e. C99
 * `double _Complex`, so the arrays can be passed straight through.
 *=================================================================*/
#include <complex.h>
#include <stdlib.h>

/* out[ii] = <bra(framesl[col]) | ket(framesr[row])> / (norm[col] norm[row])
 *   bra = overlap window of the LEFT  frame, shifted (-dx,-dy)
 *   ket = overlap window of the RIGHT frame, shifted (+dx,+dy)
 * Square frames (nx == ny), matching ket(). */
void zqqz_braket(
        const double _Complex *framesl,
        const double _Complex *framesr,
        const long *col,
        const long *row,
        const long *dx,
        const long *dy,
        int bw,
        long nnz,
        int nx,
        const double *frames_norm,
        double _Complex *out)
{
    const long fsz = (long)nx * (long)nx;   /* pixels per frame */

    #pragma omp parallel for schedule(static)
    for (long ii = 0; ii < nnz; ii++) {
        const long c = col[ii];
        const long r = row[ii];
        const long dxi = dx[ii];
        const long dyi = dy[ii];

        const long bra_r0 = (dyi < 0 ? -dyi : 0) + bw;   /* max(0,-dy)+bw */
        const long bra_c0 = (dxi < 0 ? -dxi : 0) + bw;
        const long ket_r0 = (dyi > 0 ?  dyi : 0) + bw;   /* max(0, dy)+bw */
        const long ket_c0 = (dxi > 0 ?  dxi : 0) + bw;
        const long hgt = nx - labs(dyi) - 2 * bw;
        const long wid = nx - labs(dxi) - 2 * bw;

        const double _Complex *fl = framesl + c * fsz;
        const double _Complex *fr = framesr + r * fsz;

        double _Complex acc = 0.0;
        for (long a = 0; a < hgt; a++) {
            const double _Complex *lrow = fl + (bra_r0 + a) * (long)nx + bra_c0;
            const double _Complex *rrow = fr + (ket_r0 + a) * (long)nx + ket_c0;
            for (long b = 0; b < wid; b++) {
                acc += conj(lrow[b]) * rrow[b];   /* <bra|ket> */
            }
        }
        out[ii] = acc / (frames_norm[c] * frames_norm[r]);
    }
}
