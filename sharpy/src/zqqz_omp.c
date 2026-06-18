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


/* Coupled / generalized <bra|ket> (C analog of zQQz2.cu and the Numba
 * _braket_coupled_numba): left and right frames weighted by DIFFERENT probes
 * pL, pR and a normalization qq, both pair orientations. Used for the
 * off-diagonal position-retrieval blocks O11/O22/Ox. For pair ii=(a,b):
 *   ab[ii] = sum conj(frames_a conj(pL_a)) (frames_b conj(pR_b) qq_b)
 *   ba[ii] = sum conj(frames_b conj(pL_b)) (frames_a conj(pR_a) qq_a)  */
void zqqz_braket_coupled(
        const double _Complex *frames,
        const double _Complex *pL,
        const double _Complex *pR,
        const double _Complex *qq,
        const long *col,
        const long *row,
        const long *dx,
        const long *dy,
        int bw,
        long nnz,
        int nx,
        double _Complex *ab,
        double _Complex *ba)
{
    const long fsz = (long)nx * (long)nx;

    #pragma omp parallel for schedule(static)
    for (long ii = 0; ii < nnz; ii++) {
        const long a = col[ii];
        const long b = row[ii];
        const long dxi = dx[ii];
        const long dyi = dy[ii];

        const long wn_r = (dyi < 0 ? -dyi : 0) + bw;   /* "-shift" window (frame a) */
        const long wn_c = (dxi < 0 ? -dxi : 0) + bw;
        const long wp_r = (dyi > 0 ?  dyi : 0) + bw;   /* "+shift" window (frame b) */
        const long wp_c = (dxi > 0 ?  dxi : 0) + bw;
        const long hgt = nx - labs(dyi) - 2 * bw;
        const long wid = nx - labs(dxi) - 2 * bw;

        const double _Complex *fa = frames + a * fsz, *fb = frames + b * fsz;
        const double _Complex *pLa = pL + a * fsz, *pLb = pL + b * fsz;
        const double _Complex *pRa = pR + a * fsz, *pRb = pR + b * fsz;
        const double _Complex *qa = qq + a * fsz, *qb = qq + b * fsz;

        double _Complex sab = 0.0, sba = 0.0;
        for (long i = 0; i < hgt; i++) {
            const long la = (wn_r + i) * (long)nx + wn_c;   /* frame a window row */
            const long rb = (wp_r + i) * (long)nx + wp_c;   /* frame b window row */
            for (long j = 0; j < wid; j++) {
                const double _Complex Fa = fa[la + j];
                const double _Complex Fb = fb[rb + j];
                sab += conj(Fa) * pLa[la + j] * Fb * conj(pRb[rb + j]) * qb[rb + j];
                sba += conj(Fb) * pLb[rb + j] * Fa * conj(pRa[la + j]) * qa[la + j];
            }
        }
        ab[ii] = sab;
        ba[ii] = sba;
    }
}
