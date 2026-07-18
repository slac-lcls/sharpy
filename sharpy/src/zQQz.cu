#include "cupy/complex.cuh"
//#include <thrust/transform.h>
//#include "thrust/complex.h"
//#include <iostream>
//#include <cub/cub.cuh> 
//#include <cupy/cub/cub/cub.cuh>
//#include <cub/block/block_reduce.cuh>
#include <cupy/cub/cub/block/block_reduce.cuh>

extern "C" __global__ void 
dotp(
    thrust::complex< float > * value,
    thrust::complex< float> * frames,
    //thrust::complex< float> * framesl,
    //thrust::complex< float> * framesr,
    thrust::complex< float> * frames_norm,
    thrust::complex< float> * illumination,
    thrust::complex< float> * normalization,
    size_t * col,
    size_t * row,
    long long int * dx,
    long long int * dy,
    int bw,
    int nnz,
    int frame_height, int frame_width) {
        
        // Reduce the real and imaginary parts separately with the (well-supported)
        // float specialization of cub::BlockReduce. The complex specialization
        // cub::BlockReduce<thrust::complex<float>> produced value/run-state-dependent
        // NaNs (mishandled custom-type temporaries in the block reduction) on small,
        // heavily-overlapped scenes -- see the small-image GPU sync NaN task.
        typedef cub::BlockReduce< float , 128> BlockReduce;


        // Allocate shared memory for BlockReduce
        __shared__ typename BlockReduce::TempStorage temp_storage;
        
        
        int ii = blockIdx.x ;
        if (ii >= nnz) return;

        int col00 = col[ii];
        int row00 = row[ii];
        int shiftl = col00 * frame_height * frame_width;
        int shiftr = row00 * frame_height * frame_width;
        
        thrust::complex< float> dd1 = frames_norm[col[ii]];
        thrust::complex< float> dd2 = frames_norm[row[ii]];
        
        
       /*
        if (blockIdx.x == 1 && threadIdx.x == 0) {
            printf("col: %d, row: %d\n", col00, row00);
           }
       */
       
        long long int Dx = frame_width - abs(dx[ii]) - 2*bw; /*integration width */
        long long int Dy = frame_height - abs(dy[ii]) - 2*bw; /*integration height */
      
       
        /*row-wise*/
        /* offset, including row */
        long long int DD = col00 * frame_height * frame_width  + (-dx[ii] + abs(dx[ii])) / 2 + 
        (-dy[ii] + abs(dy[ii]))/2 * frame_width + bw*(1 + frame_width); /*row-wise*/
        //long long int DD = col00 * frame_height * frame_width  + (-dx[ii] + abs(dx[ii])) / 2 * frame_height  + (-dy[ii] + abs(dy[ii]))/2; /*column-wise*/
        /* offset between frame1 and frame2 */
        long long int Dij = dx[ii] + dy[ii] * frame_width+ (row00 - col00) * frame_height * frame_width ; /*row-wise*/
        //long long int Dij = dx[ii] * frame_height + dy[ii] + (row00 - col00) * frame_height * frame_width ; /*column-wise*/
       
       /*
        if (blockIdx.x == 1 && threadIdx.x == 0) {
        printf("ii: %d\n", static_cast<int>(ii));
        printf("Dx: %lld, Dy: %lld\n", Dx, Dy);
        printf("DD: %lld, Dij: %lld\n", DD, Dij);
        printf("dx: %lld, dy: %lld\n", dx[ii], dy[ii]);
    }
    */
        thrust::complex<float> Sum0 = 0;
        size_t ii1, ii2, ii3, ii4;
            
        /* loop within frame overlap */
        for (size_t pos = threadIdx.x; pos < (Dx * Dy); pos += blockDim.x){
                //ii1 = pos / Dy * frame_height + pos % Dy + DD ; /*column-wise*/
                ii1 = pos / Dx * frame_width + pos % Dx + DD ; /*row-wise*/
                ii2 = ii1 + Dij;
                ii3 = ii1 - shiftl;
                ii4 = ii2 - shiftr;
                
                /*
                if (blockIdx.x == 1 && threadIdx.x < 3){
                    printf("ii1: %d, ii2 : %d\n", static_cast<int>(ii1), static_cast<int>(ii2));
                    printf("ii3: %d, ii3 : %d\n", static_cast<int>(ii3), static_cast<int>(ii4));
                    printf("framesl: %f + %fi\n", thrust::real(frames[ii1]), thrust::imag(frames[ii1]));
                    printf("framesr: %f + %fi\n", thrust::real(frames[ii2]), thrust::imag(frames[ii2]));
                    printf("normalization: %f + %fi\n", thrust::real(normalization[ii2]), thrust::imag(normalization[ii2]));
                    printf("illuminationl: %f + %fi\n", thrust::real(illumination[ii3]), thrust::imag(illumination[ii3]));
                    printf("illuminationr: %f + %fi\n", thrust::real(illumination[ii4]), thrust::imag(illumination[ii4]));
                    printf("dd: %f + %f\n", dd1, dd2);
                    }
                
                */
                if(illumination){
                    Sum0 += thrust::conj(frames[ii1]) * illumination[ii3] * frames[ii2] * thrust::conj(illumination[ii4]) * normalization[ii2]; }
                else{   
                    Sum0 += thrust::conj(frames[ii1])  * frames[ii2] * normalization[ii2]; 
                }

                }
      
        // Compute the block-wide sum for thread0 (real and imaginary parts
        // separately; __syncthreads between the two reuses of the shared
        // TempStorage, as cub::BlockReduce requires).
        float Sum_r = BlockReduce(temp_storage).Sum(Sum0.real());
        __syncthreads();
        float Sum_i = BlockReduce(temp_storage).Sum(Sum0.imag());
        thrust::complex< float >  Sum1(Sum_r, Sum_i);

        /*we know it is hermitian*/
        if (col00 == row00)
           Sum1.imag(0.0f);
        
        if (threadIdx.x == 0)
            //value[ii] = Sum1; 
            value[ii] = Sum1/(dd1 * dd2); 
        }



/* dotp_fetch: SM's fetch-normalization variant (2026-07-18). Identical math and reduction order to dotp,
   with ONE change: the normalization is read from the OBJECT-SIZED canvas `nu` at the absolute pixel
   (per-frame origins ax0/ay0, decoded from the same map_frames convention Splitc uses) instead of from a
   frame-sized Splitc stack -- eliminating the ~overlap-factor redundant copy (e.g. 537 MB -> 19 MB at
   4096 x 128^2). The toroidal wrap uses compare-subtract, not %, because 64-bit integer division is the
   expensive op on GPU (measured: % costs 1.5x total kernel time; cond-sub costs ~1.1x). Bit-exact vs
   dotp: same complex64 values loaded, same flat-pos loop order (validated in sync_fetch_test.py). */
extern "C" __global__ void
dotp_fetch(
    thrust::complex< float > * value,
    thrust::complex< float> * frames,
    thrust::complex< float> * frames_norm,
    thrust::complex< float> * illumination,
    const thrust::complex< float> * nu,     /* object-sized canvas, row stride canW */
    const long long * ax0,                  /* per-frame fast-axis (x) origin in canvas */
    const long long * ay0,                  /* per-frame slow-axis (y) origin in canvas */
    size_t * col,
    size_t * row,
    long long int * dx,
    long long int * dy,
    int bw,
    int nnz,
    int frame_height, int frame_width,
    int canH, int canW) {

        typedef cub::BlockReduce< float , 128> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;

        int ii = blockIdx.x ;
        if (ii >= nnz) return;

        int col00 = col[ii];
        int row00 = row[ii];
        int shiftl = col00 * frame_height * frame_width;
        int shiftr = row00 * frame_height * frame_width;

        thrust::complex< float> dd1 = frames_norm[col[ii]];
        thrust::complex< float> dd2 = frames_norm[row[ii]];

        long long int Dx = frame_width - abs(dx[ii]) - 2*bw;
        long long int Dy = frame_height - abs(dy[ii]) - 2*bw;

        /* overlap-window corner in frame col00's local coords (same decomposition as DD in dotp) */
        long long int lx0 = (-dx[ii] + abs(dx[ii])) / 2 + bw;
        long long int ly0 = (-dy[ii] + abs(dy[ii])) / 2 + bw;
        long long int Dij = dx[ii] + dy[ii] * frame_width + (row00 - col00) * frame_height * frame_width;

        long long int fx0 = ax0[col00];
        long long int fy0 = ay0[col00];

        thrust::complex<float> Sum0 = 0;
        size_t ii1, ii2, ii3, ii4, iin;

        for (size_t pos = threadIdx.x; pos < (size_t)(Dx * Dy); pos += blockDim.x){
                long long int ly = ly0 + (long long int)(pos / Dx);
                long long int lx = lx0 + (long long int)(pos % Dx);
                ii3 = (size_t)(ly * frame_width + lx);
                ii1 = ii3 + shiftl;
                ii2 = ii1 + Dij;
                ii4 = ii2 - shiftr;
                /* absolute canvas pixel; origins < can, locals < frame <= can => at most one wrap */
                long long int ay = fy0 + ly; if (ay >= canH) ay -= canH;
                long long int ax = fx0 + lx; if (ax >= canW) ax -= canW;
                iin = (size_t)(ay * canW + ax);

                if(illumination){
                    Sum0 += thrust::conj(frames[ii1]) * illumination[ii3] * frames[ii2] * thrust::conj(illumination[ii4]) * nu[iin]; }
                else{
                    Sum0 += thrust::conj(frames[ii1])  * frames[ii2] * nu[iin];
                }

                }

        float Sum_r = BlockReduce(temp_storage).Sum(Sum0.real());
        __syncthreads();
        float Sum_i = BlockReduce(temp_storage).Sum(Sum0.imag());
        thrust::complex< float >  Sum1(Sum_r, Sum_i);

        /*we know it is hermitian*/
        if (col00 == row00)
           Sum1.imag(0.0f);

        if (threadIdx.x == 0)
            value[ii] = Sum1/(dd1 * dd2);
        }
