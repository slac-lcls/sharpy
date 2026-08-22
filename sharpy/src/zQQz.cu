#include "cupy/complex.cuh"
//#include <thrust/transform.h>
//#include "thrust/complex.h"
//#include <iostream>
//#include <cub/cub.cuh> 
//#include <cupy/cub/cub/cub.cuh>
//#include <cub/block/block_reduce.cuh>
// Bundled-CUB include, portable across cupy majors. cupy 13.x relocated the vendored CCCL to
// cupy/_cccl/cub/...; <=12.x used cupy/cub/...; a system cub is a third possibility. Hardcoding
// any one of them breaks the others, and the failure is a hard NVRTC "catastrophic error: cannot
// open source file" that takes the WHOLE module down -- so EVERY kernel in this file stops
// compiling, and the traceback surfaces at whichever caller happened to run first, which makes
// it look like a bug in the caller rather than a missing header.
// Measured on cupy 13.6.0 / H200 NVL (drp-srcf-gpu007, 2026-08-21): compiles with jitify on and off.
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


