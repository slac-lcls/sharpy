#include "cupy/complex.cuh"
//#include "thrust/complex.h"
//#include <iostream>
//#include <cub/cub.cuh> 
//#include <cupy/cub/cub/cub.cuh>
#include <cub/block/block_reduce.cuh>
//#include <cupy/cub/cub/block/block_reduce.cuh>

extern "C" __global__ void 
dotp(
    thrust::complex< float > * value,
    thrust::complex< float> * framesl,
    thrust::complex< float> * framesr,
    thrust::complex< float> * frames_norm,
    size_t * col,
    size_t * row,
    long long int * dx,
    long long int * dy,
    int nnz,
    int frame_height, int frame_width) {
        
        typedef cub::BlockReduce< thrust::complex< float > , 128> BlockReduce;
        
        
        // Allocate shared memory for BlockReduce
        __shared__ typename BlockReduce::TempStorage temp_storage;
        
        
        
        int ii = blockIdx.x ;
        if (ii >= nnz) return;

        int col00 = col[ii];
        int row00 = row[ii];
        
        thrust::complex< float> dd1 = frames_norm[col[ii]];
        thrust::complex< float> dd2 = frames_norm[row[ii]];
        
       /*
        if (blockIdx.x == 1 && threadIdx.x == 0) {
            printf("col: %d, row: %d\n", col00, row00);
           }
       */    
        long long int Dx = frame_width - abs(dx[ii]); /*integration width */
        long long int Dy = frame_height - abs(dy[ii]); /*integration height */
       
       
        /*row-wise*/
        /* offset, including row */
        long long int DD = col00 * frame_height * frame_width  + (-dx[ii] + abs(dx[ii])) / 2 + 
        (-dy[ii] + abs(dy[ii]))/2 * frame_width; /*row-wise*/
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
        size_t ii1, ii2;
            
        /* loop within frame overlap */
        for (size_t pos = threadIdx.x; pos < (Dx * Dy); pos += blockDim.x){
                //ii1 = pos / Dy * frame_height + pos % Dy + DD ; /*column-wise*/
                ii1 = pos / Dx * frame_width + pos % Dx + DD ; /*row-wise*/
                ii2 = ii1 + Dij;
                
                /*
                if (blockIdx.x == 1 && threadIdx.x < 3){
                    printf("ii1: %d, ii2 : %d\n", static_cast<int>(ii1), static_cast<int>(ii2));
                    printf("framesl: %f + %fi\n", thrust::real(framesl[ii1]), thrust::imag(framesl[ii1]));
                    printf("framesr: %f + %fi\n", thrust::real(framesr[ii2]), thrust::imag(framesr[ii2]));
                    printf("dd: %f + %f\n", dd1, dd2);
                    }
                */
                
                 Sum0 += thrust::conj(framesl[ii1]) *  framesr[ii2]; /* conj/not */
                }

                
      
        // Compute the block-wide sum for thread0
        thrust::complex< float >  Sum1 = BlockReduce(temp_storage).Sum(Sum0);
        //thrust::complex< float >  Sum1 = Sum0 ;

        /*we know it is hermitian*/
        if (col00 == row00)
           Sum1.imag(0.0f);
        
        if (threadIdx.x == 0)
            //value[ii] = Sum1; 
            value[ii] = Sum1/(dd1 * dd2); 
        }


