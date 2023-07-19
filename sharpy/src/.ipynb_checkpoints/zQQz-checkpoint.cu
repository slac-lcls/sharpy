#include "cupy/complex.cuh"
//#include <iostream>
#include <cub/cub.cuh> 
// #include <cub/block/block_reduce.cuh>

extern "C" __global__ void 
dotp(
    thrust::complex< float > * value,
    thrust::complex< float> * framesl,
    thrust::complex< float> * framesr,
    long long * col,
    long long * row,
    long long * dx,
    long long * dy,
    long long nnz,
    int frame_height, int frame_width) {
        
        typedef cub::BlockReduce< thrust::complex< float > , 128> BlockReduce;
        
        
        // Allocate shared memory for BlockReduce
        __shared__ typename BlockReduce::TempStorage temp_storage;
    
        
        int ii = blockIdx.x ;
        if (ii >= nnz) return;

        size_t col00 = col[ii];
        size_t row00 = row[ii];
            
        int Dx = frame_width - abs(dx[ii]); /*integration width */
        int Dy = frame_height - abs(dy[ii]); /*integration height */
    
        /* offset, including row */
        size_t DD = row00 * frame_height * frame_width  + (dx[ii] + abs(dx[ii])) / 2 * frame_height  + (dy[ii] + abs(dy[ii]))/2;
        /* offset between frame1 and frame2 */
        size_t Dij = (-dx[ii]) * frame_height - dy[ii] + (col00 - row00) * frame_height * frame_width ;

        thrust::complex<float> Sum0 = 0;
        size_t ii1, ii2;
            
        /* loop within frame overlap */
        for (int pos = threadIdx.x; pos < (Dx * Dy); pos += blockDim.x){
                ii1 = pos / Dy * frame_height + pos % Dy + DD ; 
                ii2 = ii1 + Dij;
                 
                Sum0 += thrust::conj(framesl[ii1]) *  framesr[ii2]; /* conj/not */
            }

        // Compute the block-wide sum for thread0
        //thrust::complex< float >  Sum1 = BlockReduce(temp_storage).Sum(Sum0);
          thrust::complex< float >  Sum1 = Sum0 ;

            /*we know it is hermitian*/
            if (col00 == row00)
                //imag(Sum1) = 0;
                Sum1.imag(0.0f);
                
            value[ii] = Sum1; /*How to append to it?*/
        }


