#include "cupy/complex.cuh"

extern "C" __global__ void 
Refine_illum(
    thrust::complex<float> * illumination,
    thrust::complex< float> * frames,
    thrust::complex< float> * normalization,
    size_t * col,
    size_t * row,
    long long int * dx,
    long long int * dy,
    int bw,
    int nnz,
    int frame_height, int frame_width) {
        
      
        int ii = blockIdx.x ;
        
        if (ii >= nnz) return;

        int col00 = col[ii];
        int row00 = row[ii];
        int shiftl = col00 * frame_height * frame_width;
        int shiftr = row00 * frame_height * frame_width;
        
  
       
        if (blockIdx.x == 1 && threadIdx.x == 0) {
            printf("col: %d, row: %d\n", col00, row00);
           }
       
       
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
                
                int image_index =  ii3 * frame_width + ii4;
               
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
                
                thrust::complex<float> output;
                output = thrust::conj(frames[ii1]) * frames[ii2] / (normalization[ii3] * normalization[ii4]);
             
                atomicAdd((float*)&(illumination[image_index]), output.real());
                atomicAdd((float*)&(illumination[image_index]) + 1, output.imag());
      
                }
        }
            
    
