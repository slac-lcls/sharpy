#include "cupy/complex.cuh"

extern "C" __global__ void 
Overlap(thrust::complex< float >  * image, 
      thrust::complex< float >  * frames,
      float2 * translations,  
      thrust::complex< float >  * illumination,
      int img_height, int img_width,
      int n_frames, int frame_height, int frame_width){

	/* Each thread block is divided in tiles, each tile takes care of a frame  */

	const int frame_size = frame_width*frame_height;

	const int fid = blockIdx.x;

	if (fid>=n_frames) return;

	int f_x, f_y;


	for(int i = threadIdx.x % blockDim.x;i < frame_size;i+= blockDim.x)
	{

                f_x = i%frame_width;
                f_y = i/frame_width;
                
		int g_x = ((int)(translations[fid].x) + f_x)%img_width;
		int g_y = ((int)(translations[fid].y) + f_y)%img_height;
        
        /*
		if(g_x >= img_width || g_y >= img_height){
                        g_x -= img_width; //periodic boundary
                        g_y -= img_height; //periodic boundary
			//We don't write anything into the image if it is out of bounds
			//continue;
		}		
        */
        
		int frames_index = fid*frame_size + i;
		int image_index = g_y*img_width + g_x;
		thrust::complex< float > overlap_output;

		if(illumination){
			overlap_output = illumination[i];	
 			if(frames){
				//Either we have the "frames * Conj(illumination)"...	
                overlap_output = frames[frames_index] * thrust::conj(overlap_output);

			}		
			else{
				//...or we have the "illumination * Conj(illumination)" which is the "abs(illumination)^2"
				overlap_output = overlap_output * thrust::conj(overlap_output);

			}
		}
		else{
			overlap_output = frames[frames_index];
        }
		atomicAdd((float*)&(image[image_index]), overlap_output.real());
		atomicAdd((float*)&(image[image_index]) + 1, overlap_output.imag());

	}
}

