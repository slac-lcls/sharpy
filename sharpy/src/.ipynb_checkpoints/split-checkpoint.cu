#include "cupy/complex.cuh"

extern "C" __global__ void 
Split(thrust::complex< float > * image, 
      thrust::complex< float > * frames,
      float2 * translations,  
      thrust::complex< float > * illumination,
      int img_height, int img_width,
      int n_frames, int frame_height, int frame_width, int nthreads, int tsize){
     
	/* Each thread block is divided in tiles, each tile takes care of a frame  */
	const int frame_size = frame_width*frame_height;

	//const int fid = blockIdx.x*nthreads/tsize + threadIdx.x/tsize;
    const int fid = blockIdx.x;

	if (fid>=n_frames) return; 

	int f_x, f_y;

	//for(int i = threadIdx.x % tsize;i < frame_size;i+= tsize)
    for (size_t i = threadIdx.x; i < frame_size; i += blockDim.x)
	{
                f_x = i%frame_width;
                f_y = i/frame_width;

		int g_x = ((int)(translations[fid].x) + f_x)%img_width; 
		int g_y = ((int)(translations[fid].y) + f_y)%img_height;
       
		long long int frames_index = fid*frame_size + i;

		/*
        if(g_x >= img_width || g_y >= img_height)
		{   
                g_x -= img_width; //periodic boundary
                g_y -= img_height; //periodic boundary
			//frames[frames_index] = 0;
			//continue;
		}
        */
        
		thrust::complex< float > image_pixel = image[g_y*img_width + g_x];
        thrust::complex< float > output;

        if(illumination){
            output = image_pixel * illumination[i];
		}
		else{
			output = image_pixel;
		}

		frames[frames_index] = output; 
   
    /*
    if (blockIdx.x == 0 && threadIdx.x >= 16 && threadIdx.x <= 19) {
        printf("ii: %d\n", static_cast<int>(frames_index));
        printf("ii1: %d, ii2: %d\n", static_cast<int>(g_x),static_cast<int>(g_y));
        printf("f_x: %d, f_y: %d\n", static_cast<int>(f_x),static_cast<int>(f_y));
        printf("output: %f + %fi\n", thrust::real(output), thrust::imag(output));
        printf("output2: %f + %fi\n", thrust::real(frames[frames_index]), thrust::imag(frames[frames_index]));
    }
    */  
	}

}
