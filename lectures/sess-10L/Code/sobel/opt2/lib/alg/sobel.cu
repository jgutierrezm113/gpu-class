/* Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 * 
 * Sobel Algorithm Implementation 
 *  
 */
 
#include "sobel.h"

#define apron_size 1
#define overlap 2*apron_size


/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}

using namespace std;

void modThreshold (unsigned int value){
	threshold = value;
}

/*
 * Sobel Kernel
 */
__global__ void sobelAlgorithm(unsigned char *intensity, 
				unsigned char *result,
				unsigned int imageW,
				unsigned int imageH,
				unsigned int threshold){
				
	const int xt_start = max(blockIdx.x*TILE_SIZE, 1);
	const int yt_start = max(blockIdx.y*TILE_SIZE, 1);
	
	const int xt_end = min(xt_start + TILE_SIZE - 1, imageW - 2);
	const int yt_end = min(yt_start + TILE_SIZE - 1, imageH - 2);
    
	const int x = min(xt_start + threadIdx.x, xt_end);
	const int y = min(yt_start + threadIdx.y, yt_end);
        const int dataW = TILE_SIZE*gridDim.x;
	const int location = dataW*y + x;

	const int var1 = intensity[ dataW * (y-1) + x+1 ] - intensity[ dataW * (y+1) + x-1 ];
        const int var2 = intensity[ dataW * (y+1) + x+1 ] - intensity[ dataW * (y-1) + x-1 ];
	const int var3 = intensity[ dataW * (y)   + x+1 ] - intensity[ dataW * (y)   + x-1 ];
	const int var4 = intensity[ dataW * (y+1) + x ] - intensity[ dataW * (y-1) + x ]; 
	
    
	const int magnitude = (var1+var2+2*var3)*(var1+var2+2*var3) + (var1-var2-2*var4)*(var1-var2-2*var4);
	
	result[location] = ( (magnitude > threshold) ? 255 : 0); 		
				
        
	
}

unsigned char *sobel(unsigned char *intensity,
		unsigned int height, 
		unsigned int width){
	
	#if defined(DEBUG)
		printf("Printing input data\n");
		printf("Height: %d\n", height);
		printf("Width: %d\n", width);
	#endif

	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	// Both are the same size (CPU/GPU).
	gpu.size = XSize*YSize;
	
	// Allocate arrays in GPU memory
	#if defined(VERBOSE)
		printf ("Allocating arrays in GPU memory.\n");
	#endif
	
	// Zero padding on the boarder of the image
	/*for (int j = 0; j<height ; j++) {
	    for (int i = width-1; i>-1 ; i--) {
			intensity[width*j + i + 1] = intensity[width*j + i];
		}
		intensity[width*j] = 0;
		intensity[width*j + width + 1] = 0;
	}
	
	for (int i = 0; i<width+2 ; i++) {
	    for (int j = height-1; j>-1 ; j--) {
			intensity[width*(j+1) + i] = intensity[width*j + i];
		}
		intensity[i] = 0;
		intensity[width*(height+1) + i] = 0;
	}
		
    // 		
	for (int i = width+2; i < XSize; i++) {
		for (int j = 0; j < height+2; j++) {
			intensity[XSize*j + i] = 0;
		}	
	}*/
	     
	checkCuda(cudaMalloc((void**)&gpu.intensity, gpu.size*sizeof(char)));
	checkCuda(cudaMalloc((void**)&gpu.result, gpu.size*sizeof(char)));

        checkCuda(cudaMemset(gpu.result , 0 , gpu.size));
        checkCuda(cudaMemset(gpu.intensity , 0 , gpu.size));

	// Allocate result array in CPU memory
	gpu.resultOnCPU = new unsigned char[gpu.size];
				
        checkCuda(cudaMemcpy(gpu.intensity, 
			intensity, 
			gpu.size*sizeof(char), 
			cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());
	
	#if defined(VERBOSE)
		printf("Running algorithm on GPU.\n");
	#endif
	
	dim3 dimGrid(gridXSize, gridYSize);
    dim3 dimBlock(BLOCK_TILE_SIZE, BLOCK_TILE_SIZE);
	
	#if defined(CUDA_TIMING)
		float Ttime;
		TIMER_CREATE(Ttime);
		TIMER_START(Ttime);
	#endif	
	
	#if defined(CUDA_TIMING)
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif
	
	// Launch modified kernel to begin image segmenation
	sobelAlgorithm<<<dimGrid, dimBlock>>>(gpu.intensity, 
					      gpu.result,
						  width,
						  height,
					      threshold);
	
	checkCuda(cudaDeviceSynchronize());

	#if defined(CUDA_TIMING)
		TIMER_END(Ktime);
		printf("Modified Kernel Execution Time: %f ms\n", Ktime);
	#endif
	
	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(gpu.resultOnCPU, 
			gpu.result, 
			gpu.size*sizeof(char), 
			cudaMemcpyDeviceToHost));
			
	// Free resources and end the program
	checkCuda(cudaFree(gpu.intensity));
	checkCuda(cudaFree(gpu.result));
	
	#if defined(CUDA_TIMING)
		TIMER_END(Ttime);
		printf("Total GPU Execution Time: %f ms\n", Ttime);
	#endif
	
	return(gpu.resultOnCPU);

}

unsigned char *sobelWarmup(unsigned char *intensity,
		unsigned int height, 
		unsigned int width){

	int gridXSize = 1 + (( width + 1) / TILE_SIZE);
	int gridYSize = 1 + ((height + 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	// Both are the same size (CPU/GPU).
	gpu.size = XSize*YSize;
	
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&gpu.intensity              , gpu.size*sizeof(char)));
	checkCuda(cudaMalloc((void**)&gpu.result                 , gpu.size*sizeof(char)));
	
	// Allocate result array in CPU memory
	gpu.resultOnCPU = new unsigned char[gpu.size];
				
        checkCuda(cudaMemcpy(gpu.intensity, 
			intensity, 
			gpu.size*sizeof(char), 
			cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());

	dim3 dimGrid(gridXSize, gridYSize);
        dim3 dimBlock(BLOCK_TILE_SIZE, BLOCK_TILE_SIZE);
	
	// Launch kernel to begin image segmenation
	sobelAlgorithm<<<dimGrid, dimBlock>>>(gpu.intensity, 
					      gpu.result,
						  width,
						  height,
					      threshold);
	
	checkCuda(cudaDeviceSynchronize());

	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(gpu.resultOnCPU, 
			gpu.result, 
			gpu.size*sizeof(char), 
			cudaMemcpyDeviceToHost));
			
	// Free resources and end the program
	checkCuda(cudaFree(gpu.intensity));
	checkCuda(cudaFree(gpu.result));
	
	return(gpu.resultOnCPU);

}
