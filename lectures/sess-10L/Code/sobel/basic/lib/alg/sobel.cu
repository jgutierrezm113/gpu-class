/* Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 * 
 * Sobel Algorithm Implementation 
 *  
 */
 
#include "sobel.h"

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
				unsigned int threshold){
        
        int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
        
  	int x = bx*TILE_SIZE+tx;
	int y = by*TILE_SIZE+ty;
        
        int xsize = TILE_SIZE*gridDim.x;
        int ysize = TILE_SIZE*gridDim.y;
        
	
        if (y > 1 && y < ysize-1 && x > 1 && x < xsize-1){

                int location = 	y*xsize+x;

		int sum1 =  intensity[ xsize * (y-1) + x+1 ] - 
			    intensity[ xsize * (y-1) + x-1 ] + 
			2 * intensity[ xsize * (y)   + x+1 ] - 
			2 * intensity[ xsize * (y)   + x-1 ] + 
			    intensity[ xsize * (y+1) + x+1 ] - 
			    intensity[ xsize * (y+1) + x-1 ];

		int sum2 = intensity[ xsize * (y-1) + x-1 ] + 
		       2 * intensity[ xsize * (y-1) + x   ] + 
			   intensity[ xsize * (y-1) + x+1 ] - 
			   intensity[ xsize * (y+1) + x-1 ] - 
		       2 * intensity[ xsize * (y+1) + x   ] - 
			   intensity[ xsize * (y+1) + x+1 ];

		int magnitude =  sum1*sum1 + sum2*sum2;

		if (magnitude > threshold)
			result[location] = 255;
		else
			result[location] = 0;
	}
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
	
	#if defined(CUDA_TIMING)
		float Ttime;
		TIMER_CREATE(Ttime);
		TIMER_START(Ttime);
	#endif
	
	checkCuda(cudaMalloc((void**)&gpu.intensity              , gpu.size*sizeof(char)));
	checkCuda(cudaMalloc((void**)&gpu.result                 , gpu.size*sizeof(char)));
	
        checkCuda(cudaMemset(gpu.result , 0 , gpu.size));
        checkCuda(cudaMemset(gpu.intensity , 0 , gpu.size));

	// Allocate result array in CPU memory
	gpu.resultOnCPU = new unsigned char[gpu.size];
				
        checkCuda(cudaMemcpy(gpu.intensity, 
			intensity, 
			gpu.size*sizeof(char), 
			cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());

	#if defined(CUDA_TIMING)
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif
	
	#if defined(VERBOSE)
		printf("Running algorithm on GPU.\n");
	#endif
	
	dim3 dimGrid(gridXSize, gridYSize);
        dim3 dimBlock(BLOCK_TILE_SIZE, BLOCK_TILE_SIZE);
	
	// Launch kernel to begin image segmenation
	sobelAlgorithm<<<dimGrid, dimBlock>>>(gpu.intensity, 
					      gpu.result,
					      threshold);
	
	checkCuda(cudaDeviceSynchronize());

	#if defined(CUDA_TIMING)
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
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

	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
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