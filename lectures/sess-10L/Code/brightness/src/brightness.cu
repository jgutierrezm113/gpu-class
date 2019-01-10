
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#include "config.h"

#define TIMER_CREATE(t)               \
  cudaEvent_t t##_start, t##_end;     \
  cudaEventCreate(&t##_start);        \
  cudaEventCreate(&t##_end);               
 
 
#define TIMER_START(t)                \
  cudaEventRecord(t##_start);         \
  cudaEventSynchronize(t##_start);    \
 
 
#define TIMER_END(t)                             \
  cudaEventRecord(t##_end);                      \
  cudaEventSynchronize(t##_end);                 \
  cudaEventElapsedTime(&t, t##_start, t##_end);  \
  cudaEventDestroy(t##_start);                   \
  cudaEventDestroy(t##_end);     
  
unsigned char *input_gpu;
unsigned char *output_gpu;

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

__global__ void kernel(unsigned char *input, 
                       unsigned char *output,
                       int inc){

	// Read Input Data
	/////////////////////////////////////////////////////////////////////////////

	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	  
	int location = 	y*(gridDim.x*TILE_SIZE)+x;
			
	unsigned char value = input[location];
    	
	// Algorithm 
	/////////////////////////////////////////////////////////////////////

	if ((int) value + inc > 255) value = 255;
	else if ((int) value + inc < 0) value = 0;
	else value = value + inc;
    
    output[location] = value;

}

__global__ void warmup(unsigned char *input, 
                       unsigned char *output){

	// Read Input Data
	/////////////////////////////////////////////////////////////////////////////

	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	  
	int location = 	y*(gridDim.x*TILE_SIZE)+x;
			
	unsigned char value = 0;

    output[location] = value;

}

void gpu_function (unsigned char *data, 
                   unsigned int height, 
                   unsigned int width,
                   int inc ){
    
	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	// Both are the same size (CPU/GPU).
	int size = XSize*YSize;
	
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char)));
	
    checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
	
    // Copy data to GPU
    checkCuda(cudaMemcpy(input_gpu, 
        data, 
        size*sizeof(char), 
        cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());

    // Execute algorithm

    dim3 dimGrid(gridXSize, gridYSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);

	// Kernel Call
	#if defined(CUDA_TIMING)
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif
        
        kernel<<<dimGrid, dimBlock>>>(input_gpu, 
                                      output_gpu,
                                      inc);
                                      
        checkCuda(cudaPeekAtLastError());                                     
        checkCuda(cudaDeviceSynchronize());
	
	#if defined(CUDA_TIMING)
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif
        
	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(data, 
			output_gpu, 
			size*sizeof(unsigned char), 
			cudaMemcpyDeviceToHost));

    // Free resources and end the program
	checkCuda(cudaFree(output_gpu));
	checkCuda(cudaFree(input_gpu));

}

void gpu_warmup   (unsigned char *data, 
                   unsigned int height, 
                   unsigned int width){
    #if defined (WARMUP)                     
        int gridXSize = 1 + (( width - 1) / TILE_SIZE);
        int gridYSize = 1 + ((height - 1) / TILE_SIZE);
        
        int XSize = gridXSize*TILE_SIZE;
        int YSize = gridYSize*TILE_SIZE;
        
        // Both are the same size (CPU/GPU).
        int size = XSize*YSize;
        
        // Allocate arrays in GPU memory
        checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char)));
        checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char)));
        
        checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
                
        // Copy data to GPU
        checkCuda(cudaMemcpy(input_gpu, 
            data, 
            size*sizeof(char), 
            cudaMemcpyHostToDevice));

        checkCuda(cudaDeviceSynchronize());
            
        // Execute algorithm
            
        dim3 dimGrid(gridXSize, gridYSize);
        dim3 dimBlock(TILE_SIZE, TILE_SIZE);
        
        warmup<<<dimGrid, dimBlock>>>(input_gpu, 
                                      output_gpu);
                                             
        checkCuda(cudaDeviceSynchronize());
            
        // Retrieve results from the GPU
        checkCuda(cudaMemcpy(data, 
                output_gpu, 
                size*sizeof(unsigned char), 
                cudaMemcpyDeviceToHost));
                            
        // Free resources and end the program
        checkCuda(cudaFree(output_gpu));
        checkCuda(cudaFree(input_gpu));
    #endif
}

