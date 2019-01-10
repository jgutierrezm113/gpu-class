
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
                       unsigned int size_x,
                       unsigned int size_y,
                       int threshold){

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
		
	// Including border
	__shared__ unsigned char  inTile[TILE_SIZE+2][TILE_SIZE+2]; // input

	// Read Input Data into Shared Memory
	/////////////////////////////////////////////////////////////////////////////

	int x = bx*TILE_SIZE+tx;
	int y = by*TILE_SIZE+ty;
	  
	int location = 	y*(gridDim.x*TILE_SIZE)+x;
	
	int sharedX = tx + 1;
	int sharedY = ty + 1;
		
	inTile[sharedY][sharedX] = input[location];

	// Read Border Data into Shared Memory
	/////////////////////////////////////////////////////////////////////////////////////
	int posX;
	int posY;
	
	// Horizontal Border
	if (ty == 0){
		posX = sharedX;
		posY = 0;
		if (by == 0){
			inTile[posY][posX] = 0;
		} else {
			inTile[posY][posX] = input[(y-1)*(gridDim.x*TILE_SIZE)+x];		
		}

	} else if (ty == BLOCK_TILE_SIZE-1){
		posX = sharedX;
		posY = TILE_SIZE+1;
		if (by == gridDim.y-1){
			inTile[posY][posX] = 0;
		} else {
			inTile[posY][posX] = input[(y+1)*(gridDim.x*TILE_SIZE)+x];
		}
	}
	
	// Vertical Border
	if (tx == 0){
		posX = 0;
		posY = sharedY;
		if (bx == 0){
			inTile[posY][posX] = 0;
		} else {
			inTile[posY][posX] = input[y*(gridDim.x*TILE_SIZE)+(x-1)];
		}
		
	} else if (tx == BLOCK_TILE_SIZE-1){
		posX = TILE_SIZE+1;
		posY = sharedY;
		if (bx == gridDim.x-1){
			inTile[posY][posX] = 0;
		} else {
			inTile[posY][posX] = input[y*(gridDim.x*TILE_SIZE)+(x+1)];
		}
	}
	
	// Corners for Border
	if (tx == 0 && ty == 0){
		if (bx == 0 || by == 0){
			inTile[          0][          0] = 0;
		} else {
			inTile[          0][          0] = input [(y-1)*(gridDim.x*TILE_SIZE)+(x-1)];
		}
	} else if (tx == 0 && ty == BLOCK_TILE_SIZE-1){
		if (bx == 0 || by == gridDim.y-1){
			inTile[TILE_SIZE+1][          0] = 0;
		} else {
			inTile[TILE_SIZE+1][          0] = input [(y+1)*(gridDim.x*TILE_SIZE)+(x-1)];
		}
	} else if (tx == BLOCK_TILE_SIZE-1 && ty == 0){
		if (bx == gridDim.x-1 || by == 0){
			inTile[          0][TILE_SIZE+1] = 0;
		} else {
			inTile[          0][TILE_SIZE+1] = input [(y-1)*(gridDim.x*TILE_SIZE)+(x+1)];
		}
	} else if (tx == BLOCK_TILE_SIZE-1 && ty == BLOCK_TILE_SIZE-1){
		if (bx == gridDim.x-1 || by == gridDim.y-1){
			inTile[TILE_SIZE+1][TILE_SIZE+1] = 0;
		} else {
			inTile[TILE_SIZE+1][TILE_SIZE+1] = input [(y+1)*(gridDim.x*TILE_SIZE)+(x+1)];
		}
	}
	
	__syncthreads();
	
	// Algorithm 
	/////////////////////////////////////////////////////////////////////

	int sum1 =  inTile[sharedY-1][sharedX+1] - inTile[sharedY-1][sharedX-1] 
		 + 2 * (inTile[sharedY  ][sharedX+1] - inTile[sharedY  ][sharedX-1])
		 +      inTile[sharedY+1][sharedX+1] - inTile[sharedY+1][sharedX-1];
	int sum2 =  inTile[sharedY-1][sharedX-1] + inTile[sharedY-1][sharedX+1]
		 + 2 * (inTile[sharedY-1][sharedX  ] - inTile[sharedY+1][sharedX  ])
		 -      inTile[sharedY+1][sharedX-1] - inTile[sharedY+1][sharedX+1];

	int magnitude = sqrt( (float) (sum1*sum1+sum2*sum2));
	
	if(magnitude > threshold)
		output[location] = 255;
	else 
		output[location] = 0;

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
                   int threshold ){
    
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
                                      width,
                                      height,
                                      threshold);
                                      
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

