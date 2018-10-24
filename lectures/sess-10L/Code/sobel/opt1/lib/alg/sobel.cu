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
		
	// Including border
	__shared__ unsigned char  inTile[TILE_SIZE+2][TILE_SIZE+2]; // input

	// Read Input Data into Shared Memory
	/////////////////////////////////////////////////////////////////////////////

	int x = bx*TILE_SIZE+tx;
	int y = by*TILE_SIZE+ty;
	  
	int location = 	y*(gridDim.x*TILE_SIZE)+x;
	
	int sharedX = tx + 1;
	int sharedY = ty + 1;
		
	inTile[sharedY][sharedX] = intensity[location];

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
			inTile[posY][posX] = intensity[(y-1)*(gridDim.x*TILE_SIZE)+x];		
		}

	} else if (ty == BLOCK_TILE_SIZE-1){
		posX = sharedX;
		posY = TILE_SIZE+1;
		if (by == gridDim.y-1){
			inTile[posY][posX] = 0;
		} else {
			inTile[posY][posX] = intensity[(y+1)*(gridDim.x*TILE_SIZE)+x];
		}
	}
	
	// Vertical Border
	if (tx == 0){
		posX = 0;
		posY = sharedY;
		if (bx == 0){
			inTile[posY][posX] = 0;
		} else {
			inTile[posY][posX] = intensity[y*(gridDim.x*TILE_SIZE)+(x-1)];
		}
		
	} else if (tx == BLOCK_TILE_SIZE-1){
		posX = TILE_SIZE+1;
		posY = sharedY;
		if (bx == gridDim.x-1){
			inTile[posY][posX] = 0;
		} else {
			inTile[posY][posX] = intensity[y*(gridDim.x*TILE_SIZE)+(x+1)];
		}
	}
	
	// Corners for Border
	if (tx == 0 && ty == 0){
		if (bx == 0 || by == 0){
			inTile[          0][          0] = 0;
		} else {
			inTile[          0][          0] = intensity [(y-1)*(gridDim.x*TILE_SIZE)+(x-1)];
		}
	} else if (tx == 0 && ty == BLOCK_TILE_SIZE-1){
		if (bx == 0 || by == gridDim.y-1){
			inTile[TILE_SIZE+1][          0] = 0;
		} else {
			inTile[TILE_SIZE+1][          0] = intensity [(y+1)*(gridDim.x*TILE_SIZE)+(x-1)];
		}
	} else if (tx == BLOCK_TILE_SIZE-1 && ty == 0){
		if (bx == gridDim.x-1 || by == 0){
			inTile[          0][TILE_SIZE+1] = 0;
		} else {
			inTile[          0][TILE_SIZE+1] = intensity [(y-1)*(gridDim.x*TILE_SIZE)+(x+1)];
		}
	} else if (tx == BLOCK_TILE_SIZE-1 && ty == BLOCK_TILE_SIZE-1){
		if (bx == gridDim.x-1 || by == gridDim.y-1){
			inTile[TILE_SIZE+1][TILE_SIZE+1] = 0;
		} else {
			inTile[TILE_SIZE+1][TILE_SIZE+1] = intensity [(y+1)*(gridDim.x*TILE_SIZE)+(x+1)];
		}
	}
	
	__syncthreads();
	
	// Algorithm 
	/////////////////////////////////////////////////////////////////////

	int sum1 =      inTile[sharedY-1][sharedX+1] - inTile[sharedY-1][sharedX-1] 
		 + 2 * (inTile[sharedY  ][sharedX+1] - inTile[sharedY  ][sharedX-1])
		 +      inTile[sharedY+1][sharedX+1] - inTile[sharedY+1][sharedX-1];
	int sum2 =      inTile[sharedY-1][sharedX-1] + inTile[sharedY-1][sharedX+1]
		 + 2 * (inTile[sharedY-1][sharedX  ] - inTile[sharedY+1][sharedX  ])
		 -      inTile[sharedY+1][sharedX-1] - inTile[sharedY+1][sharedX+1];

	int magnitude = sum1*sum1+sum2*sum2;
	
	if(magnitude > threshold)
		result[location] = 255;
	else 
		result[location] = 0;

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