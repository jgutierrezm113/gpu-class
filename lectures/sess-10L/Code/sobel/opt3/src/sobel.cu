
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

__global__ void kernel(unsigned int *intensity, 
				unsigned int *result,
				unsigned int threshold){

	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;
	unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.y;
    
	// Including border
	__shared__ unsigned char   inTile[TILE_SIZE+2][TILE_SIZE+2]; // input
	__shared__ unsigned char  outTile[TILE_SIZE+2][TILE_SIZE+2]; // output

	// Read Input Data into Shared Memory
	/////////////////////////////////////////////////////////////////////////////////////

	unsigned int x = bx<<BTSB;
	x = x + tx;
	x = x<<TTSB;
	unsigned int y = by<<BTSB;
	y = y + ty;
	y = y<<TTSB;
	
	unsigned int location = 	(((x>>TTSB)&BTSMask)                ) |
                    (((y>>TTSB)&BTSMask) << BTSB        ) |
                    ((x>>TSB)            << (BTSB+BTSB) ) ;
	location += 	((y>>TSB)<<(BTSB+BTSB))*gridDim.x;
	
	unsigned int intensityData = intensity[location];
	
	unsigned int sharedX = tx*THREAD_TILE_SIZE+1;
	unsigned int sharedY = ty*THREAD_TILE_SIZE+1;
	
	inTile[sharedY  ][sharedX  ] = ( intensityData        & 0xFF);
	inTile[sharedY  ][sharedX+1] = ((intensityData >>  8) & 0xFF);
	inTile[sharedY+1][sharedX  ] = ((intensityData >> 16) & 0xFF);
	inTile[sharedY+1][sharedX+1] = ((intensityData >> 24) & 0xFF);
	
	// Read Border Data into Shared Memory
	/////////////////////////////////////////////////////////////////////////////////////
	
	// Registers meant for speed. Two given each thread will update 2 pixels.
	unsigned int shiftTileReg1 = 0;
	unsigned int shiftTileReg2 = 0;
	
	unsigned int borderXLoc = 0;
	unsigned int borderYLoc = 0;
	
	// Needed Variables
	unsigned int bLocation;
	unsigned int borderIntData;
	
	// Update horizontal border
	borderXLoc = sharedX;
	if (ty == 0 ){		
		// Location to write in shared memory
		borderYLoc = 0;
		if (by != 0) {
			// Upper block border
			y-=THREAD_TILE_SIZE;
			shiftTileReg1 = 16;
			shiftTileReg2 = 24;
		}
	} else if (ty == BLOCK_TILE_SIZE-1){
		// Location to write in shared memory
		borderYLoc = TILE_SIZE+1;			
		if (by != gridDim.y-1) {
			// Lower block border
			y+=THREAD_TILE_SIZE;
			shiftTileReg1 = 0;
			shiftTileReg2 = 8;
		}
	}
	// Read from global and write to shared memory
	if (ty == 0 || ty == BLOCK_TILE_SIZE-1) {
		if ((by == 0           && ty == 0                ) || 
		    (by == gridDim.y-1 && ty == BLOCK_TILE_SIZE-1)){
			inTile[borderYLoc][borderXLoc  ] = 0;
			inTile[borderYLoc][borderXLoc+1] = 0;
		} else {
			bLocation = (((x>>TTSB)&BTSMask)                 ) |
                        (((y>>TTSB)&BTSMask)  << BTSB        ) |
                        ((x>>TSB)             << (BTSB+BTSB) ) ;
			bLocation += ((y>>TSB)<<(BTSB+BTSB))*gridDim.x;
			
			borderIntData = intensity[bLocation];
			
			inTile[borderYLoc][borderXLoc  ] 
					= ( borderIntData >> shiftTileReg1 ) & 0xFF;
			inTile[borderYLoc][borderXLoc+1] 
					= ( borderIntData >> shiftTileReg2 ) & 0xFF;
		}
	}
		
	// Update vertical border
	x = bx<<BTSB;
	x = x + tx;
	x = x<<TTSB;
	y = by<<BTSB;
	y = y + ty;
	y = y<<TTSB;
	
	borderYLoc = sharedY;
	if (tx == 0 ){		
		// Location to write in shared memory
		borderXLoc = 0;
		if (bx != 0) {
			// Upper block border
			x-=THREAD_TILE_SIZE;
			shiftTileReg1 = 8;
			shiftTileReg2 = 24;
		}
	} else if (tx == BLOCK_TILE_SIZE-1){
		// Location to write in shared memory
		borderXLoc = TILE_SIZE+1;			
		if (bx != gridDim.x-1) {
			// Lower block border
			x+=THREAD_TILE_SIZE;
			shiftTileReg1 = 0;
			shiftTileReg2 = 16;
		}
	}
	// Read from global and write to shared memory
	if (tx == 0 || tx == BLOCK_TILE_SIZE-1) {
		if ((bx == 0           && tx == 0                ) || 
		    (bx == gridDim.x-1 && tx == BLOCK_TILE_SIZE-1)){
			inTile[borderYLoc][borderXLoc  ] = 0;
			inTile[borderYLoc+1][borderXLoc] = 0;
		} else {
			bLocation = (((x>>TTSB)&BTSMask)                 ) |
				    (((y>>TTSB)&BTSMask)  << BTSB        ) |
				     ((x>>TSB)            << (BTSB+BTSB) ) ;
			bLocation += ((y>>TSB)<<(BTSB+BTSB))*gridDim.x;
			
			borderIntData = intensity[bLocation];
			
			inTile[borderYLoc][borderXLoc  ] 
					= ( borderIntData >> shiftTileReg1 ) & 0xFF;
			inTile[borderYLoc+1][borderXLoc] 
					= ( borderIntData >> shiftTileReg2 ) & 0xFF;
		}
	}
	
	x = bx<<BTSB;
	x = x + tx;
	x = x<<TTSB;
	y = by<<BTSB;
	y = y + ty;
	y = y<<TTSB;
	
	// Corners for Border
	shiftTileReg1 = 0;
	if ((tx == 0 || tx == BLOCK_TILE_SIZE-1) && (ty == 0 || ty == BLOCK_TILE_SIZE-1)){
		if (tx == 0) {
			borderXLoc = 0;
			x-=THREAD_TILE_SIZE;
		} else {
			borderXLoc = TILE_SIZE+1;
			x+=THREAD_TILE_SIZE;
			shiftTileReg1 += 16;
		}
		
		if (ty == 0) {
			borderYLoc = 0;
			y-=THREAD_TILE_SIZE;
		} else {
			borderYLoc = TILE_SIZE+1;
			y+=THREAD_TILE_SIZE;
			shiftTileReg1 += 8;
		}
		if (
			((tx == 0                 && ty == 0                ) && 
			 (bx == 0                 || by == 0               )) || 
			((tx == 0                 && ty == BLOCK_TILE_SIZE-1) &&
			 (bx == 0                 || by == gridDim.y-1     )) ||
			((tx == BLOCK_TILE_SIZE-1 && ty == 0                ) &&
			 (bx == gridDim.x-1       || by == 0               )) ||
			((tx == BLOCK_TILE_SIZE-1 && ty == BLOCK_TILE_SIZE-1) &&
			 (bx == gridDim.x-1       || by == gridDim.y-1     ))
		     ){
			inTile[borderYLoc][borderXLoc] = 0;
		} else {
			bLocation =
				   (((x>>TTSB)&BTSMask)                ) |
				   (((y>>TTSB)&BTSMask) << BTSB        ) |
				    ((x>>TSB)           << (BTSB+BTSB) ) ;
			bLocation +=((y>>TSB)           << (BTSB+BTSB) )*gridDim.x;
			intensityData = intensity [bLocation];
			inTile[borderYLoc][borderXLoc] = (intensityData >> shiftTileReg1 ) & 0xFF;
		}
	}
	
	__syncthreads();
	
	// Algorithm 
	/////////////////////////////////////////////////////////////////////

	for (unsigned int tempY = ty+1; tempY <= TILE_SIZE; tempY+=BLOCK_TILE_SIZE ){
		for (unsigned int tempX = tx+1; tempX <= TILE_SIZE; tempX+=BLOCK_TILE_SIZE ){
			int sum1 =  inTile[tempY-1][tempX+1] - inTile[tempY-1][tempX-1] 
				 + 2 * (inTile[tempY  ][tempX+1] - inTile[tempY  ][tempX-1])
				 +      inTile[tempY+1][tempX+1] - inTile[tempY+1][tempX-1];
			int sum2 =  inTile[tempY-1][tempX-1] + inTile[tempY-1][tempX+1]
				 + 2 * (inTile[tempY-1][tempX  ] - inTile[tempY+1][tempX  ])
				 -      inTile[tempY+1][tempX-1] - inTile[tempY+1][tempX+1];

			int magnitude = sqrt( (float) (sum1*sum1+sum2*sum2));
			
			if(magnitude > threshold)
				outTile[tempY][tempX] = 255;
			else 
				outTile[tempY][tempX] = 0;
		}
	} 
	
	__syncthreads();
	
	// Write back result
	unsigned int intData1 = outTile[sharedY  ][sharedX  ] & 0xFF;
	unsigned int intData2 = outTile[sharedY  ][sharedX+1] & 0xFF;
	unsigned int intData3 = outTile[sharedY+1][sharedX  ] & 0xFF;
	unsigned int intData4 = outTile[sharedY+1][sharedX+1] & 0xFF;
		
	unsigned int intReturnData = intData1        |
                                (intData2 << 8 ) |
                                (intData3 << 16) |
                                (intData4 << 24);
				
	result[location] = intReturnData;
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
    dim3 dimBlock(BLOCK_TILE_SIZE, BLOCK_TILE_SIZE);

	// Kernel Call
	#if defined(CUDA_TIMING)
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif
        
        kernel<<<dimGrid, dimBlock>>>((unsigned int *)input_gpu, 
                                      (unsigned int *)output_gpu,
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

