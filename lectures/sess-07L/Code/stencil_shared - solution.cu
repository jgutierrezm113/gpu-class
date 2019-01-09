#include <stdio.h>
#include <stdlib.h>

#define RADIUS 8
#define BLOCK_SIZE 512

__global__ void stencil_shared(double *in, double *out, int vector_size) {
    
    __shared__ double temp[BLOCK_SIZE + 2 * RADIUS];
	
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;
	
	if (gindex < vector_size){
		
		// Read input elements into shared memory
		temp[lindex] = in[gindex];
		// At both end of a block, the sliding window moves beyond the block boundary.
		// E.g, for thread id = 512, we will read in[505] and in[1030] into temp.
		if (threadIdx.x < RADIUS) {
		   temp[lindex - RADIUS] = (gindex - RADIUS >= 0) ? in[gindex - RADIUS]: 0.0;
		   temp[lindex + BLOCK_SIZE] = (gindex + BLOCK_SIZE < vector_size) ? in[gindex + BLOCK_SIZE]: 0.0;
		}
	
	} else {
		temp[lindex] = 0.0;
	}
	
	__syncthreads();
	
	if (gindex < vector_size){
		// Apply the stencil
		double result = 0.0;
		//#pragma unroll
		for (int offset = -RADIUS ; offset <= RADIUS ; ++offset)
		   result += temp[lindex + offset];

		// Store the result
		out[gindex] = result; 
	}
}	

int main( int argc, char* argv[] ) { 

	// Parse Input arguments

	// Check the number of arguments (we only receive command + vector size)
	if (argc != 2) {
		// Tell the user how to run the program
		printf ("Usage: %s vector_size\n", argv[0]);
		// "Usage messages" are a conventional way of telling the user
		// how to run a program if they enter the command incorrectly.
		return 1;
	}
	
	// Set GPU Variables based on input arguments
	int vector_size = atoi(argv[1]);
	int grid_size   = ((vector_size-1)/BLOCK_SIZE) + 1;

	// Set device that we will use for our cuda code
	// It will be 0, 1, 2 or 3
	cudaSetDevice(0);
        
	// Time Variables
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);

	// NEW
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	
	// CPU Struct
	double *in_cpu         = new double [vector_size];
	double *out_cpu        = new double [vector_size];
	double *out_gpu_on_cpu = new double [vector_size];

	// fill the arrays 'a' and 'b' on the CPU
	printf("Initializing input arrays.\n");
	for (int i = 0; i < vector_size; i++) {
		in_cpu[i] = (rand()%100)*cos(i);
		out_cpu[i] = 0.0;
		out_gpu_on_cpu[i] = 0.0;
	}
	
	// allocate the memory on the GPU
	double *in_gpu;
	double *out_gpu;
	cudaMalloc (&in_gpu, vector_size*sizeof(double));
	cudaMalloc (&out_gpu, vector_size*sizeof(double));

	// copy the input to the GPU
	cudaMemcpy (in_gpu, in_cpu, vector_size*sizeof(double), cudaMemcpyHostToDevice);
	
	//
	// CPU Calculation
	//////////////////

	printf("Running sequential job.\n");
	cudaEventRecord(start,0);

	// Calculate C in the CPU
	for (int i = 0; i < vector_size; ++i) {
		for (int offset = -RADIUS ; offset <= RADIUS ; ++offset)
		   out_cpu[i] += (i + offset >= 0 && i + offset < vector_size) ? in_cpu[i + offset] : 0.0;
	}

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("\tSequential Job Time: %.2f ms\n", time);

	//
	// GPU Calculation
	////////////////////////

	printf("Running parallel job.\n");

	cudaEventRecord(start,0);

	// call the kernel
	stencil_shared<<<grid_size, BLOCK_SIZE>>>(in_gpu, out_gpu, vector_size);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel Job Time: %.2f ms\n", time);

	// copy the array 'c' back from the GPU to the CPU
	cudaMemcpy (out_gpu_on_cpu, out_gpu, vector_size*sizeof(double), cudaMemcpyDeviceToHost);
	
	// compare the results
	int error = 0;
	for (int i = 0; i < vector_size; i++) {
		if (out_cpu[i] != out_gpu_on_cpu[i]){
			error = 1;
			printf( "Mistake at element %d\n", i);
			int start = (i-RADIUS<0)?0:i-RADIUS;
			int end = (i+RADIUS>vector_size)?vector_size:i+RADIUS;
			for (int offset = start ; offset <= end ; offset++)
                printf( "index = %d \tin = %.5lf \tout GPU = %.5lf \tCPU %.5lf\n", offset, 
															in_cpu[offset], 
															out_gpu_on_cpu[offset], 
															out_cpu[offset] );    
		}
		if (error) break; 
	}

	if (error == 0){
		printf ("Correct result. No errors were found.\n");
	}

	// free CPU data
	free (in_cpu);
	free (out_cpu);
	free (out_gpu_on_cpu);

	// free the memory allocated on the GPU
	cudaFree (in_gpu);
	cudaFree (out_gpu);

	return 0;
}
