#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE   512

typedef struct Data {
    
	double *a;
	double *b;
	double *c;
	
} Data;

__global__ void add( Data data, int vector_size ) {
	
	// Calculate the index in the vector for the thread using the internal variables
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	
	// This if statement is added in case we have more threads executing
	// Than number of elements in the vectors. How can this help?
	if (tid < vector_size){
		
		// Compute the addition
		data.c[tid] = data.a[tid] + data.b[tid];
		
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

	// CPU Struct
	Data data_cpu;
    // HERE
	//data_cpu.a = new double [vector_size]; 
	//data_cpu.b = new double [vector_size]; 
	data_cpu.c = new double [vector_size]; 
	Data data_gpu_on_cpu;
	//data_gpu_on_cpu.c = new double [vector_size]; 
            
    cudaMallocHost((void**)&data_cpu.a, vector_size*sizeof(double));
    cudaMallocHost((void**)&data_cpu.b, vector_size*sizeof(double));
    cudaMallocHost((void**)&data_gpu_on_cpu.c, vector_size*sizeof(double));

	// fill the arrays 'a' and 'b' on the CPU
	printf("Initializing input arrays.\n");
	for (int i = 0; i < vector_size; i++) {
		data_cpu.a[i] = rand()*cos(i);
		data_cpu.b[i] = rand()*sin(i);
		data_cpu.c[i] = 0.0;
	}
	
	// allocate the memory on the GPU
	Data data_gpu; 
	cudaMalloc (&data_gpu.a, vector_size*sizeof(double));
	cudaMalloc (&data_gpu.b, vector_size*sizeof(double));
	cudaMalloc (&data_gpu.c, vector_size*sizeof(double));
	
	//
	// CPU Calculation
	//////////////////

	printf("Running sequential job.\n");
	cudaEventRecord(start,0);

	// Calculate C in the CPU
	for (int i = 0; i < vector_size; i++) {
			data_cpu.c[i] = data_cpu.a[i] + data_cpu.b[i];
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

	// copy the input to the GPU
	cudaMemcpy (data_gpu.a, data_cpu.a, vector_size*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy (data_gpu.b, data_cpu.b, vector_size*sizeof(double), cudaMemcpyHostToDevice);

	// call the kernel
	add<<<grid_size, BLOCK_SIZE>>>(data_gpu, vector_size);

	// copy the array 'c' back from the GPU to the CPU
	cudaMemcpy (data_gpu_on_cpu.c, data_gpu.c, vector_size*sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel Job Time: %.2f ms\n", time);
	
	// compare the results
	int error = 0;
	for (int i = 0; i < vector_size; i++) {
		if (data_cpu.c[i] != data_gpu_on_cpu.c[i]){
			error = 1;
			printf( "Error starting element %d, %f != %f\n", i, data_gpu_on_cpu.c[i], data_cpu.c[i] );    
		}
		if (error) break; 
	}

	if (error == 0){
		printf ("Correct result. No errors were found.\n");
	}

	// free CPU data
    // HERE
	cudaFreeHost (data_cpu.a);
	cudaFreeHost (data_cpu.b);
	free (data_cpu.c);
	cudaFreeHost (data_gpu_on_cpu.c);

	// free the memory allocated on the GPU
	cudaFree (data_gpu.a);
	cudaFree (data_gpu.b);
	cudaFree (data_gpu.c);

	return 0;
}