#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE   512

typedef struct Data {
	// HERE
	double a;
	double b;
	double c;
	
} Data;

__global__ void add( Data data, int vector_size ) {
	
	// Calculate the index in the vector for the thread using the internal variables
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	
	// This if statement is added in case we have more threads executing
	// Than number of elements in the vectors. How can this help?
	if (tid < vector_size){
		
		// Compute the addition
		// HERE
		data[tid].c = data[tid].a + data[tid].b;
		
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
    // HERE: Notice the differences on how we are creating the arrays here
	data_cpu.a = new double [vector_size]; 
	data_cpu.b = new double [vector_size]; 
	data_cpu.c = new double [vector_size]; 
	Data data_gpu_on_cpu;
	data_gpu_on_cpu.c = new double [vector_size]; 

	// fill the arrays 'a' and 'b' on the CPU
	printf("Initializing input arrays.\n");
	for (int i = 0; i < vector_size; i++) {
		data_cpu.a[i] = rand()*cos(i);
		data_cpu.b[i] = rand()*sin(i);
		data_cpu.c[i] = 0.0;
	}
	
	// allocate the memory on the GPU
	// HERE. Code below will not work any more
	Data *data_gpu; 
	cudaMalloc (&data_gpu, vector_size*sizeof(Data));

	// copy the input to the GPU
	// HERE. Code below will not work any more
	cudaMemcpy (data_gpu, data_cpu, vector_size*sizeof(Data), cudaMemcpyHostToDevice);
	
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

	// call the kernel
	add<<<grid_size, BLOCK_SIZE>>>(data_gpu, vector_size);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel Job Time: %.2f ms\n", time);

	// copy the array 'c' back from the GPU to the CPU
	// HERE. Code below will not work any more
	cudaMemcpy (data_gpu_on_cpu, data_gpu, vector_size*sizeof(Data), cudaMemcpyDeviceToHost);
	
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
	free (data_cpu.a);
	free (data_cpu.b);
	free (data_cpu.c);
	free (data_gpu_on_cpu.c);

	// free the memory allocated on the GPU
	// HERE. Code below will not work any more
	cudaFree (data_gpu);

	return 0;
}

