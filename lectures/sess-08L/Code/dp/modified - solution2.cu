#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define threshold 5 //(50% probability)
#define block_size 256

__global__ void add_calculation(    char* dev_a, 
                                    char* dev_b, 
                                    char* dev_c,
                                    int k,
                                    int j,
                                    int num_matrices,                                                
                                    int matrix_size ) {
        // Each thread handles a matrix
        int i = (blockIdx.x*blockDim.x) + threadIdx.x;

        if (i >= matrix_size) return;
        
        int index = k*matrix_size*matrix_size+j*matrix_size+i;
        dev_c[index] = dev_a[index] + dev_b[index];
        
}

__global__ void sub_calculation(    char* dev_a, 
                                    char* dev_b, 
                                    char* dev_c,
                                    int k,
                                    int j,
                                    int num_matrices,                                                
                                    int matrix_size ) {
        // Each thread handles a matrix
        int i = (blockIdx.x*blockDim.x) + threadIdx.x;

        if (i >= matrix_size) return;
        
        int index = k*matrix_size*matrix_size+j*matrix_size+i;
        dev_c[index] = dev_a[index] - dev_b[index];
        
}
__global__ void second_calculation( char* dev_a, 
                                    char* dev_b, 
                                    char* dev_c,
                                    int k,
                                    int num_matrices,                                                
                                    int matrix_size ) {
        // Each thread handles a matrix
        int j = (blockIdx.x*blockDim.x) + threadIdx.x;

        if (j >= matrix_size) return;
        
        int grid_size = ((matrix_size-1)/block_size) + 1;
                
        //If first value in the row of the matrix, do addition
        if (dev_a[k*matrix_size*matrix_size+j*matrix_size] < threshold){
                add_calculation <<<grid_size, block_size>>>( dev_a, 
                                                             dev_b, 
                                                             dev_c, 
                                                             k, 
                                                             j,
                                                             num_matrices, 
                                                             matrix_size);
        //Do subtraction
        } else {
                sub_calculation <<<grid_size, block_size>>>( dev_a, 
                                                             dev_b, 
                                                             dev_c, 
                                                             k, 
                                                             j,
                                                             num_matrices, 
                                                             matrix_size);
        }
}

__global__ void calculation(    char* dev_a, 
                                char* dev_b, 
                                char* dev_c,
                                int num_matrices,                                                
                                int matrix_size ) {
        // Each thread handles a matrix
        int k = (blockIdx.x*blockDim.x) + threadIdx.x;

        if (k >= num_matrices) return;
        
        // If first element is different than 0 do the computation
        if (dev_a[k*matrix_size*matrix_size] != 0){
                int grid_size = ((matrix_size-1)/block_size) + 1;
                second_calculation <<<grid_size, block_size>>>( dev_a, 
                                                                dev_b, 
                                                                dev_c, 
                                                                k, 
                                                                num_matrices, 
                                                                matrix_size);
        }
}

int main( int argc, char* argv[] ) { 

        // Parse Input arguments
        
        // Check the number of arguments (we only receive command + vector size)
        if (argc != 3) {
                // Tell the user how to run the program
                printf ("Usage:\n%s <number of matrices> <matrix_size>\n", argv[0]);
                // "Usage messages" are a conventional way of telling the user
                // how to run a program if they enter the command incorrectly.
                return -1;
        }
        
        srand ( time(NULL) );

        // Set variables with input arguments
        int num_matrices = atoi(argv[1]);
        int matrix_size  = atoi(argv[2]);
                
        // Set device that we will use for our cuda code
        cudaSetDevice(0);
        
	// Time Variables
	cudaEvent_t stp_start, stp_stop;
	cudaEvent_t cpu_start, cpu_stop;
        cudaEvent_t gpu_start, gpu_stop;
        cudaEvent_t ker_start, ker_stop;
	
        cudaEventCreate (&stp_start);
	cudaEventCreate (&stp_stop);
        
        cudaEventCreate (&cpu_start);
	cudaEventCreate (&cpu_stop);
        
	cudaEventCreate (&gpu_start);
        cudaEventCreate (&gpu_stop);
        
	cudaEventCreate (&ker_start);
        cudaEventCreate (&ker_stop);
	
	float time, ker_time;
        
        // Input Arrays and variables
        char *a       = new char [num_matrices*matrix_size*matrix_size];
        char *b       = new char [num_matrices*matrix_size*matrix_size];
        char *c_cpu   = new char [num_matrices*matrix_size*matrix_size]; 
        char *c_gpu   = new char [num_matrices*matrix_size*matrix_size];

        // Pointers in GPU memory
        char *dev_a;
        char *dev_b;
        char *dev_c;

        //
        // Fill arrays
        //////////////////
        
	cudaEventRecord(stp_start,0);
        #if defined(_OPENMP)
                printf("Setting up input arrays in parallel.\n");
                omp_set_num_threads(8);
        #else
                printf("Setting up input arrays.\n");
        #endif
        #pragma omp parallel for
        for (int k = 0; k < num_matrices; k++) {
                #if defined(_OPENMP)                
                        if (k == 0) printf ("Using %d threads.\n", omp_get_num_threads());
                #endif
                for (int j = 0; j < matrix_size*matrix_size; j++){
                        a[k*matrix_size*matrix_size + j] = j%9+1;
                        b[k*matrix_size*matrix_size + j] = j%10;
                        c_cpu[k*matrix_size*matrix_size + j] = 0;
                        c_gpu[k*matrix_size*matrix_size + j] = 0;
                        
                }
                
        }
        
	cudaEventRecord(stp_stop,0);
	cudaEventSynchronize(stp_stop);
        
	cudaEventElapsedTime(&time, stp_start, stp_stop);
	printf("\tSetup Time: %.2f ms\n", time);

        //
        // CPU Calculation
        //////////////////
        
	printf("Running sequential job.\n");
	cudaEventRecord(cpu_start,0);
        
        // Calculate C in the CPU
        for (int k = 0; k < num_matrices; k++) {
                // If first element is different than 0 do the computation
                if (a[k*matrix_size*matrix_size] != 0){
                        for (int j = 0; j < matrix_size; j++){
                                //If first value in the row of the matrix, do addition
                                if (a[k*matrix_size*matrix_size+j*matrix_size] < threshold){
                                        for (int i = 0; i < matrix_size; i++){
                                                int index = k*matrix_size*matrix_size+j*matrix_size+i;
                                                c_cpu[index] = a[index] + b[index];
                                        }
                                //Do subtraction
                                } else {
                                        for (int i = 0; i < matrix_size; i++){
                                                int index = k*matrix_size*matrix_size+j*matrix_size+i;
                                                c_cpu[index] = a[index] - b[index];
                                        }
                                }
                        }                        
                }
        }
        
	cudaEventRecord(cpu_stop,0);
	cudaEventSynchronize(cpu_stop);
        
	cudaEventElapsedTime(&time, cpu_start, cpu_stop);
	printf("\tSequential Job Time: %.2f ms\n", time);
      
        //
        // GPU Calculation
        //////////////////
        
        printf("Running parallel job.\n");
        
        int grid_size    = ((num_matrices-1)/block_size) + 1;
	
        cudaEventRecord(gpu_start,0);
        
        // allocate the memory on the GPU
        cudaMalloc( (void**)&dev_a,    num_matrices * matrix_size * matrix_size * sizeof(char) );
        cudaMalloc( (void**)&dev_b,    num_matrices * matrix_size * matrix_size * sizeof(char) );
        cudaMalloc( (void**)&dev_c,    num_matrices * matrix_size * matrix_size * sizeof(char) );

        // set arrays to 0
        cudaMemset(dev_a,      0, num_matrices * matrix_size * matrix_size * sizeof(char));
        cudaMemset(dev_b,      0, num_matrices * matrix_size * matrix_size * sizeof(char));
        cudaMemset(dev_c,      0, num_matrices * matrix_size * matrix_size * sizeof(char));
        
        // copy the 'data' to the GPU
        cudaMemcpy( dev_a, a, num_matrices * matrix_size * matrix_size * sizeof(char), cudaMemcpyHostToDevice );
        cudaMemcpy( dev_b, b, num_matrices * matrix_size * matrix_size * sizeof(char), cudaMemcpyHostToDevice );
        
        // run kernel
        cudaEventRecord(ker_start,0);
        calculation<<<grid_size,block_size>>>(  dev_a, 
                                                dev_b, 
                                                dev_c,
                                                num_matrices,                                                
                                                matrix_size );
        cudaEventRecord(ker_stop,0);
                                                        
        // copy the array 'c' back from the GPU to the CPU
        cudaMemcpy( c_gpu, dev_c, num_matrices * matrix_size * matrix_size * sizeof(char), cudaMemcpyDeviceToHost );

	cudaEventRecord(gpu_stop,0);
	cudaEventSynchronize(gpu_stop);

	cudaEventElapsedTime(&time    , gpu_start, gpu_stop);
	cudaEventElapsedTime(&ker_time, ker_start, ker_stop);
	printf("\tParallel Job Time: %.2f ms\n", time);
	printf("\tKernel Exec. Time: %.2f ms\n", ker_time);

        //
        // Compare Results
        //////////////////
        int error = 0;
        for (int i = 0; i < num_matrices * matrix_size * matrix_size; i++) {
                if (c_cpu[i] != c_gpu[i]){
                        error = 1;
                        printf( "Error starting element %d, %d != %d\n", i, c_gpu[i], c_cpu[i] );    
                }
		if (error) break; 
        }
        
        if (error == 0){
                printf ("Correct result. No errors were found.\n");
        }

        //
        // Free resources
        //////////////////
        
        // free the memory allocated on the GPU
        cudaFree( dev_a );
        cudaFree( dev_b );
        cudaFree( dev_c );
        
        // free cuda events
        cudaEventDestroy (cpu_start);
	cudaEventDestroy (gpu_start);
	cudaEventDestroy (ker_start);
        
	cudaEventDestroy (cpu_stop);
	cudaEventDestroy (gpu_stop);
	cudaEventDestroy (ker_stop);
        
        // free CPU memory        
	free(a);    
	free(b);
	free(c_cpu);
	free(c_gpu);
	
        return 0;
}