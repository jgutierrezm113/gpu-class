#include <stdio.h>
#include <stdlib.h>

__global__ void add( int *a, int *b, int *c, int vector_size ) {
    
    // Calculate the index in the vector for the thread using the internal variables
    int tid = blockIdx.x*blockDim.x + threadIdx.x; // HERE
    
    // This if statement is added in case we have more threads executing
    // Than number of elements in the vectors. How can this help?
    if (tid < vector_size){
        
        // Compute the addition
        // HERE
        c[tid] = a[tid] + b[tid];
        
    }
}

int main( int argc, char* argv[] ) { 

    // Parse Input arguments

    // Check the number of arguments
    if (argc <= 2) {
        // Tell the user how to run the program
        printf ("Usage: %s vector_size block_size\n", argv[0]);
        // "Usage messages" are a conventional way of telling the user
        // how to run a program if they enter the command incorrectly.
        return 1;
    }
    
    // Set GPU Variables based on input arguments
    int vector_size = atoi(argv[1]);
    int block_size  = atoi(argv[2]);
    int grid_size   = ((vector_size-1)/block_size) + 1;

    // Set device that we will use for our cuda code
    cudaSetDevice(0);
        
    // Time Variables
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);

    // Input Arrays and variables
    int *a        = new int [vector_size]; 
    int *b        = new int [vector_size]; 
    int *c_cpu    = new int [vector_size]; 
    int *c_gpu    = new int [vector_size];

    // Pointers in GPU memory
    int *dev_a;
    int *dev_b;
    int *dev_c;

    // fill the arrays 'a' and 'b' on the CPU
    printf("Initializing input arrays.\n");
    for (int i = 0; i < vector_size; i++) {
        a[i] = rand()%10;
        b[i] = rand()%10;
    }

    //
    // CPU Calculation
    //////////////////

    printf("Running sequential job.\n");
    cudaEventRecord(start,0);

    // Calculate C in the CPU
    for (int i = 0; i < vector_size; i++) {
            c_cpu[i] = a[i] + b[i];
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("\tSequential Job Time: %.2f ms\n", time);

    // allocate the memory on the GPU
    // HERE
    cudaMalloc (&dev_a, vector_size*sizeof(int));
    cudaMalloc (&dev_b, vector_size*sizeof(int));
    cudaMalloc (&dev_c, vector_size*sizeof(int));

    // copy the arrays 'a' and 'b' to the GPU
    // HERE
    cudaMemcpy (dev_a, a, vector_size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy (dev_b, b, vector_size*sizeof(int), cudaMemcpyHostToDevice);

    //
    // GPU Calculation
    ////////////////////////

    printf("Running parallel job.\n");

    cudaEventRecord(start,0);

    // call the kernel
    // HERE
    add<<<grid_size, block_size>>>(dev_a, dev_b, dev_c, vector_size);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    printf("\tParallel Job Time: %.2f ms\n", time);

    // copy the array 'c' back from the GPU to the CPU
    // HERE (there's one more at the end, don't miss it!)
    cudaMemcpy (c_gpu, dev_c, vector_size*sizeof(int), cudaMemcpyDeviceToHost);
    
    // compare the results
    int error = 0;
    for (int i = 0; i < vector_size; i++) {
        if (c_cpu[i] != c_gpu[i]){
            error = 1;
            printf( "Error starting element %d, %d != %d\n", i, c_gpu[i], c_cpu[i] );    
        }
        if (error) break; 
    }

    if (error == 0){
        printf ("Correct result. No errors were found.\n");
    }

    // free CPU data
    free (a);
    free (b);
    free (c_cpu);
    free (c_gpu);

    // free the memory allocated on the GPU
    // HERE
    cudaFree (dev_a);
    cudaFree (dev_b);
    cudaFree (dev_c);

    return 0;
}

