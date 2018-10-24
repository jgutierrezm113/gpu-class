#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void initialize (int N, float *a, float *b, float *c){
	int i = threadIdx.x+blockIdx.x*blockDim.x;
	if (i < N){
		c[i] = 0;
		a[i] = 1 + i;
		b[i] = 1 - i;
	}
}

__global__ void addVectors (int N, float *a, float *b, float *c){
	int i = threadIdx.x+blockIdx.x*blockDim.x;
	if (i < N){
		c[i] = a[i] + b[i];
	}
}

int main (int argc, char **argv){
	
	if (argc != 2) exit (1);
	int N = atoi(argv[1]);

	float *a, *b, *c;
	cudaMallocManaged (&a, N*sizeof(float));
	cudaMallocManaged (&b, N*sizeof(float));
	cudaMallocManaged (&c, N*sizeof(float));

	dim3 block(1024);
	dim3 grid((N-1)/1024+1);

	initialize<<<grid, block>>>(N,a,b,c);
	
	addVectors<<<grid, block>>>(N,a,b,c);

	cudaDeviceSynchronize ();

	for (int i = 0; i < 5; i++) {
		printf("%f\n", c[i]);
	}

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}
