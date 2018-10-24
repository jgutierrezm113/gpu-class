/* 
 * Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 * 
 * Sobel Algorithm Header
 *
 */
 
#ifndef ALG_H
#define ALG_H

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <cuda.h>

#include "../cuda/timing.h"
#include "../config.h"

#define DEFAULT_THRESHOLD  3000

unsigned int threshold = DEFAULT_THRESHOLD;

typedef struct __gpuData {
	
	unsigned int size;
	unsigned char* intensity;
	unsigned char* result;
	unsigned char* resultOnCPU;
	
} gpuData;

gpuData gpu;

// Modify the value of max_iterations
void modThreshold (unsigned int value);

inline cudaError_t checkCuda(cudaError_t result);

__global__ void sobelAlgorithm(unsigned char *intensity, 
		unsigned char *result,
		unsigned int threshold);
			 
unsigned char *sobel(unsigned char *intensity,
		unsigned int height, 
		unsigned int width);

unsigned char *sobelWarmup(unsigned char *intensity,
		unsigned int height, 
		unsigned int width);
		
#endif
