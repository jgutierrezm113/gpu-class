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

#define DEFAULT_BRIGHTNESS  50

char inc = DEFAULT_BRIGHTNESS;

typedef struct __gpuData {
	
	unsigned int size;
	unsigned char* intensity;
	unsigned char* result;
	unsigned char* resultOnCPU;
	
} gpuData;

gpuData gpu;

// Modify the value of max_iterations
void modBrightness (unsigned int value);

inline cudaError_t checkCuda(cudaError_t result);

__global__ void brightnessAlgorithm(unsigned char *intensity, 
		unsigned char *result,
		unsigned int brightness);
			 
unsigned char *brightness(unsigned char *intensity,
		unsigned int height, 
		unsigned int width);

unsigned char *brightnessWarmup(unsigned char *intensity,
		unsigned int height, 
		unsigned int width);
		
#endif
