/* 
 * Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 *   
 */
 
#ifndef CONFIG_H
#define CONFIG_H

/* 
 * Configuration Options:
 * 	Variable controlling Masks and Bits (for efficient coding)
 * 	Allowed values:
		- BLOCK_TILE_SIZE = 32 | 16 | 8 | 4 | 2
 */
#define BLOCK_TILE_SIZE 16

// Defining verbosity of run
#define VERBOSE

// Defining important constants
#define MAX_LABELS_PER_IMAGE 32

// Threshold of max number of iterations of analysis
#define MAX_ITER 5000

// timing directives
#define CUDA_TIMING
#define KERNEL_TIMING

// Debug Cuda Errors
//#define DEBUG

// Warmup directive 
#define WARMUP

#if BLOCK_TILE_SIZE >= 32
	#undef BLOCK_TILE_SIZE
	
	#define BLOCK_TILE_SIZE   32
	#define TILE_SIZE	  32
	
#elif BLOCK_TILE_SIZE >= 16
	#define TILE_SIZE	  16
	
#elif BLOCK_TILE_SIZE >= 8
	#define TILE_SIZE	   8

#elif BLOCK_TILE_SIZE >= 4
	#define TILE_SIZE	   4
	
#elif BLOCK_TILE_SIZE >= 2
	#define TILE_SIZE	   2

#else // Default just in case
	#undef BLOCK_TILE_SIZE 	
	#define BLOCK_TILE_SIZE   16
	#define TILE_SIZE	  16
#endif

#endif

