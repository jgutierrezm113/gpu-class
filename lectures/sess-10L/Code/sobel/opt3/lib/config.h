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

/* 
 * Variable controlling Masks and Bits (for efficiency)
 *
 * Definitions as follows:
 * 	- BTSMask: 	BLOCK_TILE_SIZE-1 
 * 	- BTSB: 	BLOCK_TILE_SIZE bits (2^BTSB = BLOCK_TILE_SIZE)
 * 	- TILE_SIZE: 	BLOCK_TILE_SIZE*THREAD_TILE_SIZE
 * 	- TSMask: 	TILE_SIZE-1
 * 	- TSB: 		TILE_SIZE bits (2^TSB = TILE_SIZE)
 *	- TTSMask:	THREAD_TILE_SIZE-1
 *	- TTSB:		THREAD_TILE_SIZE bits (2^TTSB = THREAD_TILE_SIZE)
 *
 */
#if BLOCK_TILE_SIZE >= 32
	#undef BLOCK_TILE_SIZE
	
	#define BLOCK_TILE_SIZE   32
	#define BTSMask		  31
	#define BTSB		   5
	
	#define THREAD_TILE_SIZE   2
	#define TTSMask		   1
	#define TTSB 		   1

	#define TILE_SIZE	  64
	#define TSMask		  63
	#define TSB		   6
	
#elif BLOCK_TILE_SIZE >= 16
	#define BTSMask		  15
	#define BTSB		   4
	
	#define THREAD_TILE_SIZE   2
	#define TTSMask		   1
	#define TTSB 		   1

	#define TILE_SIZE	  32
	#define TSMask		  31
	#define TSB		   5
	
#elif BLOCK_TILE_SIZE >= 8
	#define BTSMask		   7
	#define BTSB		   3
	
	#define THREAD_TILE_SIZE   2
	#define TTSMask		   1
	#define TTSB 		   1

	#define TILE_SIZE	  16
	#define TSMask		  15
	#define TSB		   4

#elif BLOCK_TILE_SIZE >= 4
	#define BTSMask		   3
	#define BTSB		   2
	
	#define THREAD_TILE_SIZE   2
	#define TTSMask		   1
	#define TTSB 		   1
	
	#define TILE_SIZE	   8
	#define TSMask		   7
	#define TSB		   3
	
#elif BLOCK_TILE_SIZE >= 2
	#define BTSMask		   1
	#define BTSB		   1
	
	#define THREAD_TILE_SIZE   2
	#define TTSMask		   1
	#define TTSB 		   1
	 
	#define TILE_SIZE	   4
	#define TSMask		   3
	#define TSB		   2

#else // Default just in case
	#undef BLOCK_TILE_SIZE 	
	#define BLOCK_TILE_SIZE   16
	#define BTSMask		  15
	#define BTSB		   4
	
	#define THREAD_TILE_SIZE   2
	#define TTSMask		   1
	#define TTSB 		   1
	
	#define TILE_SIZE	  32
	#define TSMask		  31
	#define TSB		   5
#endif

#endif

