#ifndef CONFIG_H
#define CONFIG_H

#define BLOCK_TILE_SIZE  16
#define BTSMask		     15
#define BTSB		      4

#define THREAD_TILE_SIZE  2
#define TTSMask		      1
#define TTSB 		      1

#define TILE_SIZE	     32
#define TSMask		     31
#define TSB		          5

//#define CUDA_TIMING
#define WARMUP

#endif