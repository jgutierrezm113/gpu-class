/* Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 * 
 * Sobel Algorithm Implementation 
 *  
 */

#include "sobel.h"

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
#endif
    return result;
}

using namespace std;

void modThreshold (unsigned int value){
    threshold = value;
}

texture<unsigned char, 2, cudaReadModeElementType> tex8u;

/*
 * Sobel Kernel
 */
__global__ void sobelAlgorithmNico(unsigned char *intensity, 
        unsigned char *result,
        unsigned int threshold){
    
    
    const int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    const int y = blockIdx.y*TILE_SIZE+threadIdx.y;

    const int xsize = TILE_SIZE*gridDim.x;
    const int ysize = TILE_SIZE*gridDim.y;
    
    // Skip this thread if outside the size
    if (x < 0 || x > xsize-0 || y > ysize-0 || y < 0)
        return;

    const int location= y*xsize+x;

    const int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    const int sobel_y[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };	


    int sum[2] ={0};

    for (int i=0;i<3;++i){
        for (int j=0;j<3;++j){
            const int x_elem = i+x-1;
            const int y_elem = j+y-1;
            sum[0] += tex2D(tex8u,x_elem,y_elem)*sobel_x[i][j];
            sum[1] += tex2D(tex8u,x_elem,y_elem)*sobel_y[i][j];
        }
    }

    const int magnitude =  sum[0]*sum[0] + sum[1]*sum[1];

    if (magnitude > threshold)
        result[location] = 255;
    else
        result[location] = 0;
    
}

/*
 * Sobel Kernel
 */
__global__ void sobelAlgorithm(unsigned char *intensity, 
        unsigned char *result,
        unsigned int threshold){

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int x = bx*TILE_SIZE+tx;
    int y = by*TILE_SIZE+ty;

    int xsize = TILE_SIZE*gridDim.x;
    int ysize = TILE_SIZE*gridDim.y;


    if (y > 1 && y < ysize-1 && x > 1 && x < xsize-1){

        int location = 	y*xsize+x;

        int sum1 =  intensity[ xsize * (y-1) + x+1 ] - 
            intensity[ xsize * (y-1) + x-1 ] + 
            2 * intensity[ xsize * (y)   + x+1 ] - 
            2 * intensity[ xsize * (y)   + x-1 ] + 
            intensity[ xsize * (y+1) + x+1 ] - 
            intensity[ xsize * (y+1) + x-1 ];

        int sum2 = intensity[ xsize * (y-1) + x-1 ] + 
            2 * intensity[ xsize * (y-1) + x   ] + 
            intensity[ xsize * (y-1) + x+1 ] - 
            intensity[ xsize * (y+1) + x-1 ] - 
            2 * intensity[ xsize * (y+1) + x   ] - 
            intensity[ xsize * (y+1) + x+1 ];

        int magnitude =  sum1*sum1 + sum2*sum2;

        if (magnitude > threshold)
            result[location] = 255;
        else
            result[location] = 0;
    }
}

unsigned char *sobel(unsigned char *intensity,
        unsigned int height, 
        unsigned int width){

#if defined(DEBUG)
    printf("Printing input data\n");
    printf("Height: %d\n", height);
    printf("Width: %d\n", width);
#endif

    int gridXSize = 1 + (( width - 1) / TILE_SIZE);
    int gridYSize = 1 + ((height - 1) / TILE_SIZE);

    int XSize = gridXSize*TILE_SIZE;
    int YSize = gridYSize*TILE_SIZE;

    // Both are the same size (CPU/GPU).
    gpu.size = XSize*YSize;

    // Allocate arrays in GPU memory
#if defined(VERBOSE)
    printf ("Allocating arrays in GPU memory.\n");
#endif

#if defined(CUDA_TIMING)
    float Ttime;
    TIMER_CREATE(Ttime);
    TIMER_START(Ttime);
#endif


    // ----- ALLOCATE MEMORY ON DEVICE
    checkCuda(cudaMalloc((void**)&gpu.intensity              , gpu.size*sizeof(char)));
    checkCuda(cudaMalloc((void**)&gpu.result                 , gpu.size*sizeof(char)));

    checkCuda(cudaMemset(gpu.result , 0 , gpu.size));
    checkCuda(cudaMemset(gpu.intensity , 0 , gpu.size));

    checkCuda(cudaMemcpy(gpu.intensity,
            intensity, 
            gpu.size*sizeof(char),
            cudaMemcpyHostToDevice));
    
    // TEXTURE - Create texture parameters
    tex8u.addressMode[0] = cudaAddressModeMirror; // 
    tex8u.addressMode[1] = cudaAddressModeMirror; // Pad with zeros 
    tex8u.filterMode = cudaFilterModePoint;
    tex8u.normalized = false;

    // TEXTURE - Bind Texture
    size_t pitch = sizeof(char)*width;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    checkCuda(cudaBindTexture2D(NULL, tex8u,
                gpu.intensity, channelDesc,
                width, height, pitch));



    //// Allocate result array in CPU memory
    gpu.resultOnCPU = new unsigned char[gpu.size];

    checkCuda(cudaDeviceSynchronize());



#if defined(CUDA_TIMING)
    float Ktime;
    TIMER_CREATE(Ktime);
    TIMER_START(Ktime);
#endif

#if defined(VERBOSE)
    printf("Running algorithm on GPU.\n");
#endif

    dim3 dimGrid(gridXSize/GRID_SPLIT, gridYSize/GRID_SPLIT);
    dim3 dimBlock(BLOCK_TILE_SIZE, BLOCK_TILE_SIZE);

    // Launch kernel to begin image segmenation
    sobelAlgorithmNico<<<dimGrid, dimBlock>>>(gpu.intensity, 
            gpu.result,
            threshold
            );

    checkCuda(cudaDeviceSynchronize());

#if defined(CUDA_TIMING)
    TIMER_END(Ktime);
    printf("Kernel Execution Time: %f ms\n", Ktime);
#endif

    // Retrieve results from the GPU
    checkCuda(cudaMemcpy(gpu.resultOnCPU, 
                gpu.result, 
                gpu.size*sizeof(char), 
                cudaMemcpyDeviceToHost));
    
    
    // Free resources and end the program
    checkCuda(cudaUnbindTexture(tex8u));
    checkCuda(cudaFree(gpu.intensity));
    checkCuda(cudaFree(gpu.result));

#if defined(CUDA_TIMING)
    TIMER_END(Ttime);
    printf("Total GPU Execution Time: %f ms\n", Ttime);
#endif

    return(gpu.resultOnCPU);

}

unsigned char *sobelWarmup(unsigned char *intensity,
        unsigned int height, 
        unsigned int width){

    int gridXSize = 1 + (( width - 1) / TILE_SIZE);
    int gridYSize = 1 + ((height - 1) / TILE_SIZE);

    int XSize = gridXSize*TILE_SIZE;
    int YSize = gridYSize*TILE_SIZE;

    // Both are the same size (CPU/GPU).
    gpu.size = XSize*YSize;

    // Allocate arrays in GPU memory
    checkCuda(cudaMalloc((void**)&gpu.intensity              , gpu.size*sizeof(char)));
    checkCuda(cudaMalloc((void**)&gpu.result                 , gpu.size*sizeof(char)));

    checkCuda(cudaMemset(gpu.result , 0 , gpu.size));
    checkCuda(cudaMemset(gpu.intensity , 0 , gpu.size));

    checkCuda(cudaMemcpy(gpu.intensity,
            intensity,
            gpu.size*sizeof(char),
            cudaMemcpyHostToDevice));
    
    // TEXTURE - Create texture parameters
    tex8u.addressMode[0] = cudaAddressModeBorder; // 
    tex8u.addressMode[1] = cudaAddressModeBorder; // Pad with zeros 
    tex8u.filterMode = cudaFilterModePoint;
    tex8u.normalized = false;

    // TEXTURE - Bind Texture
    size_t pitch = sizeof(char)*width;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    checkCuda(cudaBindTexture2D(NULL, tex8u,
                gpu.intensity, channelDesc,
                width, height, pitch));


    // Allocate result array in CPU memory
    gpu.resultOnCPU = new unsigned char[gpu.size];

    checkCuda(cudaDeviceSynchronize());

    dim3 dimGrid(gridXSize/GRID_SPLIT, gridYSize/GRID_SPLIT);
    dim3 dimBlock(BLOCK_TILE_SIZE, BLOCK_TILE_SIZE);

    // Launch kernel to begin image segmenation
    sobelAlgorithmNico<<<dimGrid, dimBlock>>>(gpu.intensity, 
            gpu.result,
            threshold);

    checkCuda(cudaDeviceSynchronize());

    // Retrieve results from the GPU
    checkCuda(cudaMemcpy(gpu.resultOnCPU, 
                gpu.result, 
                gpu.size*sizeof(char), 
                cudaMemcpyDeviceToHost));

    // Free resources and end the program
    checkCuda(cudaUnbindTexture(tex8u));
    checkCuda(cudaFree(gpu.intensity));
    checkCuda(cudaFree(gpu.result));

    return(gpu.resultOnCPU);

}
