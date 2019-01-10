#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>

#include "config.h"

using namespace cv;
using namespace std;

// Timer function
double CLOCK() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC,  &t);
	return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

unsigned int imageLocation (unsigned int x, 
		unsigned int y, 
		unsigned int gridXSize){
	unsigned int location =   ( x&TTSMask )                               |
                             (( y&TTSMask )          <<  TTSB        )    |
                            ((( x>>TTSB ) &BTSMask ) << (TTSB+TTSB)  )    |
                            ((( y>>TTSB ) &BTSMask ) << (BTSB+TTSB+TTSB)) |
                             (( x>>TSB  )            << (TSB+TSB)    ) ;
	location += 	         (( y>>TSB  )            << (TSB+TSB)    )*gridXSize;
	return (location);
}

extern void gpu_function (unsigned char *data, 
                          unsigned int height, 
                          unsigned int width,
                          int threshold);
                          
extern void gpu_warmup (unsigned char *data, 
                        unsigned int height, 
                        unsigned int width);

int main( int argc, const char** argv ) {
        
    double start_cpu, finish_cpu;
    double start_gpu, finish_gpu;
    
    // Check inputs
    if (argc != 3){
        cout << "Incorrect number of inputs" << endl;
        cout << argv[0] << " <input file> <threshold>" << endl;
        return -1;
    }
    
    // Read input image from argument
    Mat input_image = imread(argv[1], IMREAD_COLOR);

    if (input_image.empty()){
        cout << "Image cannot be loaded..!!" << endl;
        return -1;
    }
    
    int threshold = atoi(argv[2]);

    // Convert the color image to grayscale image
    cvtColor(input_image, input_image, COLOR_BGR2GRAY); 
    
    unsigned int height = input_image.rows;
    unsigned int  width = input_image.cols;
    
    cout << "Image size: " << height << "x" << width << endl;
    
    // Construct padded image
    Mat padded;
    
    int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	unsigned int XSize = gridXSize*TILE_SIZE;
	unsigned int YSize = gridYSize*TILE_SIZE;
     
    padded.create(YSize, XSize, input_image.type());
    padded.setTo(cv::Scalar::all(0));
	    
	for (unsigned int y = 0 ; y < YSize ; y++){
		for (unsigned int x = 0 ; x < XSize ; x++){
				unsigned int newLocation = imageLocation (x, y, gridXSize);
                unsigned int newx = newLocation % XSize;
                unsigned int newy = newLocation / XSize;
			if (x < width && y < height) {
				padded.at<uchar>(newy, newx) = input_image.at<uchar>(y, x);
			}
		}
	}
    
    ///////////////////////
    // START CPU Processing
    ///////////////////////
    start_cpu = CLOCK();
    
    // CPU Execution
    Mat cpu_output = input_image.clone();
    
    for(int y = 0 ; y < (int) height ; y++) {
        for(int x = 0 ; x < (int) width ; x++) {
            
            if (y >= 1 && y < (int) height - 1 && x >= 1 && x < (int) width - 1){
                // Algorithm 
                /////////////////////////////////////////////////////////////////////

                int sum1 = input_image.at<uchar>(y-1, x+1) - 
                           input_image.at<uchar>(y-1, x-1) + 
                       2 * input_image.at<uchar>(y  , x+1) - 
                       2 * input_image.at<uchar>(y  , x-1) + 
                           input_image.at<uchar>(y+1, x+1) - 
                           input_image.at<uchar>(y+1, x-1);

                int sum2 = input_image.at<uchar>(y-1, x-1) + 
                       2 * input_image.at<uchar>(y-1, x  ) + 
                           input_image.at<uchar>(y-1, x+1) - 
                           input_image.at<uchar>(y+1, x-1) - 
                       2 * input_image.at<uchar>(y+1, x  ) - 
                           input_image.at<uchar>(y+1, x+1);

                int magnitude =  sqrt( (float) (sum1*sum1 + sum2*sum2));

                if (magnitude > threshold)
                    cpu_output.at<uchar>(y, x) = 255;
                else
                    cpu_output.at<uchar>(y, x) = 0;
            } else
                    cpu_output.at<uchar>(y, x) = 0;
        }
	}
    
    finish_cpu = CLOCK();
        
    ///////////////////////
    // START GPU Warmup
    ///////////////////////
    
    Mat temp = padded.clone();
    
    gpu_warmup((unsigned char *) temp.data, 
                                 height, 
                                 width);
    
    ///////////////////////
    // START GPU Processing
    ///////////////////////
    
    start_gpu = CLOCK();

    Mat gpu_output = padded.clone();

    gpu_function((unsigned char *) gpu_output.data, 
                                   height, 
                                   width,
                                   threshold);

    finish_gpu = CLOCK();

    // Calculate % difference between GPU and CPU
    unsigned int wrong_pixels = 0;
    for(unsigned int y = 0; y < height; y++){
        for(unsigned int x = 0; x < width; x++){
            unsigned char cpu_pixel = cpu_output.data[width * y + x ];
            unsigned char gpu_pixel = gpu_output.data[imageLocation(x, y, gridXSize)];
            
            // If percentage difference is more than a threshold (10% diff), value is wrong
            if (abs(cpu_pixel - gpu_pixel) > 20)
                wrong_pixels++;
        }
    }
    
    cout << "CPU execution time: " << finish_cpu - start_cpu << " ms" << endl;
    cout << "GPU execution time: " << finish_gpu - start_gpu << " ms" << endl;
    cout << "Percentage difference: " << wrong_pixels*100.0/(height*width) << " %\n";

    
    imwrite ("input.jpg", input_image);   
    imwrite ("output_cpu.jpg", cpu_output);
    
    Mat gpu_output_to_file(cv::Size(input_image.cols, input_image.rows), CV_8U, Scalar(0));
    for (unsigned int y = 0 ; y < height ; y++){
		for (unsigned int x = 0 ; x < width ; x++){
			unsigned int newLocation = imageLocation (x, y, gridXSize);
            unsigned int newx = newLocation % XSize;
            unsigned int newy = newLocation / XSize;
			
			gpu_output_to_file.at<uchar>(y, x) = gpu_output.at<uchar>(newy, newx);
		}
	} 
    imwrite ("output_gpu.jpg", gpu_output_to_file);

    return 0;
}
