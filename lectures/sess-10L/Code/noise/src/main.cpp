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

extern void gpu_function (unsigned char *data, 
                          unsigned int height, 
                          unsigned int width,
                          int inc);
                          
extern void gpu_warmup (unsigned char *data, 
                        unsigned int height, 
                        unsigned int width);

int main( int argc, const char** argv ) {
        
    double start_cpu, finish_cpu;
    double start_gpu, finish_gpu;
    
    // Check inputs
    if (argc != 2){
        cout << "Incorrect number of inputs" << endl;
        cout << argv[0] << " <input file>" << endl;
        return -1;
    }
    
    // Read input image from argument
    Mat input_image = imread(argv[1], IMREAD_COLOR);

    if (input_image.empty()){
        cout << "Image cannot be loaded..!!" << endl;
        return -1;
    }
    
    int inc = 0;

    // Convert the color image to grayscale image
    cvtColor(input_image, input_image, COLOR_BGR2GRAY); 
    
    unsigned int height = input_image.rows;
    unsigned int  width = input_image.cols;
    
    cout << "Image size: " << height << "x" << width << endl;
    
    // Construct padded image
    Mat padded;
    
    int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
     
    padded.create(YSize, XSize, input_image.type());
    padded.setTo(cv::Scalar::all(0));

    input_image.copyTo(padded(Rect(0, 0, input_image.cols, input_image.rows)));
    
    ///////////////////////
    // START CPU Processing
    ///////////////////////
    start_cpu = CLOCK();
    
    // CPU Execution
    Mat cpu_output = input_image.clone();

    // Apply the filter (excluding borders)
    for (int j = 0; j < input_image.rows; j++){
        for (int k = 0; k < input_image.cols; k++){
            if (j >= 1 && j < input_image.rows-1 &&
                k >= 1 && k < input_image.cols-1){
                vector<unsigned char> values;
                for (int jj = 0; jj < 3; jj++){
                    for (int kk = 0; kk < 3; kk++){
                        values.push_back(input_image.at<uchar>(j-(1-jj),k-(1-kk)));                
                    }
                }
                sort (values.begin(), values.end());
                cpu_output.at<uchar>(j,k) = values[4];    
            }
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
                                   inc);

    finish_gpu = CLOCK();

    // Calculate % difference between GPU and CPU
    unsigned int wrong_pixels = 0;
    for(unsigned int j = 0; j < height; j++){
        for(unsigned int i = 0; i < width; i++){
            unsigned char cpu_pixel = cpu_output.data[width * j + i ];
            unsigned char gpu_pixel = gpu_output.data[width * j + i ];
            
            // If percentage difference is more than a threshold (10% diff), value is wrong
            if (abs(cpu_pixel - gpu_pixel) > 0)
                wrong_pixels++;
        }
    }
    
    cout << "CPU execution time: " << finish_cpu - start_cpu << " ms" << endl;
    cout << "GPU execution time: " << finish_gpu - start_gpu << " ms" << endl;
    cout << "Percentage difference: " << wrong_pixels*100.0/(height*width) << " %\n";

    imwrite ("input.jpg", input_image);
    cv::Mat subImg1 = cpu_output(cv::Range(0, input_image.rows), cv::Range(0, input_image.cols));    
    imwrite ("output_cpu.jpg", subImg1);
    cv::Mat subImg2 = gpu_output(cv::Range(0, input_image.rows), cv::Range(0, input_image.cols));   
    imwrite ("output_gpu.jpg", subImg2);

    return 0;
}
