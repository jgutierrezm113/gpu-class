#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;

extern double CLOCK();

extern void histogram_gpu(unsigned char *data, 
                          unsigned int height, 
                          unsigned int width);
                          
extern void histogram_gpu_warmup(unsigned char *data, 
                          unsigned int height, 
                          unsigned int width);

int main( int argc, const char** argv ) {
        
    double start_cpu, finish_cpu;
    double start_gpu, finish_gpu;
    
    // Read input image from argument
    Mat input_image = imread(argv[1], IMREAD_COLOR);

    if (input_image.empty()){
        cout << "Image cannot be loaded..!!" << endl;
        return -1;
    }

    // Convert the color image to grayscale image
    cvtColor(input_image, input_image, COLOR_BGR2GRAY); 
    
    unsigned int height = input_image.rows;
    unsigned int  width = input_image.cols;
    
    ///////////////////////
    // START CPU Processing
    ///////////////////////
    start_cpu = CLOCK();
    
    // Equalize the histogram
    Mat img_hist_equalized_cpu;
    equalizeHist(input_image, img_hist_equalized_cpu); 

    finish_cpu = CLOCK();
    
    ///////////////////////
    // START GPU Warmup
    ///////////////////////
    
    Mat temp = input_image.clone();

    histogram_gpu_warmup((unsigned char *) temp.data, 
                                    height, 
                                    width);
    
    ///////////////////////
    // START GPU Processing
    ///////////////////////
    
    start_gpu = CLOCK();

    Mat img_hist_equalized_gpu = input_image.clone();

    histogram_gpu((unsigned char *) img_hist_equalized_gpu.data, 
                                    height, 
                                    width);

    finish_gpu = CLOCK();

    // Calculate % difference between GPU and CPU
    unsigned int wrong_pixels = 0;
    for(unsigned int j = 0; j < height; j++){
        for(unsigned int i = 0; i < width; i++){
            unsigned char cpu_pixel = img_hist_equalized_cpu.data[width * j + i ];
            unsigned char gpu_pixel = img_hist_equalized_gpu.data[width * j + i ];
            
            // If percentage difference is more than a threshold (10% diff), value is wrong
            if (abs(cpu_pixel - gpu_pixel) > 25)
                wrong_pixels++;
        }
    }
    
    cout << "CPU execution time: " << finish_cpu - start_cpu << " ms" << endl;
    cout << "GPU execution time: " << finish_gpu - start_gpu << " ms" << endl;
    cout << "Percentage difference: " << wrong_pixels*100.0/(height*width) << " %\n";
            
    // Create windows
    //namedWindow("Original Image", WINDOW_NORMAL );
    //namedWindow("Histogram Equalized CPU", WINDOW_NORMAL );
    //namedWindow("Histogram Equalized GPU", WINDOW_NORMAL );
    
    // Show images (or store results)
    //imshow("Original Image", input_image);
    //imshow("Histogram Equalized CPU", img_hist_equalized_cpu);
    //imshow("Histogram Equalized GPU", img_hist_equalized_gpu);

    // Resize windows
    //resizeWindow("Original Image", 640, 480 );
    //resizeWindow("Histogram Equalized CPU", 640, 480 );
    //resizeWindow("Histogram Equalized GPU", 640, 480 );
   
    imwrite ("input_baw.jpg", input_image); 
    imwrite ("output_cpu.jpg", img_hist_equalized_cpu);
    imwrite ("output_gpu.jpg", img_hist_equalized_gpu);

    //waitKey(0); //wait for key press

    //destroyAllWindows(); //destroy all open windows

    return 0;
}
