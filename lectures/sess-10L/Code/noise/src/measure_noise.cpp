#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <math.h>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    
    // Check inputs
    if (argc != 2){
        cout << "Incorrect number of inputs." << endl;
        cout << argv[0] << " <input file>" << endl;
        return -1;
    }
    
    // Read input image from argument
    Mat input_image = imread(argv[1], IMREAD_COLOR);

    if (input_image.empty()){
        cout << "Image cannot be loaded..!!" << endl;
        return -1;
    }
    
    // Convert the color image to grayscale image
    cvtColor(input_image, input_image, COLOR_BGR2GRAY); 
    
    Mat mean, stddev;
    meanStdDev(input_image, mean, stddev);
    
    cout << "The average noise (stddev) is: " << stddev.at<double>(0,0) << endl;
    //cout << "The average value is: " << mean.at<double>(0,0) << endl;

    return 0;
}
