/* Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 * 
 * Sobel Algorithm Main program 
 *  
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <ctime>

#include "img/imghandler.h"
#include "alg/locationhandler.h"
#include "config.h"

typedef struct __cpuData {
	
	unsigned int height;
	unsigned int width;
	unsigned int gridXSize;
	unsigned int gridYSize;
	unsigned int size;	

	unsigned char* intensity;
	unsigned char* result;
	
} cpuData;

cpuData cpu;

extern unsigned char *sobel(unsigned char *intensity, 
		unsigned int height, 
		unsigned int width);

extern unsigned char *sobelWarmup(unsigned char *intensity, 
		unsigned int height, 
		unsigned int width);
		
extern void modThreshold (unsigned int value);

extern unsigned int imageLocation (unsigned int x, 
		unsigned int y, 
		unsigned int gridXSize);

int main(int argc, char* argv[]) {

	// Files needed
	char* imageFile = NULL;

	for(int i = 1 ; i < argc ; i++) {
		if(strcmp(argv[i], "--image") == 0) {
			if(i + 1 < argc)
				imageFile = argv[++i];
		} else if(strcmp(argv[i], "--thresh") == 0) {
			if(i + 1 < argc)
				modThreshold(atoi(argv[++i]));
		}
	}
	
	if(imageFile == NULL) {
		cerr << "Missing or incorrect image file. " << endl;
		exit(1);
	}

        // Load Intensity Image
	image<unsigned char>* intensityInput = loadPGM(imageFile);
		
	cpu.height = intensityInput->height();
	cpu.width  = intensityInput->width();
	
	cpu.gridXSize = 1 + (( cpu.width - 1) / TILE_SIZE);
	cpu.gridYSize = 1 + ((cpu.height - 1) / TILE_SIZE);
	
	int XSize = cpu.gridXSize*TILE_SIZE;
	int YSize = cpu.gridYSize*TILE_SIZE;
	
	cpu.size = XSize*YSize;

	cpu.intensity = new unsigned char[cpu.size];
		
	for (unsigned int y = 0 ; y < YSize ; y++){
		for (unsigned int x = 0 ; x < XSize ; x++){
				unsigned int newLocation = imageLocation (x, y, cpu.gridXSize);
			if (x < cpu.width && y < cpu.height) {
				cpu.intensity[newLocation] = intensityInput->data[y*cpu.width + x];
			} else{
				// Necessary in case image size is not a multiple of TILE_SIZE
				cpu.intensity[newLocation] = 0;
			}
		}
	}

	#if defined(VERBOSE)
		cout << "Finished Processing files with image of size " << cpu.height
		     << "x" << cpu.width << endl;
		cout << "Running warmup" << endl;
	#endif

	#if defined(WARMUP)
		cpu.result = sobelWarmup(cpu.intensity, 
			   cpu.height, 
			   cpu.width);
	#endif
	
	#if defined(VERBOSE)
		cout << "Running Algorithm" << endl;
	#endif
	
	cpu.result = sobel(cpu.intensity, 
			   cpu.height, 
			   cpu.width);
	
	// Output RGB images		
	char filename[64];
	sprintf(filename, "result.ppm");

	srand(1000);
	
	#if defined(VERBOSE)
		cout << "Produccing final image file " << filename << "." << endl;
	#endif
	
	Color color;
	
	// Create output image
	image<Color> output = image<Color>(cpu.width, cpu.height, true);
	image<Color>* im = &output;
	
	Color randomcolor = randomColor();
	for (unsigned int y = 0 ; y < cpu.height ; y++){
		for (unsigned int x = 0 ; x < cpu.width ; x++){
			unsigned int newLocation = imageLocation (x, y, cpu.gridXSize);
			
			color.r = cpu.result[newLocation];
			color.g = cpu.result[newLocation];
			color.b = cpu.result[newLocation];
			im->access[y][x] = color;
		}
	}
	savePPM(im, filename);
	
	// Free resources and end the program
	free(cpu.intensity);

        return 0;
}
