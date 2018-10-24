/* Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 * 
 * Image Handler Header 
 *  
 */
 
#ifndef IMGHANDLER_H
#define IMGHANDLER_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <climits>

using namespace std;

#include "image.h"

#define BUF_SIZE 256

#define RAND_MAX_VAL 150

struct Color {
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

// Input File function
void readPNM(ifstream &file, char* buf);

// Image file functions
void savePPM(image<Color>* im, const char* name);
image<unsigned char>* loadPGM(const char* name);

Color randomColor();

#endif