/* Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 * 
 * Image Handler Function Implementation 
 *  
 */
 
#include "imghandler.h"

image<unsigned char>* loadPGM(const char* name) {
	char buf[BUF_SIZE];

	// Read header
	ifstream file(name, ios::in | ios::binary);
	readPNM(file, buf);
	if(strncmp(buf, "P5", 2)) {
		cerr << "Unable to open '" << name << "'." << endl;
	}

	readPNM(file, buf);
	int width = atoi(buf);
	readPNM(file, buf);
	int height = atoi(buf);

	readPNM(file, buf);
	if(atoi(buf) > UCHAR_MAX) {
		cerr << "Unable to open '" << name << "'." << endl;
	}

	// Read data
	image<unsigned char>* im = new image<unsigned char>(width, height);
	file.read((char*)imPtr(im, 0, 0), width*height*sizeof(unsigned char));

	return im;
}


void readPNM(ifstream &file, char* buf) {
	char doc[BUF_SIZE];
	char c;

	file >> c;
	while (c == '#') {
		file.getline(doc, BUF_SIZE);
		file >> c;
	}
	file.putback(c);

	file.width(BUF_SIZE);
	file >> buf;
	file.ignore();
}


void savePPM(image<Color>* im, const char* name) {
	int width = im->width();
	int height = im->height();
	ofstream file(name, ios::out | ios::binary);

	file << "P6\n" << width << " " << height << "\n" << UCHAR_MAX << "\n";
	file.write((char*)imPtr(im, 0, 0), width*height*sizeof(Color));
}


Color randomColor() {
	Color c;
	c.r = (unsigned char) rand()%RAND_MAX_VAL;
	c.g = (unsigned char) rand()%RAND_MAX_VAL;
	c.b = (unsigned char) rand()%RAND_MAX_VAL;

	return c;
}