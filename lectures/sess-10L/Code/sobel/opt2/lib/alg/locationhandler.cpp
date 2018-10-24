/* Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 * 
 * Location Handler Implementation 
 *  
 */

#include "locationhandler.h"

using namespace std;

unsigned int imageLocation (unsigned int x, 
		unsigned int y, 
		unsigned int gridXSize){
	unsigned int location = y*(gridXSize*TILE_SIZE) + x;
	
	return (location);
}
