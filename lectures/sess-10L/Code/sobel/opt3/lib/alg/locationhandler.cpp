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
	unsigned int location =  ( x&TTSMask )                               |
			        (( y&TTSMask )          <<  TTSB        )    |
			       ((( x>>TTSB ) &BTSMask ) << (TTSB+TTSB)  )    |
			       ((( y>>TTSB ) &BTSMask ) << (BTSB+TTSB+TTSB)) |
			        (( x>>TSB  )            << (TSB+TSB)    ) ;
	location += 	        (( y>>TSB  )            << (TSB+TSB)    )*gridXSize;
	return (location);
}
