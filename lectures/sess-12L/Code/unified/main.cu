#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

void initialize (int N, float *a, float *b, float *c){
	for (int i = 0; i < N; i++){
		if (i < N){
			c[i] = 0;
			a[i] = 1 + i;
			b[i] = 1 - i;
		}
	}
}

void addVectors (int N, float *a, float *b, float *c){
	for (int i = 0; i < N; i++){
		if (i < N){
			c[i] = a[i] + b[i];
		}
	}
}

int main (int argc, char **argv){
	
	if (argc != 2) exit (1);
	int N = atoi(argv[1]);

	float *a, *b, *c;
	a = (float *) malloc(N*sizeof(float));
	b = (float *) malloc(N*sizeof(float));
	c = (float *) malloc(N*sizeof(float));

	initialize(N,a,b,c);
	
	addVectors(N,a,b,c);

	for (int i = 0; i < 5; i++) {
		printf("%f\n", c[i]);
	}

	free(a);
	free(b);
	free(c);
}
