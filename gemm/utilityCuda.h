#ifndef SIZE
#define SIZE 256
#endif

#ifndef UTILITY_CUDA_H
#define UTILITY_CUDA_H

#define BLOCK 32
#define LOW 2.0
#define HIGH 5.0

void printMatrix(double *A);
void randomizeMatrix(double *A);
void transposeMatrix(double *A);
double frobeniusNorm(double *A, double *B);
__global__ void gemmCuda(double *dA, double *dB, double *dC);
void gemmCuBlas(double *dA, double *dB, double *dC);

#endif
