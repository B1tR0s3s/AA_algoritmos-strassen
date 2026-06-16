#ifndef SIZE
#define SIZE 256
#endif

#ifndef UTILITY_H
#define UTILITY_H

#define BLOCK 32
#define LOW 2.0
#define HIGH 5.0

#include <queue>
#include <mutex>

struct Task
{
    int ii;
    int jj;
};

void printMatrix(double *A);
void randomMatrix(double *A);
void transposeMatrix(double *A);
double frobeniusNorm(double *A, double *B);
void gemmOmp(double *A, double *B, double *C);
void gemmBlas(double *A, double *B, double *C);
void gemmCppThreads(double *A, double *B, double *C);
void gemmCppThreadsWorker(std::queue<Task> &queue, std::mutex &mutex, double *A, double *B, double *C);

#endif
