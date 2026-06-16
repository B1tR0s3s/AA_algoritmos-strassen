#include <iostream>
#include <cstdlib>
#include <chrono>
#include "utility.h"

void gemmBlasHelper(double *A, double *B, double *C);
void gemmOmpHelper(double *A, double *B, double *C);
void gemmCppThreadsHelper(double *A, double *B, double *C);

int main(int argc, char *argv[])
{
    // Allocate memory
    double *A = new double[SIZE * SIZE]();
    double *B = new double[SIZE * SIZE]();
    double *C = new double[SIZE * SIZE]();
    double *D = new double[SIZE * SIZE]();
    double *E = new double[SIZE * SIZE]();

    // Initialize memory
    randomMatrix(A), randomMatrix(B);

    // Timed matrix multiplication
    std::cout << "Matrix size = " << SIZE << ", Block size = " << BLOCK << std::endl;
    gemmBlasHelper(A, B, C);
    gemmOmpHelper(A, B, D);
    gemmCppThreadsHelper(A, B, E);

    // Validate product matrix
    double norm1 = frobeniusNorm(C, D);
    double norm2 = frobeniusNorm(C, E);
    std::cout << "BLAS and OpenMP residual (Frobenius norm) = " << norm1 << std::endl;
    std::cout << "BLAS and C++ threads residual (Frobenius norm) = " << norm2 << std::endl;

    // Release memory
    delete[] A, delete[] B;
    delete[] C, delete[] D, delete[] E;

    return EXIT_SUCCESS;
}

/**
 * @brief Timed BLAS matrix multiplication
 * @param A Left matrix
 * @param B Right matrix
 * @param C Product matrix
 */
void gemmBlasHelper(double *A, double *B, double *C)
{
    // Initialize memory
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration;

    // Timed BLAS matrix multiplication
    start = std::chrono::high_resolution_clock::now();
    gemmBlas(A, B, C);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end - start);
    std::cout << "BLAS elapsed time (ms) = " << duration.count() << std::endl;
}

/**
 * @brief Timed OpenMP matrix multiplication
 * @param A Left matrix
 * @param B Right matrix
 * @param C Product matrix
 */
void gemmOmpHelper(double *A, double *B, double *C)
{
    // Initialize memory
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration;

    // Timed OpenMP matrix multiplication
    start = std::chrono::high_resolution_clock::now();
    gemmOmp(A, B, C);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end - start);
    std::cout << "OpenMP elapsed time (ms) = " << duration.count() << std::endl;
}

/**
 * @brief Timed C++ threads matrix multiplication
 * @param A Left matrix
 * @param B Right matrix
 * @param C Product matrix
 */
void gemmCppThreadsHelper(double *A, double *B, double *C)
{
    // Initialize memory
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration;

    // Timed C++ threads matrix multiplication
    start = std::chrono::high_resolution_clock::now();
    gemmCppThreads(A, B, C);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end - start);
    std::cout << "C++ threads elapsed time (ms) = " << duration.count() << std::endl;
}
