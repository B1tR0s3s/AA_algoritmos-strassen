#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda.h>
#include <cublas_v2.h>
#include "utilityCuda.h"

void gemmCuBlasHelper(double *A, double *B, double *C, double *dA, double *dB, double *dC);
void gemmCudaHelper(double *A, double *B, double *C, double *dA, double *dB, double *dC);

int main(int argc, char *argv[])
{
    // Create pointers
    double *A, *B, *C, *D;
    double *dA, *dB, *dC, *dD;

    // Allocate memory
    A = new double[SIZE * SIZE]();
    B = new double[SIZE * SIZE]();
    C = new double[SIZE * SIZE]();
    D = new double[SIZE * SIZE]();
    cudaMalloc((void **)&dA, SIZE * SIZE * sizeof(double));
    cudaMalloc((void **)&dB, SIZE * SIZE * sizeof(double));
    cudaMalloc((void **)&dC, SIZE * SIZE * sizeof(double));
    cudaMalloc((void **)&dD, SIZE * SIZE * sizeof(double));

    // Initialize memory
    randomizeMatrix(A), randomizeMatrix(B);

    // Timed matrix multiplication
    std::cout << "Matrix size = " << SIZE << ", Block size = " << BLOCK << std::endl;
    gemmCuBlasHelper(A, B, C, dA, dB, dC);
    gemmCudaHelper(A, B, D, dA, dB, dD);

    // Validate product matrix
    transposeMatrix(C);
    double norm = frobeniusNorm(C, D);
    std::cout << "CuBLAS and CUDA residual (Frobenius norm) = " << norm << std::endl;

    // Release memory
    delete[] A, delete[] B, delete[] C, delete[] D;
    cudaFree(dA), cudaFree(dB), cudaFree(dC), cudaFree(dD);

    return EXIT_SUCCESS;
}

/**
 * @brief Timed CuBLAS matrix multiplication
 * @param A Host left matrix
 * @param B Host right matrix
 * @param C Host product matrix
 * @param dA Device left matrix
 * @param dB Device right matrix
 * @param dC Device product matrix
 */
void gemmCuBlasHelper(double *A, double *B, double *C, double *dA, double *dB, double *dC)
{
    // Initialize memory
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> durationExecution;
    std::chrono::duration<double, std::milli> durationMemcpyHostToDevice;
    std::chrono::duration<double, std::milli> durationMemcpyDeviceToHost;

    // Timed copy from host to device
    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(dA, A, SIZE * SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, SIZE * SIZE * sizeof(double), cudaMemcpyHostToDevice);
    end = std::chrono::high_resolution_clock::now();
    durationMemcpyHostToDevice = std::chrono::duration<double, std::milli>(end - start);

    // Timed CuBLAS matrix multiplication
    start = std::chrono::high_resolution_clock::now();
    gemmCuBlas(dA, dB, dC);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    durationExecution = std::chrono::duration<double, std::milli>(end - start);
    std::cout << "CuBLAS execution elapsed time (ms) = " << durationExecution.count() << std::endl;

    // Timed copy from device to host
    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(C, dC, SIZE * SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    end = std::chrono::high_resolution_clock::now();
    durationMemcpyDeviceToHost = std::chrono::duration<double, std::milli>(end - start);
    std::cout << "CuBLAS copy elapsed time (ms) = " << durationMemcpyHostToDevice.count() + durationMemcpyDeviceToHost.count() << std::endl;
}

/**
 * @brief Timed CUDA matrix multiplication
 * @param A Host left matrix
 * @param B Host right matrix
 * @param C Host product matrix
 * @param dA Device left matrix
 * @param dB Device right matrix
 * @param dC Device product matrix
 */
void gemmCudaHelper(double *A, double *B, double *C, double *dA, double *dB, double *dC)
{
    // Initialize memory
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> durationExecution;
    std::chrono::duration<double, std::milli> durationMemcpyHostToDevice;
    std::chrono::duration<double, std::milli> durationMemcpyDeviceToHost;

    // Timed copy from host to device
    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(dA, A, SIZE * SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, SIZE * SIZE * sizeof(double), cudaMemcpyHostToDevice);
    end = std::chrono::high_resolution_clock::now();
    durationMemcpyHostToDevice = std::chrono::duration<double, std::milli>(end - start);

    // Timed CUDA matrix multiplication
    double ceil = std::ceil((double)SIZE / BLOCK);
    dim3 grid(ceil, ceil);
    dim3 block(BLOCK, BLOCK);
    int shared = 2 * BLOCK * BLOCK * sizeof(double);
    start = std::chrono::high_resolution_clock::now();
    gemmCuda<<<grid, block, shared>>>(dA, dB, dC);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    durationExecution = std::chrono::duration<double, std::milli>(end - start);
    std::cout << "CUDA execution elapsed time (ms) = " << durationExecution.count() << std::endl;

    // Timed copy from device to host
    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(C, dC, SIZE * SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    end = std::chrono::high_resolution_clock::now();
    durationMemcpyHostToDevice = std::chrono::duration<double, std::milli>(end - start);
    std::cout << "CUDA copy elapsed time (ms) = " << durationMemcpyHostToDevice.count() + durationMemcpyDeviceToHost.count() << std::endl;
}
