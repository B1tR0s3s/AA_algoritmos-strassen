#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <cuda.h>
#include <cublas_v2.h>
#include "utilityCuda.h"

/**
 * @brief Print matrix
 * @param A Matrix to print
 */
void printMatrix(double *A)
{
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            std::cout << A[i * SIZE + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/**
 * @brief Randomize matrix
 * @param A Matrix to randomize
 */
void randomizeMatrix(double *A)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(LOW, HIGH);
    for (int i = 0; i < SIZE * SIZE; i++)
    {
        A[i] = dist(gen);
    }
}

/**
 * @brief Tranpose matrix
 * @param A Matrix to transpose
 */
void transposeMatrix(double *A)
{
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < i; j++)
        {
            std::swap(A[i * SIZE + j], A[j * SIZE + i]);
        }
    }
}

/**
 * @brief Frobenius norm residual
 * @param A First matrix
 * @param B Second matrix
 */
double frobeniusNorm(double *A, double *B)
{
    double sum = 0.0;
    for (int i = 0; i < SIZE * SIZE; i++)
    {
        double diff = A[i] - B[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

/**
 * @brief CUDA matrix multiplication
 * @param dA Device left matrix
 * @param dB Device right matrix
 * @param dC Device product matrix
 */
__global__ void gemmCuda(double *dA, double *dB, double *dC)
{
    // Create shared memory
    extern __shared__ double shared[];
    double *sharedA = shared;
    double *sharedB = &shared[BLOCK * BLOCK];

    // Get row and column inside of grid
    int ii = blockIdx.y * blockDim.y + threadIdx.y;
    int jj = blockIdx.x * blockDim.x + threadIdx.x;

    // Execute dot product of many tiles
    double sum = 0.0;
    for (int kk = 0; kk < SIZE; kk += BLOCK)
    {
        // Get row and column inside of block
        int i = threadIdx.y, j = threadIdx.x;

        // Load shared memory
        sharedA[i * BLOCK + j] = ii < SIZE && kk + j < SIZE ? dA[ii * SIZE + (kk + j)] : 0.0;
        sharedB[i * BLOCK + j] = kk + i < SIZE && jj < SIZE ? dB[(kk + i) * SIZE + jj] : 0.0;

        __syncthreads();

        // Execute dot product of one tile
        for (int k = 0; k < BLOCK; k++)
        {
            sum += sharedA[i * BLOCK + k] * sharedB[k * BLOCK + j];
        }

        __syncthreads();
    }

    // Update product matrix
    if (ii < SIZE && jj < SIZE)
    {
        dC[ii * SIZE + jj] = sum;
    }
}

/**
 * @brief CuBLAS matrix multiplication
 * @param dA Device left matrix
 * @param dB Device right matrix
 * @param dC Device product matrix
 */
void gemmCuBlas(double *dA, double *dB, double *dC)
{
    double alpha = 1.0, beta = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, SIZE, SIZE, SIZE, &alpha, dA, SIZE, dB, SIZE, &beta, dC, SIZE);
    cublasDestroy(handle);
}
