#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <thread>
#include <vector>
#include <omp.h>
#include <cblas.h>
#include "utility.h"

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
void randomMatrix(double *A)
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
 * @return Frobenius norm residual
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
 * @brief OpenMP matrix multiplication
 * @param A Left matrix
 * @param B Right matrix
 * @param C Product matrix
 */
void gemmOmp(double *A, double *B, double *C)
{
#pragma omp parallel for collapse(2)
    for (int ii = 0; ii < SIZE; ii += BLOCK)
    {
        for (int jj = 0; jj < SIZE; jj += BLOCK)
        {
            for (int kk = 0; kk < SIZE; kk += BLOCK)
            {
                int endi = std::min(ii + BLOCK, SIZE);
                int endj = std::min(jj + BLOCK, SIZE);
                int endk = std::min(kk + BLOCK, SIZE);
                for (int i = ii; i < endi; i++)
                {
                    for (int j = jj; j < endj; j++)
                    {
                        double sum = 0.0;
                        for (int k = kk; k < endk; k++)
                        {
                            sum += A[i * SIZE + k] * B[k * SIZE + j];
                        }
                        C[i * SIZE + j] += sum;
                    }
                }
            }
        }
    }
}

/**
 * @brief BLAS matrix multiplication
 * @param A Left matrix
 * @param B Right matrix
 * @param C Product matrix
 */
void gemmBlas(double *A, double *B, double *C)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, SIZE, SIZE, SIZE, 1.0, A, SIZE, B, SIZE, 1.0, C, SIZE);
}

/**
 * @brief C++ threads matrix multiplication
 * @param A Left matrix
 * @param B Right matrix
 * @param C Product matrix
 */
void gemmCppThreads(double *A, double *B, double *C)
{
    // Create task queue
    std::queue<Task> queue;
    for (int ii = 0; ii < SIZE; ii += BLOCK)
    {
        for (int jj = 0; jj < SIZE; jj += BLOCK)
        {
            queue.push({ii, jj});
        }
    }

    // Start worker threads
    int nthreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::mutex mutex;
    for (int i = 0; i < nthreads; i++)
    {
        threads.push_back(std::thread(gemmCppThreadsWorker, std::ref(queue), std::ref(mutex), A, B, C));
    }

    // Await worker threads
    for (int i = 0; i < nthreads; i++)
    {
        threads[i].join();
    }
}

/**
 * @brief C++ threads matrix multiplication worker
 * @param queue Task queue of matrix blocks
 * @param mutex Task queue mutex lock
 * @param A Left matrix
 * @param B Right matrix
 * @param C Product matrix
 */
void gemmCppThreadsWorker(std::queue<Task> &queue, std::mutex &mutex, double *A, double *B, double *C)
{
    while (true)
    {
        // Thread safe task queue operations
        mutex.lock();
        if (queue.empty())
        {
            mutex.unlock();
            return;
        }
        Task task = queue.front();
        queue.pop();
        mutex.unlock();

        // Blocked matrix multiplication
        for (int kk = 0; kk < SIZE; kk += BLOCK)
        {
            int endi = std::min(task.ii + BLOCK, SIZE);
            int endj = std::min(task.jj + BLOCK, SIZE);
            int endk = std::min(kk + BLOCK, SIZE);
            for (int i = task.ii; i < endi; i++)
            {
                for (int j = task.jj; j < endj; j++)
                {
                    double sum = 0.0;
                    for (int k = kk; k < endk; k++)
                    {
                        sum += A[i * SIZE + k] * B[k * SIZE + j];
                    }
                    C[i * SIZE + j] += sum;
                }
            }
        }
    }
}
