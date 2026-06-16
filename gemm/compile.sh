#!/bin/bash

g++ -DSIZE=$1 utility.cpp main.cpp -o main \
    -fopenmp \
    -lopenblas

nvcc -DSIZE=$1 utilityCuda.cu mainCuda.cu -o mainCuda \
    -lcublas