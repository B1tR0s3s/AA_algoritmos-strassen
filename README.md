# Matrix Multiplication with Strassen and CUDA

This repository contains the experimental development of a project aimed at evaluating a hybrid approach for dense matrix multiplication using the Strassen algorithm, CUDA, and GEMM. The main goal is to analyze the practical performance of a strategy that combines algorithmic reduction and GPU acceleration.

> Note: A measurement script was developed for each repository to run the experiments and record the performance metrics defined for each implementation. Obtaining results requires running those scripts. For repositories 1, 3, and 4, the tests use square matrices of size $n \times n$ with values 64, 128, 256, 512, and 1024; for the fourth repository, $n \times k$ matrices are used.

## Reviewed Repositories

| No. | Repository | Focus |
|---|---|---|
| 1 | [F1000 - MCM + Strassen](https://github.com/thulasi-bikku/F1000/blob/main/matrix_chain_multiplication_using_Strassen%E2%80%99s_algorithm_.ipynb) | Python notebook with Matrix Chain Multiplication and Strassen |
| 2 | [TSM2X](https://github.com/codyjrivera/tsm2x-imp) | Tall-and-skinny matrix multiplication in CUDA |
| 3 | [CUTLASS Stream-K](https://github.com/NVIDIA/cutlass/blob/main/examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk.cu) | GEMM optimized with CUTLASS Stream-K |
| 4 | [Davis GEMM](https://github.com/davisethan/gemm) | Comparison of GEMM with CUDA, cuBLAS, BLAS, OpenMP, and C++ Threads |



## General Requirements
- Python 3
- CUDA Toolkit
- Compilador nvcc
- NVIDIA GPU compatible 7.0+ (RTX SM 8.0+ para ampere_gemm_universal_streamk.cu)
- psutil
- jupyter
- nbclient
- nbformat
- numpy
- matplotlib


## Install Python Dependencies

```
pip install psutil jupyter nbclient nbformat numpy matplotlib pandas
```
### 1. F1000 - MCM + Strassen Notebook

See the [F1000 README][f1000-readme].

### 2. TSM2X

See the [TSM2X README][tsm2x-readme].

### 3. Cutlass

See the [CUTLASS README][cutlass-readme].

### 4. Davis GEMM

See the [GEMM README][gemm-readme].

### 5. Own Implementation Proposal

See the [proposal README][proposal-readme].


[f1000-readme]: https://github.com/B1tR0s3s/AA_algoritmos-strassen/blob/main/F1000/README.md
[tsm2x-readme]: https://github.com/B1tR0s3s/AA_algoritmos-strassen/blob/main/tsm2x-imp/README.md
[cutlass-readme]: https://github.com/B1tR0s3s/AA_algoritmos-strassen/blob/main/cutlass/README.md
[gemm-readme]: https://github.com/B1tR0s3s/AA_algoritmos-strassen/blob/main/gemm/README.md
[proposal-readme]: https://github.com/B1tR0s3s/AA_algoritmos-strassen/blob/main/propuesta_strassen/README.md
