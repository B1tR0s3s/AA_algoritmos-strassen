# Proposal Implementation

This project contains two Strassen matrix multiplication implementations and an input matrix generator.

## Structure

- `generar/generar_matrices.cpp`: creates `A.txt` and `B.txt` with random values.
- `strassenCPU/strassen.cpp`: sequential C++ version.
- `strassen_cuda.cu`: CUDA version with recursion and a hybrid threshold.

## Basic Requirements

- Linux or a compatible system with C/C++ build tools.
- `g++` to compile the CPU version and the generator.
- `nvcc` and CUDA Toolkit to compile the CUDA version.
- A CUDA-compatible GPU to run `strassen_cuda.cu`.
- Input matrices must be square, the same size, and saved as `A.txt` and `B.txt` in the `propuesta_strassen` folder.
- For Strassen to work correctly, the size must be a power of 2: 4, 8, 16, 32, 64, etc.

## Required Dependencies

### CPU and Generator

- C++ standard library.
- No external dependencies.

### CUDA

- CUDA Runtime.
- CUDA Toolkit installed and configured in the environment.

## Run Commands

All commands assume you are inside `propuesta_strassen`.

### 1. Generate Input Matrices

Build:

```bash
g++ -O2 -std=c++17 -o generar_matrices generar/generar_matrices.cpp
```

Run:

```bash
./generar_matrices
```

This creates `A.txt` and `B.txt` in the current folder.

### 2. Run the CPU Version

Compila:

```bash
g++ -O2 -std=c++17 -o strassen_cpu strassenCPU/strassen.cpp
```

Run:

```bash
./strassen_cpu
```

It reads `A.txt` and `B.txt`, multiplies both matrices, and saves the result to `C.txt`.

### 3. Run the CUDA Version

Compila:

```bash
nvcc -O2 -o strassen_cuda strassen_cuda.cu
```

Run:

```bash
./strassen_cuda
```

It reads `A.txt` and `B.txt`, runs the multiplication on the GPU, and saves the result to `C.txt`.

## Function Logic

### `generar/generar_matrices.cpp`

#### `guardar(const char* archivo, float* M, int n)`

Writes a matrix to a text file. It first saves the size `n` and then prints all values row by row.

#### `main()`

Initializes the random seed, allocates memory for two `N x N` matrices, fills both with random integers between 1 and 9, and then saves them to `A.txt` and `B.txt`. Finally, it frees memory and shows a confirmation message.

### `strassenCPU/strassen.cpp`

#### `crearMatriz(int n)`

Creates an `n x n` matrix initialized to zero. It is used as the base for building submatrices and intermediate results.

#### `sumar(const Matrix& A, const Matrix& B)`

Performs element-wise addition between two matrices of the same size and returns a new matrix with the result.

#### `restar(const Matrix& A, const Matrix& B)`

Performs element-wise subtraction between two matrices of the same size and returns a new matrix with the result.

#### `strassen(const Matrix& A, const Matrix& B)`

Implements Strassen multiplication recursively.

- If the matrix size is `1 x 1`, it multiplies the values directly.
- Otherwise, it divides both matrices into four quadrants.
- It computes Strassen's 7 intermediate products (`M1` to `M7`).
- It combines those products to build the 4 result quadrants (`C11`, `C12`, `C21`, `C22`).
- It merges the quadrants into a single matrix and returns it.

#### `leerMatriz(const string& nombreArchivo)`

Opens a text file, ignores empty lines or comments starting with `#`, reads the size `n`, and then loads the `n x n` matrix values.

#### `guardarMatriz(const Matrix& M, const string& nombreArchivo)`

Saves a matrix to a text file. It first writes `n` and then each row with values separated by spaces.

#### `main()`

Reads `A.txt` and `B.txt`, validates that both matrices have the same size, measures the execution time of `strassen`, saves the result to `C.txt`, and prints the total time in milliseconds.

### `strassen_cuda.cu`

#### `gemm_kernel(...)`

Classic matrix multiplication kernel with shared memory. It is used as the base case when the size is small. Computes `C = alpha * A * B + beta * C`.

#### `mat_add_kernel(...)`

Element-wise addition/subtraction kernel. Computes `R = alpha * A + beta * B` to speed up Strassen's auxiliary operations.

#### `extract_submatrix(...)`

Extracts a quadrant from a large matrix on the GPU and copies it to a destination submatrix.

#### `insert_submatrix(...)`

Inserts a submatrix into a specific position in the destination matrix.

#### `gpu_gemm(...)`

Host function that configures blocks and grids to launch `gemm_kernel`.

#### `gpu_add(...)`

Host function that launches `mat_add_kernel` to add or subtract matrices on the GPU.

#### `strassen_gpu(...)`

Implements recursive Strassen on the GPU.

- If `n` is less than or equal to `STRASSEN_THRESHOLD`, it uses classic GEMM.
- Otherwise, it divides `A` and `B` into four quadrants.
- It extracts the quadrants into temporary GPU memory.
- It computes the 7 Strassen products with recursive calls.
- It combines the results to form the quadrants of `C`.
- It inserts the final quadrants into the output matrix.
- It frees all allocated temporary memory.

#### `leer_matriz(...)`

Opens `A.txt` or `B.txt`, reads the matrix size, allocates host memory, and loads all values. If an error occurs, it returns `-1`.

#### `guardar_matriz(...)`

Saves a matrix to a file with a readable header, the size `n`, and values formatted with four decimals.

#### `imprimir_matriz(...)`

Prints a matrix to the console. It is used only when the size is small to make visual inspection easier.

#### `main()`

Reads `A.txt` and `B.txt`, validates sizes, copies data to the GPU, runs `strassen_gpu`, measures the time with CUDA events, copies the result back to the host, and saves it to `C.txt`. Finally, it frees memory and destroys the events.

## Recommended Workflow

1. Build `generar_matrices`.
2. Run `./generar_matrices` to create `A.txt` and `B.txt`.
3. Build `strassen_cpu` or `strassen_cuda`.
4. Run the version you want to test.
5. Check `C.txt` to see the result.

## Important Notes

- `A.txt` and `B.txt` must exist before running either Strassen version.
- The CPU version uses integers in logical memory, while the CUDA version works with `float`.
- In `generar_matrices.cpp`, the size `N` is defined as a constant and must remain a power of 2 to avoid issues with Strassen.
- In `strassen_cuda.cu`, the program assumes the GPU supports CUDA and that the matrix size is a power of 2.
