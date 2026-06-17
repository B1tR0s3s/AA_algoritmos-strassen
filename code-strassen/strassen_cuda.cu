// =============================================================
//  strassen_cuda.cu — Strassen en CUDA con E/S por archivos
//
//  Lee   : A.txt, B.txt   (generados por generar_matrices)
//  Escribe: C.txt          (resultado de A × B)
//
//  Compilar: nvcc -O2 -o strassen_cuda strassen_cuda.cu
//  Usar    : ./strassen_cuda
//
//  Requiere: CUDA Toolkit, GPU compute capability >= 6.0
// =============================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// -------------------------------------------------------------
//  Parámetros globales
// -------------------------------------------------------------
#define BLOCK_SIZE          16    // Tiles shared memory (16×16 = 256 threads)
#define STRASSEN_THRESHOLD  64    // Umbral: debajo de este tamaño usa GEMM clásico

// =============================================================
//  KERNELS CUDA
// =============================================================

// KERNEL 1: GEMM clásico con shared memory — C = alpha·A·B + beta·C
__global__ void gemm_kernel(const float* A, const float* B, float* C,
                             int n, float alpha, float beta)
{
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        int aCol = t * BLOCK_SIZE + threadIdx.x;
        int bRow = t * BLOCK_SIZE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? A[row * n + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = alpha * acc + beta * C[row * n + col];
}

// KERNEL 2: Suma/resta elemento a elemento — R = alpha·A + beta·B
__global__ void mat_add_kernel(const float* A, const float* B, float* R,
                                int n, float alpha, float beta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n)
        R[idx] = alpha * A[idx] + beta * B[idx];
}

// KERNEL 3: Extrae un cuadrante (submatriz) de una matriz en GPU
__global__ void extract_submatrix(const float* src, float* dst,
                                   int n, int half,
                                   int row_off, int col_off)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < half && c < half)
        dst[r * half + c] = src[(r + row_off) * n + (c + col_off)];
}

// KERNEL 4: Inserta un cuadrante en una matriz destino en GPU
__global__ void insert_submatrix(const float* src, float* dst,
                                  int n, int half,
                                  int row_off, int col_off)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < half && c < half)
        dst[(r + row_off) * n + (c + col_off)] = src[r * half + c];
}

// =============================================================
//  HELPERS EN HOST
// =============================================================

#define ALLOC(ptr, n) cudaMalloc(&ptr, (size_t)(n)*(n)*sizeof(float))
#define FREE(ptr)     cudaFree(ptr)

void gpu_gemm(const float* dA, const float* dB, float* dC, int n)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    gemm_kernel<<<grid, block>>>(dA, dB, dC, n, 1.0f, 0.0f);
}

void gpu_add(const float* dA, const float* dB, float* dR,
             int n, float alpha, float beta)
{
    int threads = 256;
    int blocks  = (n * n + threads - 1) / threads;
    mat_add_kernel<<<blocks, threads>>>(dA, dB, dR, n, alpha, beta);
}

// =============================================================
//  STRASSEN RECURSIVO EN GPU
//  C = A × B  (matrices n×n, n = potencia de 2)
// =============================================================
void strassen_gpu(const float* dA, const float* dB, float* dC, int n)
{
    // Caso base: matrices pequeñas → GEMM clásico
    if (n <= STRASSEN_THRESHOLD) {
        gpu_gemm(dA, dB, dC, n);
        return;
    }

    int h = n / 2;

    // --- Paso 1: Extraer los 8 cuadrantes de A y B ---
    //  A = [A11 A12]    B = [B11 B12]
    //      [A21 A22]        [B21 B22]
    float *dA11, *dA12, *dA21, *dA22;
    float *dB11, *dB12, *dB21, *dB22;

    ALLOC(dA11, h); ALLOC(dA12, h); ALLOC(dA21, h); ALLOC(dA22, h);
    ALLOC(dB11, h); ALLOC(dB12, h); ALLOC(dB21, h); ALLOC(dB22, h);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((h + BLOCK_SIZE-1)/BLOCK_SIZE, (h + BLOCK_SIZE-1)/BLOCK_SIZE);

    extract_submatrix<<<grid,block>>>(dA, dA11, n, h, 0, 0);
    extract_submatrix<<<grid,block>>>(dA, dA12, n, h, 0, h);
    extract_submatrix<<<grid,block>>>(dA, dA21, n, h, h, 0);
    extract_submatrix<<<grid,block>>>(dA, dA22, n, h, h, h);

    extract_submatrix<<<grid,block>>>(dB, dB11, n, h, 0, 0);
    extract_submatrix<<<grid,block>>>(dB, dB12, n, h, 0, h);
    extract_submatrix<<<grid,block>>>(dB, dB21, n, h, h, 0);
    extract_submatrix<<<grid,block>>>(dB, dB22, n, h, h, h);

    // --- Paso 2: Los 7 productos de Strassen ---
    //  M1 = (A11 + A22)(B11 + B22)
    //  M2 = (A21 + A22) × B11
    //  M3 = A11 × (B12 - B22)
    //  M4 = A22 × (B21 - B11)
    //  M5 = (A11 + A12) × B22
    //  M6 = (A21 - A11)(B11 + B12)
    //  M7 = (A12 - A22)(B21 + B22)
    float *dM1, *dM2, *dM3, *dM4, *dM5, *dM6, *dM7;
    ALLOC(dM1, h); ALLOC(dM2, h); ALLOC(dM3, h); ALLOC(dM4, h);
    ALLOC(dM5, h); ALLOC(dM6, h); ALLOC(dM7, h);

    float *dT1, *dT2;
    ALLOC(dT1, h); ALLOC(dT2, h);

    gpu_add(dA11, dA22, dT1, h,  1.0f,  1.0f);
    gpu_add(dB11, dB22, dT2, h,  1.0f,  1.0f);
    strassen_gpu(dT1, dT2, dM1, h);

    gpu_add(dA21, dA22, dT1, h,  1.0f,  1.0f);
    strassen_gpu(dT1, dB11, dM2, h);

    gpu_add(dB12, dB22, dT1, h,  1.0f, -1.0f);
    strassen_gpu(dA11, dT1, dM3, h);

    gpu_add(dB21, dB11, dT1, h,  1.0f, -1.0f);
    strassen_gpu(dA22, dT1, dM4, h);

    gpu_add(dA11, dA12, dT1, h,  1.0f,  1.0f);
    strassen_gpu(dT1, dB22, dM5, h);

    gpu_add(dA21, dA11, dT1, h,  1.0f, -1.0f);
    gpu_add(dB11, dB12, dT2, h,  1.0f,  1.0f);
    strassen_gpu(dT1, dT2, dM6, h);

    gpu_add(dA12, dA22, dT1, h,  1.0f, -1.0f);
    gpu_add(dB21, dB22, dT2, h,  1.0f,  1.0f);
    strassen_gpu(dT1, dT2, dM7, h);

    // --- Paso 3: Combinar para obtener cuadrantes de C ---
    //  C11 = M1 + M4 - M5 + M7
    //  C12 = M3 + M5
    //  C21 = M2 + M4
    //  C22 = M1 - M2 + M3 + M6
    float *dC11, *dC12, *dC21, *dC22;
    ALLOC(dC11, h); ALLOC(dC12, h); ALLOC(dC21, h); ALLOC(dC22, h);

    gpu_add(dM1, dM4,  dT1,  h,  1.0f,  1.0f);
    gpu_add(dT1, dM5,  dT2,  h,  1.0f, -1.0f);
    gpu_add(dT2, dM7,  dC11, h,  1.0f,  1.0f);

    gpu_add(dM3, dM5,  dC12, h,  1.0f,  1.0f);
    gpu_add(dM2, dM4,  dC21, h,  1.0f,  1.0f);

    gpu_add(dM1, dM2,  dT1,  h,  1.0f, -1.0f);
    gpu_add(dT1, dM3,  dT2,  h,  1.0f,  1.0f);
    gpu_add(dT2, dM6,  dC22, h,  1.0f,  1.0f);

    // --- Paso 4: Ensamblar C desde los 4 cuadrantes ---
    insert_submatrix<<<grid,block>>>(dC11, dC, n, h, 0, 0);
    insert_submatrix<<<grid,block>>>(dC12, dC, n, h, 0, h);
    insert_submatrix<<<grid,block>>>(dC21, dC, n, h, h, 0);
    insert_submatrix<<<grid,block>>>(dC22, dC, n, h, h, h);

    // --- Liberar memoria temporal ---
    FREE(dA11); FREE(dA12); FREE(dA21); FREE(dA22);
    FREE(dB11); FREE(dB12); FREE(dB21); FREE(dB22);
    FREE(dM1);  FREE(dM2);  FREE(dM3);  FREE(dM4);
    FREE(dM5);  FREE(dM6);  FREE(dM7);
    FREE(dT1);  FREE(dT2);
    FREE(dC11); FREE(dC12); FREE(dC21); FREE(dC22);
}

// =============================================================
//  E/S — leer y guardar matrices en archivos .txt
// =============================================================

// Lee una matriz desde archivo. El archivo debe tener:
//   línea 1: n  (tamaño)
//   líneas siguientes: n filas con n floats
// Devuelve el tamaño leído, o -1 si hay error.
int leer_matriz(const char* archivo, float** out_M, int* out_n)
{
    FILE* f = fopen(archivo, "r");
    if (!f) {
        fprintf(stderr, "ERROR: no se puede abrir '%s'.\n"
                        "  Asegúrate de haber ejecutado ./generar_matrices primero.\n",
                archivo);
        return -1;
    }

    int n = 0;
    char linea[4096];

    // Saltar líneas de comentario (#) y leer el tamaño
    while (fgets(linea, sizeof(linea), f)) {
        if (linea[0] == '#') continue;
        if (sscanf(linea, "%d", &n) == 1) break;
    }

    if (n <= 0) {
        fprintf(stderr, "ERROR: tamaño inválido en '%s'.\n", archivo);
        fclose(f);
        return -1;
    }

    float* M = (float*)malloc(n * n * sizeof(float));
    if (!M) {
        fprintf(stderr, "ERROR: fallo de malloc para n=%d.\n", n);
        fclose(f);
        return -1;
    }

    for (int i = 0; i < n * n; i++) {
        if (fscanf(f, "%f", &M[i]) != 1) {
            fprintf(stderr, "ERROR: datos insuficientes en '%s' (elemento %d).\n",
                    archivo, i);
            free(M);
            fclose(f);
            return -1;
        }
    }

    fclose(f);
    *out_M = M;
    *out_n = n;
    return 0;
}

// Guarda una matriz en archivo con cabecera legible
void guardar_matriz(const char* archivo, float* M, int n, const char* etiqueta)
{
    FILE* f = fopen(archivo, "w");
    if (!f) {
        fprintf(stderr, "ERROR: no se puede escribir '%s'.\n", archivo);
        return;
    }

    fprintf(f, "# Matriz %s  [%d x %d]  — generada por strassen_cuda\n", etiqueta, n, n);
    fprintf(f, "%d\n", n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(f, "%10.4f", M[i * n + j]);
            if (j < n - 1) fprintf(f, " ");
        }
        fprintf(f, "\n");
    }

    fclose(f);
    printf("  -> Guardado: %s\n", archivo);
}

// Imprime en consola (solo matrices pequeñas)
void imprimir_matriz(const char* etiqueta, float* M, int n)
{
    printf("\n%s [%d x %d]:\n", etiqueta, n, n);
    for (int i = 0; i < n; i++) {
        printf("  ");
        for (int j = 0; j < n; j++)
            printf("%8.2f", M[i * n + j]);
        printf("\n");
    }
}

// =============================================================
//  MAIN
// =============================================================
int main()
{
    printf("=== Strassen CUDA — multiplicación de matrices ===\n\n");

    // ----------------------------------------------------------
    //  1. Leer A y B desde archivos
    // ----------------------------------------------------------
    float* hA = NULL;
    float* hB = NULL;
    int nA = 0, nB = 0;

    printf("Leyendo A.txt...\n");
    if (leer_matriz("A.txt", &hA, &nA) != 0) return 1;

    printf("Leyendo B.txt...\n");
    if (leer_matriz("B.txt", &hB, &nB) != 0) { free(hA); return 1; }

    if (nA != nB) {
        fprintf(stderr, "ERROR: tamaños incompatibles — A es %dx%d, B es %dx%d.\n",
                nA, nA, nB, nB);
        free(hA); free(hB);
        return 1;
    }

    int n = nA;
    size_t bytes = (size_t)n * n * sizeof(float);
    printf("Tamaño de matriz: %d x %d\n\n", n, n);

    // Mostrar en consola si la matriz es pequeña
    if (n <= 16) {
        imprimir_matriz("Matriz A", hA, n);
        imprimir_matriz("Matriz B", hB, n);
    }

    // ----------------------------------------------------------
    //  2. Copiar al device (GPU)
    // ----------------------------------------------------------
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, bytes);

    // ----------------------------------------------------------
    //  3. Ejecutar Strassen y medir tiempo
    // ----------------------------------------------------------
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);

    strassen_gpu(dA, dB, dC, n);
    cudaDeviceSynchronize();

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);

    printf("\nStrassen GPU completado.\n");
    printf("  Tiempo de ejecucion: %.3f ms\n", ms);

    // ----------------------------------------------------------
    //  4. Copiar resultado al host y guardar
    // ----------------------------------------------------------
    float* hC = (float*)malloc(bytes);
    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    printf("\nGuardando resultado...\n");
    guardar_matriz("C.txt", hC, n, "C = A × B  (Strassen GPU)");

    if (n <= 16)
        imprimir_matriz("Resultado C = A × B", hC, n);

    printf("\nAhora puedes ejecutar: ./verificar\n");

    // ----------------------------------------------------------
    //  5. Limpiar
    // ----------------------------------------------------------
    free(hA); free(hB); free(hC);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return 0;
}
