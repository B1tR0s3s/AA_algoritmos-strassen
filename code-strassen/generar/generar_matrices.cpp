// =============================================================
//  generar_matrices.cpp
//  Genera A.txt y B.txt con valores enteros aleatorios (1..9)
//
//  Compilar: g++ -O2 -o generar_matrices generar_matrices.cpp
//  Usar    : ./generar_matrices
// =============================================================

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//  g++ -O2 -o generar_matrices generar_matrices.cpp
// --- CAMBIA AQUÍ el tamaño (debe ser potencia de 2: 4, 8, 16, 32...) ---
static const int N = 4096;
// ------------------------------------------------------------------------

void guardar(const char* archivo, float* M, int n)
{
    FILE* f = fopen(archivo, "w");
    fprintf(f, "%d\n", n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            fprintf(f, "%6.0f", M[i*n+j]);
        fprintf(f, "\n");
    }
    fclose(f);
}

int main()
{
    srand((unsigned int)time(NULL));

    float* A = (float*)malloc(N*N*sizeof(float));
    float* B = (float*)malloc(N*N*sizeof(float));

    for (int i = 0; i < N*N; i++) {
        A[i] = (float)(rand() % 9 + 1);
        B[i] = (float)(rand() % 9 + 1);
    }

    guardar("A.txt", A, N);
    guardar("B.txt", B, N);

    printf("A.txt y B.txt generados (%dx%d).\n", N, N);

    free(A);
    free(B);
    return 0;
}
