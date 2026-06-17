# Implementacion de la propuesta

Este proyecto contiene dos implementaciones de multiplicación de matrices con Strassen y un generador de matrices de entrada.

## Estructura

- `generar/generar_matrices.cpp`: crea `A.txt` y `B.txt` con valores aleatorios.
- `strassenCPU/strassen.cpp`: versión secuencial en C++.
- `strassen_cuda.cu`: versión en CUDA con recursión y umbral híbrido.

## Requisitos básicos

- Sistema Linux o compatible con herramientas de compilación de C/C++.
- `g++` para compilar la versión CPU y el generador.
- `nvcc` y CUDA Toolkit para compilar la versión CUDA.
- Una GPU compatible con CUDA para ejecutar `strassen_cuda.cu`.
- Las matrices de entrada deben ser cuadradas, del mismo tamaño y estar guardadas como `A.txt` y `B.txt` en la carpeta `propuesta_strassen`.
- Para que Strassen funcione correctamente, el tamaño debe ser potencia de 2: 4, 8, 16, 32, 64, etc.

## Dependencias necesarias

### CPU y generador

- Biblioteca estándar de C++.
- No usa dependencias externas.

### CUDA

- CUDA Runtime.
- CUDA Toolkit instalado y configurado en el entorno.

## Comandos de ejecución

Todos los comandos asumen que estás parado dentro de `propuesta_strassen`.

### 1. Generar matrices de entrada

Compila:

```bash
g++ -O2 -std=c++17 -o generar_matrices generar/generar_matrices.cpp
```

Ejecuta:

```bash
./generar_matrices
```

Esto crea `A.txt` y `B.txt` en la carpeta actual.

### 2. Ejecutar la versión CPU

Compila:

```bash
g++ -O2 -std=c++17 -o strassen_cpu strassenCPU/strassen.cpp
```

Ejecuta:

```bash
./strassen_cpu
```

Lee `A.txt` y `B.txt`, multiplica ambas matrices y guarda el resultado en `C.txt`.

### 3. Ejecutar la versión CUDA

Compila:

```bash
nvcc -O2 -o strassen_cuda strassen_cuda.cu
```

Ejecuta:

```bash
./strassen_cuda
```

Lee `A.txt` y `B.txt`, ejecuta la multiplicación en GPU y guarda el resultado en `C.txt`.

## Lógica de cada función

### `generar/generar_matrices.cpp`

#### `guardar(const char* archivo, float* M, int n)`

Escribe una matriz en un archivo de texto. Primero guarda el tamaño `n` y luego imprime todos los valores fila por fila.

#### `main()`

Inicializa la semilla aleatoria, reserva memoria para dos matrices `N x N`, llena ambas con enteros aleatorios entre 1 y 9, y luego las guarda en `A.txt` y `B.txt`. Al final libera memoria y muestra un mensaje de confirmación.

### `strassenCPU/strassen.cpp`

#### `crearMatriz(int n)`

Crea una matriz `n x n` inicializada en cero. Se usa como base para construir submatrices y resultados intermedios.

#### `sumar(const Matrix& A, const Matrix& B)`

Realiza suma elemento a elemento entre dos matrices del mismo tamaño y devuelve una nueva matriz con el resultado.

#### `restar(const Matrix& A, const Matrix& B)`

Realiza resta elemento a elemento entre dos matrices del mismo tamaño y devuelve una nueva matriz con el resultado.

#### `strassen(const Matrix& A, const Matrix& B)`

Implementa la multiplicación de Strassen de forma recursiva.

- Si la matriz es de tamaño `1 x 1`, multiplica directamente los valores.
- Si no, divide ambas matrices en cuatro cuadrantes.
- Calcula los 7 productos intermedios de Strassen (`M1` a `M7`).
- Combina esos productos para construir los 4 cuadrantes del resultado (`C11`, `C12`, `C21`, `C22`).
- Une los cuadrantes en una sola matriz y la retorna.

#### `leerMatriz(const string& nombreArchivo)`

Abre un archivo de texto, ignora líneas vacías o comentarios que empiecen con `#`, lee el tamaño `n` y luego carga los `n x n` valores de la matriz.

#### `guardarMatriz(const Matrix& M, const string& nombreArchivo)`

Guarda una matriz en un archivo de texto. Primero escribe `n` y después cada fila con sus valores separados por espacios.

#### `main()`

Lee `A.txt` y `B.txt`, valida que ambas matrices tengan el mismo tamaño, mide el tiempo de ejecución de `strassen`, guarda el resultado en `C.txt` y muestra el tiempo total en milisegundos.

### `strassen_cuda.cu`

#### `gemm_kernel(...)`

Kernel clásico de multiplicación de matrices con memoria compartida. Se usa como caso base cuando el tamaño es pequeño. Calcula `C = alpha * A * B + beta * C`.

#### `mat_add_kernel(...)`

Kernel de suma/resta elemento a elemento. Calcula `R = alpha * A + beta * B` para acelerar las operaciones auxiliares de Strassen.

#### `extract_submatrix(...)`

Extrae un cuadrante de una matriz grande en GPU y lo copia a una submatriz destino.

#### `insert_submatrix(...)`

Inserta una submatriz en una posición concreta de la matriz destino.

#### `gpu_gemm(...)`

Función de host que configura bloques y grillas para lanzar `gemm_kernel`.

#### `gpu_add(...)`

Función de host que lanza `mat_add_kernel` para sumar o restar matrices en GPU.

#### `strassen_gpu(...)`

Implementa Strassen recursivo en GPU.

- Si `n` es menor o igual al umbral `STRASSEN_THRESHOLD`, usa GEMM clásico.
- Si no, divide `A` y `B` en cuatro cuadrantes.
- Extrae los cuadrantes a memoria temporal de GPU.
- Calcula los 7 productos de Strassen con llamadas recursivas.
- Combina los resultados para formar los cuadrantes de `C`.
- Inserta los cuadrantes finales en la matriz de salida.
- Libera toda la memoria temporal reservada.

#### `leer_matriz(...)`

Abre `A.txt` o `B.txt`, lee el tamaño de la matriz, reserva memoria en host y carga todos los valores. Si ocurre un error, devuelve `-1`.

#### `guardar_matriz(...)`

Guarda una matriz en archivo con una cabecera legible, el tamaño `n` y los valores formateados con cuatro decimales.

#### `imprimir_matriz(...)`

Imprime una matriz en consola. Se usa solo cuando el tamaño es pequeño para facilitar la inspección visual.

#### `main()`

Lee `A.txt` y `B.txt`, valida tamaños, copia datos a la GPU, ejecuta `strassen_gpu`, mide el tiempo con eventos CUDA, copia el resultado de vuelta al host y lo guarda en `C.txt`. Al final libera memoria y destruye los eventos.

## Flujo recomendado de uso

1. Compilar `generar_matrices`.
2. Ejecutar `./generar_matrices` para crear `A.txt` y `B.txt`.
3. Compilar `strassen_cpu` o `strassen_cuda`.
4. Ejecutar la versión que quieras probar.
5. Revisar `C.txt` para ver el resultado.

## Notas importantes

- El archivo `A.txt` y `B.txt` deben existir antes de ejecutar cualquiera de las dos versiones de Strassen.
- La versión CPU usa enteros en memoria lógica, mientras que la versión CUDA trabaja con `float`.
- En `generar_matrices.cpp`, el tamaño `N` se define como una constante y debe mantenerse como potencia de 2 para evitar problemas con Strassen.
- En `strassen_cuda.cu`, el programa asume que la GPU soporta CUDA y que el tamaño de la matriz es potencia de 2.