# Multiplicación de matrices mediante Strassen y CUDA

Este repositorio contiene el desarrollo experimental del proyecto orientado a evaluar una propuesta híbrida para la multiplicación de matrices densas mediante el algoritmo de Strassen, CUDA y GEMM. El objetivo principal es analizar el rendimiento práctico de una estrategia que combine reducción algorítmica y aceleración en GPU.

> Nota: Se desarrolló un script de medición para cada repositorio, encargado de ejecutar los experimentos y registrar las métricas de rendimiento definidas para cada implementación. La obtención de resultados requiere la ejecución de dichos scripts. Para los repositorios 1, 3 y 4, las pruebas se realizan utilizando matrices cuadradas de dimensión $n \times n$, tomando los valores de 64, 128, 256, 512 y 1024; para el cuarto repositorio se uso matrices de $n \times k$. 

## Repositorios revisados

| N.º | Repositorio | Enfoque |
|---|---|---|
| 1 | [F1000 - MCM + Strassen](https://github.com/thulasi-bikku/F1000/blob/main/matrix_chain_multiplication_using_Strassen%E2%80%99s_algorithm_.ipynb) | Notebook en Python con Matrix Chain Multiplication y Strassen |
| 2 | [TSM2X](https://github.com/codyjrivera/tsm2x-imp) | Multiplicación de matrices tall-and-skinny en CUDA |
| 3 | [CUTLASS Stream-K](https://github.com/NVIDIA/cutlass/blob/main/examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk.cu) | GEMM optimizado mediante CUTLASS Stream-K |
| 4 | [Davis GEMM](https://github.com/davisethan/gemm) | Comparación de GEMM con CUDA, cuBLAS, BLAS, OpenMP y C++ Threads |



## Requisitos generales
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


## Instalación de dependencias de Python:

```
pip install psutil jupyter nbclient nbformat numpy matplotlib pandas
```
### 1. F1000 - Notebook MCM + Strassen

> Ver README de F1000 en carpeta F1000

### 2. TSM2X

> Ver README de F1000 en carpeta F1000
<!-- Aqui va lo del tercer repo -->

### 3. Cutlass
> Ver README de cutlass

### 4. Davis GEMM

> Ver README de GEMM

### 5. Propuesta de implementacion propia
> Ver README de la propuesta