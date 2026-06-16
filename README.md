# Multiplicación de matrices mediante Strassen y CUDA

Este repositorio contiene el desarrollo experimental del proyecto orientado a evaluar una propuesta híbrida para la multiplicación de matrices densas mediante el algoritmo de Strassen, CUDA y GEMM. El objetivo principal es analizar el rendimiento práctico de una estrategia que combine reducción algorítmica y aceleración en GPU.


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
pip install psutil jupyter nbclient nbformat numpy matplotlib pandas

## Ejecución de mediciones
1. F1000 - Notebook MCM + Strassen


```python
python scripts/medir_f1000_ipynb.py f1000_strassen.ipynb --reps 3
```

Métricas obtenidas:

- Tiempo total de ejecución
- Memoria RAM pico
- Memoria RAM promedio
- Tiempo por celda

2. TSM2X

Compilación:

```
make
```

Ejecución de medición:

```python
python scripts/medir_tsm2x.py --mode right --sizes 1024 2048 --cols 2 4 --reps 3
```
Métricas obtenidas:

- Tiempo promedio
- RAM pico
- VRAM pico
- Uso promedio de GPU

4. Davis GEMM

Compilación:

```
./compile.sh [tamano_matriz]
```

Ejecución:

```
./mainCuda
```

Medición:

```python
python scripts/medir_davis_gemm.py --programs mainCuda --sizes 1000 2000 3000 4000 --reps 3
```

Métricas obtenidas:

- Tiempo promedio
- GFLOPs externos
- RAM pico
- VRAM pico
- Uso promedio de GPU