# Multiplicación de matrices mediante Strassen y CUDA

Este repositorio contiene el desarrollo experimental del proyecto orientado a evaluar una propuesta híbrida para la multiplicación de matrices densas mediante el algoritmo de Strassen, CUDA y GEMM. El objetivo principal es analizar el rendimiento práctico de una estrategia que combine reducción algorítmica y aceleración en GPU.

> Nota: Se desarrolló un script de medición para cada repositorio, encargado de ejecutar los experimentos y registrar las métricas de rendimiento definidas para cada implementación. La obtención de resultados requiere la ejecución de dichos scripts. Para los repositorios 1, 3 y 4, las pruebas se realizan utilizando matrices cuadradas de dimensión $n x n$, tomando los valores de 64, 128, 256, 512 y 1024; para el cuarto repositorio se uso matrices de $n x k$. 

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

1. F1000 - Notebook MCM + Strassen
## Ejecución de mediciones

### En Visual Studio Code
Requisitos:
- Jupiter - Google Colab
- Python 3

Instrucciones:
- Cambiar el entorno de ejecucion a 


El notebook original fue usado como base. Para obtener resultados reales por tamaño de matriz, se reemplazó la celda de datos estáticos por una celda de ejecución que genera matrices, calcula la multiplicación estándar y Strassen, mide tiempos, memoria y error numérico.


Métricas obtenidas:

- Tiempo total de ejecución
- Memoria RAM pico
- Memoria RAM promedio
- Tiempo por celda

2. TSM2X

Compilación:

```bash
make
```

Ejecución de medición:

``` bash
cd tsm2x-imp
```

```bash
python medicion_tsm2x.py
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
# Activar entorno con dependencias (si es necesario)
pip install numpy psutil

# Ejecutar evaluación completa (64, 128, 256, 512, 1024)
python evaluar_gemm.py --reps 30

# Solo CPU (si no hay GPU)
python evaluar_gemm.py --reps 30 --out evaluacion_cpu.csv

```

Métricas obtenidas:

- Tiempo promedio
- GFLOPs externos
- RAM pico
- VRAM pico
- Uso promedio de GPU