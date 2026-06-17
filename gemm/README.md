# General Matrix Multiplication (GEMM)

[![DOI](https://zenodo.org/badge/956219726.svg)](https://doi.org/10.5281/zenodo.17299738)

Implementación y evaluación de rendimiento de la multiplicación general de matrices (GEMM) en CPU y GPU.

## Instalación

```bash
# Descargar OpenBLAS
wget https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.29.tar.gz
tar -xvzf v0.3.29.tar.gz
cd OpenBLAS-0.3.29

# Compilar e instalar
make -j$(nproc) USE_OPENMP=1
make PREFIX=~/openblas install
```

## Compilación y ejecución

Compilar el proyecto para un tamaño de matriz específico:

```bash
./compile.sh <tamano_matriz>
```

Ejecutar la versión para CPU:

```bash
./main
```

Ejecutar la versión para GPU (CUDA):

```bash
./mainCuda
```

## Ejecutar todas las pruebas

```bash
./execute.sh
```

## Evaluación de rendimiento

Ejecutar la batería completa de pruebas:

```bash
# Evaluar todos los tamaños de matriz (64, 128, 256, 512, 1024)
python evaluar_gemm.py --reps 30

# Evaluación solo en CPU
python evaluar_gemm.py --reps 30 --out evaluacion_cpu.csv
```

### Métricas obtenidas

- Tiempo promedio de ejecución
- GFLOPs
- Uso máximo de RAM
- Uso máximo de VRAM
- Utilización promedio de GPU

## Resultados de ejemplo

Los siguientes gráficos fueron generados con `evaluar_gemm.py`. Los resultados corresponden a 30 ejecuciones por tamaño de matriz y muestran el rendimiento promedio (GFLOPs) junto con intervalos de confianza del 95% estimados mediante *bootstrapping*.

### Rendimiento en GPU

![Performance v.s. Matrix Size](assets/gpu.png)

### Rendimiento en CPU

![Performance v.s. Matrix Size](assets/cpu.png)

### Resultados detallados

![Detailed Performance v.s. Matrix Size](assets/table.png)