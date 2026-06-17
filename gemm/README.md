# General Matrix Multiplication (GEMM)

[![DOI](https://zenodo.org/badge/956219726.svg)](https://doi.org/10.5281/zenodo.17299738)

## Installation

```bash
# Descargar OpenBlas
wget https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.29.tar.gz
tar -xvzf v0.3.29.tar.gz
cd OpenBLAS-0.3.29

# Instalar
make -j$(nproc) USE_OPENMP=1
make PREFIX=~/openblas install
```

## Compila y ejecuta

```bash
./compile.sh <matrix-size>
./main
./mainCuda
```

## Compilar y ejecutar todos los test

```bash
./execute.sh
```

#### Compilación:

```
./compile.sh [tamano_matriz]
```

#### Ejecución:

```
./mainCuda
```

#### Medición:

```python
# Ejecutar evaluación completa (64, 128, 256, 512, 1024)
python evaluar_gemm.py --reps 30

# Solo CPU (si no hay GPU)
python evaluar_gemm.py --reps 30 --out evaluacion_cpu.csv

```

#### Métricas obtenidas:

- Tiempo promedio
- GFLOPs externos
- RAM pico
- VRAM pico
- Uso promedio de GPU

## Sample Results

Plotted are sample means with confidence intervals of FLOPS that were calculated from 30 runtimes for each algorithm and matrix size pair and resampled 10,000 times with replacement using bootstrapping.

![Performance v.s. Matrix Size](assets/gpu.png)

![Performance v.s. Matrix Size](assets/cpu.png)

![Detailed Performance v.s. Matrix Size](assets/table.png)
