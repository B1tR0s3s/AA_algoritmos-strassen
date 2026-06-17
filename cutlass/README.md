# CUTLASS Stream-K GEMM Benchmark

Este repositorio contiene un script para compilar y ejecutar el ejemplo oficial de NVIDIA CUTLASS:

```txt
examples/47_ampere_gemm_universal_streamk
```

El archivo original del ejemplo es:

```txt
examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk.cu
```

Ese archivo no es un script shell. Es un archivo CUDA/C++ que se compila para generar el ejecutable:

```txt
47_ampere_gemm_universal_streamk
```

El objetivo del script incluido en este repositorio es automatizar la compilación y ejecución de ese ejemplo de CUTLASS.

---

## Qué hace este ejemplo

El ejemplo evalúa una operación GEMM, es decir, una multiplicación general de matrices.

Compara varias formas de ejecutar GEMM en GPU:

- Basic data-parallel GEMM
- StreamK GEMM
- StreamK emulando data-parallel GEMM
- Basic Split-K GEMM
- StreamK emulando Split-K GEMM

Este benchmark permite medir rendimiento en milisegundos y GFLOPs para distintos tamaños de matrices.

---

## Requisitos

- GPU NVIDIA compatible con CUDA
- CUDA Toolkit
- CMake
- Make
- CUTLASS clonado localmente

Clonar CUTLASS:

```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
```

---

## Uso con script

Este repositorio contiene un script que automatiza los comandos necesarios.

Dar permisos de ejecución:

```bash
chmod +x run_streamk.sh
```

Ejecutar el script:

```bash
./run_streamk.sh
```

> Si el script tiene otro nombre, reemplaza `run_streamk.sh` por el nombre real del archivo.

---

## Compilación manual

También puedes compilar el ejemplo manualmente desde la raíz del repositorio de CUTLASS.

---

## RTX 3060 / 3070 / 3080 / 3090

Para GPUs RTX serie 30 se usa arquitectura CUDA:

```txt
sm_86
```

Comandos:

```bash
mkdir build
cd build

cmake .. -DCUTLASS_NVCC_ARCHS=86 -DCUTLASS_ENABLE_TESTS=OFF
make 47_ampere_gemm_universal_streamk -j$(nproc)
```

---

## RTX 4060 / 4070 / 4080 / 4090

Para GPUs RTX serie 40 se usa arquitectura CUDA:

```txt
sm_89
```

Comandos:

```bash
mkdir build
cd build

cmake .. -DCUTLASS_NVCC_ARCHS=89 -DCUTLASS_ENABLE_TESTS=OFF
make 47_ampere_gemm_universal_streamk -j$(nproc)
```

---

## Ejecutar el ejemplo

Desde el directorio `cutlass/build`:

```bash
./examples/47_ampere_gemm_universal_streamk/47_ampere_gemm_universal_streamk
```

---

## Ejecutar benchmarks

### GEMM 1024 x 1024 x 1024

```bash
./examples/47_ampere_gemm_universal_streamk/47_ampere_gemm_universal_streamk --m=1024 --n=1024 --k=1024 --iterations=1000
```

### GEMM 2048 x 2048 x 2048

```bash
./examples/47_ampere_gemm_universal_streamk/47_ampere_gemm_universal_streamk --m=2048 --n=2048 --k=2048 --iterations=1000
```

### GEMM 4096 x 4096 x 4096

```bash
./examples/47_ampere_gemm_universal_streamk/47_ampere_gemm_universal_streamk --m=4096 --n=4096 --k=4096 --iterations=300
```

---

## Parámetros disponibles

El ejecutable permite modificar el tamaño del problema GEMM y algunos parámetros de ejecución.

```txt
--m              Dimensión M de GEMM
--n              Dimensión N de GEMM
--k              Dimensión K de GEMM
--alpha          Escalar alpha
--beta           Escalar beta
--split          Factor Split-K
--iterations     Número de iteraciones del benchmark
--help           Muestra la ayuda del ejecutable
```

Ejemplo:

```bash
./examples/47_ampere_gemm_universal_streamk/47_ampere_gemm_universal_streamk --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707
```

---

## Ver ayuda

```bash
./examples/47_ampere_gemm_universal_streamk/47_ampere_gemm_universal_streamk --help
```

---

## Limpiar compilación

Si necesitas compilar desde cero:

```bash
rm -rf build
```

Luego vuelve a ejecutar el script o los comandos manuales correspondientes a tu GPU.

---

## Nota

Este repositorio no modifica el ejemplo original de CUTLASS. Solo proporciona un script para facilitar su compilación y ejecución.