# CUTLASS Stream-K GEMM Benchmark

This repository contains a script to build and run the official NVIDIA CUTLASS example:

```txt
examples/47_ampere_gemm_universal_streamk
```

The original example file is:

```txt
examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk.cu
```

That file is not a shell script. It is a CUDA/C++ file that is compiled to produce the executable:

```txt
47_ampere_gemm_universal_streamk
```

The goal of the script included in this repository is to automate the build and execution of that CUTLASS example.

---

## What This Example Does

The example evaluates a GEMM operation, that is, general matrix multiplication.

It compares several ways to run GEMM on the GPU:

- Basic data-parallel GEMM
- StreamK GEMM
- StreamK emulando data-parallel GEMM
- Basic Split-K GEMM
- StreamK emulando Split-K GEMM

This benchmark measures performance in milliseconds and GFLOPs for different matrix sizes.

---

## Requirements

- GPU NVIDIA compatible con CUDA
- CUDA Toolkit
- CMake
- Make
- CUTLASS clonado localmente

Clone CUTLASS:

```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
```

---

## Using the Script

This repository contains a script that automates the required commands.

Grant execution permissions:

```bash
chmod +x run_streamk.sh
```

Run the script:

```bash
./run_streamk.sh
```

> If the script has a different name, replace `run_streamk.sh` with the actual file name.

---

## Manual Build

You can also build the example manually from the root of the CUTLASS repository.

---

## RTX 3060 / 3070 / 3080 / 3090

For RTX 30 series GPUs, use CUDA architecture:

```txt
sm_86
```

Commands:

```bash
mkdir build
cd build

cmake .. -DCUTLASS_NVCC_ARCHS=86 -DCUTLASS_ENABLE_TESTS=OFF
make 47_ampere_gemm_universal_streamk -j$(nproc)
```

---

## RTX 4060 / 4070 / 4080 / 4090

For RTX 40 series GPUs, use CUDA architecture:

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

## Run the Example

From the `cutlass/build` directory:

```bash
./examples/47_ampere_gemm_universal_streamk/47_ampere_gemm_universal_streamk
```

---

## Run Benchmarks

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

## Available Parameters

The executable lets you modify the GEMM problem size and some execution parameters.

```txt
--m              GEMM M dimension
--n              GEMM N dimension
--k              GEMM K dimension
--alpha          Alpha scalar
--beta           Beta scalar
--split          Split-K factor
--iterations     Number of benchmark iterations
--help           Show the executable help
```

Ejemplo:

```bash
./examples/47_ampere_gemm_universal_streamk/47_ampere_gemm_universal_streamk --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707
```

---

## View Help

```bash
./examples/47_ampere_gemm_universal_streamk/47_ampere_gemm_universal_streamk --help
```

---

## Clean Build

If you need to rebuild from scratch:

```bash
rm -rf build
```

Then run the script again or the manual commands that match your GPU.

---

## Note

This repository does not modify the original CUTLASS example. It only provides a script to make building and running it easier.
