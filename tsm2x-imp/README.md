# TSM2X: High-Performance Matrix-Matrix Multiplication for Tall-and-Skinny Matrices on GPUs

by
Cody Rivera [[cjrivera1@crimson.ua.edu](mailto:cjrivera1@crimson.ua.edu)],
Jieyang Chen [[chenj3@ornl.gov](mailto:chenj3@ornl.gov)] y
Dingwen Tao [[dingwen.tao@wsu.edu](mailto:dingwen.tao@wsu.edu)]

This repository contains an implementation of two irregular-shape matrix-matrix multiplication algorithms: `TSM2R` and `TSM2L`.

`TSM2R` is designed to efficiently multiply a large, or nearly square, matrix by a tall-and-skinny matrix. More specifically, it performs matrix-matrix multiplication between a matrix of size `(m * k)` and a matrix of size `(k * n)`, where `m` and `k` are approximately equal, and `n` is much smaller than `k`.

`TSM2L` is designed to efficiently multiply a tall-and-skinny matrix by a small square matrix. More specifically, it performs matrix-matrix multiplication between a matrix of size `(m * k)` and a matrix of size `(k * n)`, where `k` is much smaller than `m`, and `k` and `n` are approximately equal.

We propose `TSM2R` and `TSM2L` in our preprint,
“TSM2X: High-Performance Tall-and-Skinny Matrix-Matrix Multiplication on GPUs” [1].

Our work extends an ICS conference paper [2] that introduces `TSM2R`, expanding its techniques to different matrix sizes and porting the algorithm to the Nvidia Tesla V100 GPU.

We implemented the kernels as templates, with parameters `t1`, `t2`, and `t3` as template variables [1]. The program selects an optimal kernel depending on the size of the input matrices.

Currently, this repository provides a set of optimal kernels only for the Nvidia V100 GPU.

## Instructions:

This implementation is designed for Unix platforms and can be built using `make`.

Important note:
> The Makefile must be modified to match the type of GPU you have; by default, it is set to sm_70 in the original repository.

Program usage:

```bash
./multiply [-d] [-i] a.mtx b.mtx c.mtx
```

where `a.mtx` and `b.mtx` are input matrices, and `c.mtx` is the output matrix.

The `-d` option indicates double-precision matrices, while `-i` indicates that `TSM2L` will be used instead of `TSM2R`.

Matrix format is binary, with the following structure:

```C++
template <typename FloatType>
struct matrixFormat {
    uint32_t rows, cols;
    FloatType values[rows * cols];
};
```

The matrix is stored in column-major format, that is, by columns.

All multibyte values are in little-endian format.

You can use the provided `gen.cpp` program to generate input matrices.

El uso es:

```bash
./gen [-d] -r ROW_COUNT -c COL_COUNT file
```

where `-d` indicates double precision.

You can also use the provided `print.cpp` program to print matrices.

El uso es:

```bash
./print [-d] file
```
https://github.com/codyjrivera/tsm2x-imp/tree/master
To evaluate performance across a range of inputs, a Python 3 script named `test.py` is provided.

El script puede ejecutarse con:

```bash
python3 test.py
```

The measurement run is performed with:

```bash
python medicion_tsm2x.py
```
#### Metrics obtained:

- Tiempo promedio
- RAM pico
- VRAM pico
- Uso promedio de GPU


Original repository:
https://github.com/codyjrivera/tsm2x-imp/tree/master

The program requires `../multiply` and `../gen` to exist, and writes its output to CSV files.

## Notes:

[1] Cody Rivera, Jieyang Chen, Nan Xiong, Shuaiwen Leon Song y Dingwen Tao.
“TSM2X: High-Performance Tall-and-Skinny Matrix-Matrix Multiplication on GPUs.”
2020. [arXiv:2002.03258](https://arxiv.org/abs/2002.03258v4) [cs.DC].

[2] Jieyang Chen, Nan Xiong, Xin Liang, Dingwen Tao, Sihuan Li, Kaiming Ouyang, Kai Zhao, Nathan DeBardeleben, Qiang Guan y Zizhong Chen.
“TSM2: optimizing tall-and-skinny matrix-matrix multiplication on GPUs.”
En *Proceedings of the ACM International Conference on Supercomputing* (ICS), pp. 106–116. ACM, 2019.
https://doi.org/10.1145/3330345.3330355
