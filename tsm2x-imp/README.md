# TSM2X: Multiplicación matriz-matriz de alto rendimiento para matrices altas y delgadas en GPU

por
Cody Rivera [[cjrivera1@crimson.ua.edu](mailto:cjrivera1@crimson.ua.edu)],
Jieyang Chen [[chenj3@ornl.gov](mailto:chenj3@ornl.gov)] y
Dingwen Tao [[dingwen.tao@wsu.edu](mailto:dingwen.tao@wsu.edu)]

Este repositorio contiene una implementación de dos algoritmos de multiplicación matriz-matriz con formas irregulares: `TSM2R` y `TSM2L`.

`TSM2R` está diseñado para multiplicar eficientemente una matriz cuadrada grande —o casi cuadrada— por una matriz alta y delgada. Más específicamente, realiza una multiplicación matriz-matriz entre una matriz de tamaño `(m * k)` y una matriz de tamaño `(k * n)`, donde `m` y `k` son aproximadamente iguales, y `n` es mucho menor que `k`.

`TSM2L` está diseñado para multiplicar eficientemente una matriz alta y delgada por una matriz cuadrada pequeña. Más específicamente, realiza una multiplicación matriz-matriz entre una matriz de tamaño `(m * k)` y una matriz de tamaño `(k * n)`, donde `k` es mucho menor que `m`, y `k` y `n` son aproximadamente iguales.

Proponemos `TSM2R` y `TSM2L` en nuestro preprint,
“TSM2X: High-Performance Tall-and-Skinny Matrix-Matrix Multiplication on GPUs” [1].

Nuestro trabajo extiende un artículo de la conferencia ICS [2], el cual introduce `TSM2R`, ampliando sus técnicas para distintos tamaños de matrices, además de portar el algoritmo a la GPU Nvidia Tesla V100.

Hemos implementado los kernels como plantillas, con los parámetros `t1`, `t2` y `t3` como variables de plantilla [1]. El programa seleccionará un kernel óptimo dependiendo del tamaño de las matrices de entrada.

Actualmente, este repositorio proporciona un conjunto de kernels óptimos únicamente para la GPU Nvidia V100.

## Instrucciones:

Esta implementación está diseñada para plataformas Unix y puede compilarse usando `make`.

Observacion importante:
> Se tiene que modificar Makefile para adaptarlo al tipo de GPU que se posea, por defecto, se encuentra con sm_70 en el repositorio original

El uso de este programa es:

```bash
./multiply [-d] [-i] a.mtx b.mtx c.mtx
```

donde `a.mtx` y `b.mtx` son matrices de entrada, y `c.mtx` es la matriz de salida.

La opción `-d` indica que las matrices son de doble precisión, mientras que `-i` indica que se usará `TSM2L` en lugar de `TSM2R`.

El formato de las matrices es binario, con la siguiente estructura:

```C++
template <typename FloatType>
struct matrixFormat {
    uint32_t rows, cols;
    FloatType values[rows * cols];
};
```

La matriz se almacena en formato column-major, es decir, por columnas.

Todos los valores multibyte están en formato little-endian.

Puede utilizar el programa proporcionado `gen.cpp` para generar matrices de entrada.

El uso es:

```bash
./gen [-d] -r ROW_COUNT -c COL_COUNT file
```

donde `-d` indica doble precisión.

También puede utilizar el programa proporcionado `print.cpp` para imprimir matrices.

El uso es:

```bash
./print [-d] file
```
https://github.com/codyjrivera/tsm2x-imp/tree/master
Para evaluar el rendimiento en un rango de entradas, se proporciona un script de Python 3 llamado `test.py`.

El script puede ejecutarse con:

```bash
python3 test.py
```

Repositorio original: 
https://github.com/codyjrivera/tsm2x-imp/tree/master

El programa requiere que existan `../multiply` y `../gen`, y escribe su salida en archivos CSV.

## Notas:

[1] Cody Rivera, Jieyang Chen, Nan Xiong, Shuaiwen Leon Song y Dingwen Tao.
“TSM2X: High-Performance Tall-and-Skinny Matrix-Matrix Multiplication on GPUs.”
2020. [arXiv:2002.03258](https://arxiv.org/abs/2002.03258v4) [cs.DC].

[2] Jieyang Chen, Nan Xiong, Xin Liang, Dingwen Tao, Sihuan Li, Kaiming Ouyang, Kai Zhao, Nathan DeBardeleben, Qiang Guan y Zizhong Chen.
“TSM2: optimizing tall-and-skinny matrix-matrix multiplication on GPUs.”
En *Proceedings of the ACM International Conference on Supercomputing* (ICS), pp. 106–116. ACM, 2019.
https://doi.org/10.1145/3330345.3330355

