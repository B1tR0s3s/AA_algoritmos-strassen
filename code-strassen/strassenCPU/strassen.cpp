#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>  
using namespace std;
using namespace std::chrono;

typedef vector<vector<int>> Matrix;

Matrix crearMatriz(int n)
{
    return Matrix(n, vector<int>(n, 0));
}

Matrix sumar(const Matrix& A, const Matrix& B)
{
    int n = A.size();
    Matrix C = crearMatriz(n);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];

    return C;
}

Matrix restar(const Matrix& A, const Matrix& B)
{
    int n = A.size();
    Matrix C = crearMatriz(n);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];

    return C;
}

Matrix strassen(const Matrix& A, const Matrix& B)
{
    int n = A.size();

    if (n == 1)
    {
        Matrix C = crearMatriz(1);
        C[0][0] = A[0][0] * B[0][0];
        return C;
    }

    int m = n / 2;

    Matrix A11 = crearMatriz(m);
    Matrix A12 = crearMatriz(m);
    Matrix A21 = crearMatriz(m);
    Matrix A22 = crearMatriz(m);

    Matrix B11 = crearMatriz(m);
    Matrix B12 = crearMatriz(m);
    Matrix B21 = crearMatriz(m);
    Matrix B22 = crearMatriz(m);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + m];
            A21[i][j] = A[i + m][j];
            A22[i][j] = A[i + m][j + m];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + m];
            B21[i][j] = B[i + m][j];
            B22[i][j] = B[i + m][j + m];
        }
    }

    Matrix M1 = strassen(sumar(A11, A22), sumar(B11, B22));
    Matrix M2 = strassen(sumar(A21, A22), B11);
    Matrix M3 = strassen(A11, restar(B12, B22));
    Matrix M4 = strassen(A22, restar(B21, B11));
    Matrix M5 = strassen(sumar(A11, A12), B22);
    Matrix M6 = strassen(restar(A21, A11), sumar(B11, B12));
    Matrix M7 = strassen(restar(A12, A22), sumar(B21, B22));

    Matrix C11 = sumar(restar(sumar(M1, M4), M5), M7);
    Matrix C12 = sumar(M3, M5);
    Matrix C21 = sumar(M2, M4);
    Matrix C22 = sumar(restar(sumar(M1, M3), M2), M6);

    Matrix C = crearMatriz(n);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            C[i][j] = C11[i][j];
            C[i][j + m] = C12[i][j];
            C[i + m][j] = C21[i][j];
            C[i + m][j + m] = C22[i][j];
        }
    }

    return C;
}

Matrix leerMatriz(const string& nombreArchivo)
{
    ifstream archivo(nombreArchivo);

    if (!archivo.is_open())
    {
        cerr << "Error al abrir " << nombreArchivo << endl;
        exit(1);
    }

    string linea;
    int n = 0;

    while (getline(archivo, linea))
    {
        if (linea.empty())
            continue;

        if (linea[0] == '#')
            continue;

        n = stoi(linea);
        break;
    }

    Matrix M(n, vector<int>(n));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            archivo >> M[i][j];
        }
    }

    archivo.close();

    return M;
}

void guardarMatriz(const Matrix& M, const string& nombreArchivo)
{
    ofstream archivo(nombreArchivo);

    int n = M.size();

    archivo << n << "\n";

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            archivo << M[i][j];

            if (j < n - 1)
                archivo << " ";
        }

        archivo << "\n";
    }

    archivo.close();
}

int main()
{
    cout << "Leyendo A.txt..." << endl;
    Matrix A = leerMatriz("A.txt");

    cout << "Leyendo B.txt..." << endl;
    Matrix B = leerMatriz("B.txt");

    if (A.size() != B.size())
    {
        cerr << "Error: las matrices tienen tamaños diferentes." << endl;
        return 1;
    }

    auto inicio = high_resolution_clock::now();

    Matrix C = strassen(A, B);

    auto fin = high_resolution_clock::now();

    double tiempo_ms =
        duration<double, milli>(fin - inicio).count();

    cout << "\nResultado de la multiplicacion:\n\n";

    guardarMatriz(C, "C.txt");

    cout << "\nResultado guardado en C.txt" << endl;

    cout << "\nTiempo de ejecucion: "
         << tiempo_ms
         << " ms" << endl;

    return 0;
}