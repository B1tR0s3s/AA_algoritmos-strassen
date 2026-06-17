## Ejecucion de repositorios
### 1. F1000 - Notebook MCM + Strassen
#### Ejecución de mediciones

#### En Visual Studio Code
Requisitos:
- Python 3
- Extensión Jupyter para Visual Studio Code o acceso a Google Colab
- Cuenta de Google (si se utilizará Google Colab)

Instrucciones:
- Abrir el notebook en Visual Studio Code.
- Seleccionar un entorno de ejecución compatible: Jupyter (local) o Google Colab.
- Si se utiliza Google Colab, iniciar sesión con una cuenta de Google.
- Verificar que el kernel de Python 3 esté seleccionado.
- Ejecutar todas las celdas del notebook mediante la opción Run All.
- Esperar a que finalice la ejecución para obtener las mediciones y resultados correspondientes.

El notebook original fue usado como base. Para obtener resultados reales por tamaño de matriz, se reemplazó la celda de datos estáticos por una celda de ejecución que genera matrices, calcula la multiplicación estándar y Strassen, mide tiempos, memoria y error numérico.


#### Métricas obtenidas:

- Tiempo total de ejecución
- Memoria RAM pico
- Memoria RAM promedio
- Tiempo por celda
