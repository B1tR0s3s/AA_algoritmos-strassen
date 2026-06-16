import argparse
import csv
import os
import time
import threading
from pathlib import Path
from statistics import mean, stdev

import nbformat
import psutil
from nbclient import NotebookClient


class TimedNotebookClient(NotebookClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell_times = []

    def execute_cell(self, cell, cell_index, execution_count=None, store_history=True):
        start = time.perf_counter()
        status = "ok"
        error = ""

        try:
            result = super().execute_cell(
                cell,
                cell_index,
                execution_count=execution_count,
                store_history=store_history
            )
        except Exception as e:
            result = None
            status = "error"
            error = str(e)

        end = time.perf_counter()

        self.cell_times.append({
            "cell_index": cell_index,
            "cell_type": cell.get("cell_type", ""),
            "time_ms": (end - start) * 1000,
            "status": status,
            "error": error[:300]
        })

        if status == "error":
            raise RuntimeError(f"Error en la celda {cell_index}: {error}")

        return result


def monitor_memory(stop_event, samples, pid):
    while not stop_event.is_set():
        total_mb = 0.0

        try:
            parent = psutil.Process(pid)
            processes = [parent] + parent.children(recursive=True)

            for process in processes:
                try:
                    total_mb += process.memory_info().rss / (1024 * 1024)
                except psutil.NoSuchProcess:
                    pass

            samples.append(total_mb)

        except psutil.NoSuchProcess:
            pass

        time.sleep(0.1)


def remove_install_cells(nb):
    cleaned_cells = []

    for cell in nb.cells:
        source = cell.get("source", "").strip().lower()

        if source.startswith("pip install"):
            continue

        if source.startswith("!pip install"):
            continue

        if source.startswith("%pip install"):
            continue

        cleaned_cells.append(cell)

    nb.cells = cleaned_cells
    return nb


def run_notebook(notebook_path, output_notebook, skip_install=True, timeout=3600):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    if skip_install:
        nb = remove_install_cells(nb)

    memory_samples = []
    stop_event = threading.Event()

    monitor_thread = threading.Thread(
        target=monitor_memory,
        args=(stop_event, memory_samples, os.getpid())
    )

    client = TimedNotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={
            "metadata": {
                "path": str(Path(notebook_path).parent.resolve())
            }
        }
    )

    status = "ok"
    error = ""

    start = time.perf_counter()
    monitor_thread.start()

    try:
        client.execute()
    except Exception as e:
        status = "error"
        error = str(e)

    end = time.perf_counter()

    stop_event.set()
    monitor_thread.join()

    with open(output_notebook, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    total_time_ms = (end - start) * 1000
    peak_memory_mb = max(memory_samples) if memory_samples else 0.0
    avg_memory_mb = mean(memory_samples) if memory_samples else 0.0

    return {
        "status": status,
        "error": error[:300],
        "total_time_ms": total_time_ms,
        "peak_memory_mb": peak_memory_mb,
        "avg_memory_mb": avg_memory_mb,
        "cell_times": client.cell_times
    }


def main():
    parser = argparse.ArgumentParser(
        description="Medición de tiempo y memoria para el notebook F1000."
    )

    parser.add_argument(
        "notebook",
        help="Ruta del archivo .ipynb"
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Número de repeticiones"
    )

    parser.add_argument(
        "--include-install",
        action="store_true",
        help="Incluye celdas de instalación pip en la medición"
    )

    parser.add_argument(
        "--out",
        default="resultados_f1000.csv",
        help="Archivo CSV de resumen"
    )

    parser.add_argument(
        "--cells-out",
        default="resultados_f1000_celdas.csv",
        help="Archivo CSV con tiempos por celda"
    )

    args = parser.parse_args()

    notebook_path = Path(args.notebook)

    if not notebook_path.exists():
        raise FileNotFoundError(f"No existe el notebook: {notebook_path}")

    summary_rows = []
    cell_rows = []

    for rep in range(1, args.reps + 1):
        print(f"\nEjecutando repetición {rep}/{args.reps}")

        output_notebook = notebook_path.with_name(
            f"{notebook_path.stem}_ejecutado_rep{rep}.ipynb"
        )

        result = run_notebook(
            notebook_path=notebook_path,
            output_notebook=output_notebook,
            skip_install=not args.include_install
        )

        summary_rows.append({
            "repositorio": "F1000_Strassen_Notebook",
            "repeticion": rep,
            "status": result["status"],
            "total_time_ms": result["total_time_ms"],
            "peak_memory_mb": result["peak_memory_mb"],
            "avg_memory_mb": result["avg_memory_mb"],
            "error": result["error"]
        })

        for cell in result["cell_times"]:
            cell_rows.append({
                "repositorio": "F1000_Strassen_Notebook",
                "repeticion": rep,
                "cell_index": cell["cell_index"],
                "cell_type": cell["cell_type"],
                "time_ms": cell["time_ms"],
                "status": cell["status"],
                "error": cell["error"]
            })

        print(f"Estado: {result['status']}")
        print(f"Tiempo total: {result['total_time_ms']:.3f} ms")
        print(f"Memoria pico: {result['peak_memory_mb']:.3f} MB")
        print(f"Memoria promedio: {result['avg_memory_mb']:.3f} MB")

        if result["status"] == "error":
            print(f"Error: {result['error']}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "repositorio",
                "repeticion",
                "status",
                "total_time_ms",
                "peak_memory_mb",
                "avg_memory_mb",
                "error"
            ]
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    with open(args.cells_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "repositorio",
                "repeticion",
                "cell_index",
                "cell_type",
                "time_ms",
                "status",
                "error"
            ]
        )

        writer.writeheader()
        writer.writerows(cell_rows)

    print("\nMedición finalizada.")
    print(f"Resumen generado: {args.out}")
    print(f"Tiempos por celda: {args.cells_out}")


if __name__ == "__main__":
    main()