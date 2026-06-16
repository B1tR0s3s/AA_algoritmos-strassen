import argparse
import csv
import subprocess
import time
import threading
from pathlib import Path
from statistics import mean, stdev

import psutil


# ============================================================
# Monitor de memoria RAM del proceso
# ============================================================

def memory_monitor_process(proc, stop_event, samples):
    while not stop_event.is_set():
        try:
            p = psutil.Process(proc.pid)
            procesos = [p] + p.children(recursive=True)

            total_mb = 0.0
            for proceso in procesos:
                try:
                    total_mb += proceso.memory_info().rss / (1024 * 1024)
                except psutil.NoSuchProcess:
                    pass

            samples.append(total_mb)

        except psutil.NoSuchProcess:
            pass

        time.sleep(0.05)


# ============================================================
# Monitor de GPU con nvidia-smi
# ============================================================

def memory_monitor_gpu(stop_event, samples):
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=2
            )

            line = result.stdout.strip().split("\n")[0]

            if line:
                mem, util = line.split(",")
                samples.append({
                    "gpu_memory_mb": float(mem.strip()),
                    "gpu_util": float(util.strip())
                })

        except Exception:
            pass

        time.sleep(0.1)


# ============================================================
# Ejecución medida
# ============================================================

def run_measured(command, cwd):
    ram_samples = []
    gpu_samples = []

    stop_event = threading.Event()

    start = time.perf_counter()

    proc = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    ram_thread = threading.Thread(
        target=memory_monitor_process,
        args=(proc, stop_event, ram_samples)
    )

    gpu_thread = threading.Thread(
        target=memory_monitor_gpu,
        args=(stop_event, gpu_samples)
    )

    ram_thread.start()
    gpu_thread.start()

    stdout, stderr = proc.communicate()

    end = time.perf_counter()

    stop_event.set()
    ram_thread.join()
    gpu_thread.join()

    wall_time_ms = (end - start) * 1000

    ram_peak_mb = max(ram_samples) if ram_samples else 0.0
    ram_avg_mb = mean(ram_samples) if ram_samples else 0.0

    gpu_mem_peak_mb = max([x["gpu_memory_mb"] for x in gpu_samples]) if gpu_samples else 0.0
    gpu_util_avg = mean([x["gpu_util"] for x in gpu_samples]) if gpu_samples else 0.0

    return {
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "wall_time_ms": wall_time_ms,
        "ram_peak_mb": ram_peak_mb,
        "ram_avg_mb": ram_avg_mb,
        "gpu_mem_peak_mb": gpu_mem_peak_mb,
        "gpu_util_avg": gpu_util_avg
    }


# ============================================================
# Generación de matrices
# ============================================================

def generate_matrix(rows, cols, filename, double_precision):
    flag = "-d" if double_precision else ""
    command = f"./gen {flag} -r {rows} -c {cols} {filename}"
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Error generando matriz {filename}\n{result.stderr}"
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Medición de multiplicación de matrices en TSM2X."
    )

    parser.add_argument(
        "--repo",
        default=".",
        help="Ruta del repositorio tsm2x-imp"
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1024, 2048, 4096],
        help="Tamaños N para matrices principales"
    )

    parser.add_argument(
        "--cols",
        nargs="+",
        type=int,
        default=[2, 4, 8, 16],
        help="Columnas de la matriz tall-and-skinny"
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Número de repeticiones"
    )

    parser.add_argument(
        "--double",
        action="store_true",
        help="Usar doble precisión"
    )

    parser.add_argument(
        "--mode",
        choices=["right", "left"],
        default="right",
        help="right = TSM2R, left = TSM2L"
    )

    parser.add_argument(
        "--out",
        default="resultados_tsm2x.csv",
        help="Archivo CSV de salida"
    )

    args = parser.parse_args()

    repo = Path(args.repo).resolve()

    multiply_path = repo / "multiply"
    gen_path = repo / "gen"

    if not multiply_path.exists():
        raise FileNotFoundError(
            "No se encontró ./multiply. Ejecuta primero: make"
        )

    if not gen_path.exists():
        raise FileNotFoundError(
            "No se encontró ./gen. Ejecuta primero: make"
        )

    data_dir = repo / "matrices_generadas"
    data_dir.mkdir(exist_ok=True)

    rows = []

    for size in args.sizes:
        for cols in args.cols:
            print(f"\nPreparando prueba: size={size}, cols={cols}, mode={args.mode}")

            if args.mode == "right":
                # TSM2R:
                # A = size x size
                # B = size x cols
                a_rows, a_cols = size, size
                b_rows, b_cols = size, cols
                tsm_flag = ""

            else:
                # TSM2L:
                # A = size x cols
                # B = cols x cols
                a_rows, a_cols = size, cols
                b_rows, b_cols = cols, cols
                tsm_flag = "-i"

            precision = "double" if args.double else "single"

            a_file = data_dir / f"A_{args.mode}_{size}_{cols}_{precision}.mtx"
            b_file = data_dir / f"B_{args.mode}_{size}_{cols}_{precision}.mtx"
            c_file = data_dir / f"C_{args.mode}_{size}_{cols}_{precision}.mtx"

            print("Generando matriz A...")
            generate_matrix(a_rows, a_cols, a_file, args.double)

            print("Generando matriz B...")
            generate_matrix(b_rows, b_cols, b_file, args.double)

            tiempos = []
            ram_peaks = []
            gpu_peaks = []
            gpu_utils = []

            for rep in range(1, args.reps + 1):
                double_flag = "-d" if args.double else ""

                command = f"./multiply {double_flag} {tsm_flag} {a_file} {b_file} {c_file}"

                command = " ".join(command.split())

                print(f"  Repetición {rep}/{args.reps}")
                print(f"  Comando: {command}")

                result = run_measured(command, repo)

                tiempos.append(result["wall_time_ms"])
                ram_peaks.append(result["ram_peak_mb"])
                gpu_peaks.append(result["gpu_mem_peak_mb"])
                gpu_utils.append(result["gpu_util_avg"])

                print(
                    f"    Tiempo: {result['wall_time_ms']:.3f} ms | "
                    f"RAM pico: {result['ram_peak_mb']:.2f} MB | "
                    f"VRAM pico: {result['gpu_mem_peak_mb']:.2f} MB | "
                    f"GPU uso prom.: {result['gpu_util_avg']:.2f}%"
                )

                if result["returncode"] != 0:
                    print("    Error:")
                    print(result["stderr"][:500])

            row = {
                "repositorio": "TSM2X",
                "modo": "TSM2R" if args.mode == "right" else "TSM2L",
                "precision": precision,
                "size": size,
                "cols": cols,
                "repeticiones": args.reps,
                "tiempo_prom_ms": mean(tiempos),
                "tiempo_std_ms": stdev(tiempos) if len(tiempos) > 1 else 0.0,
                "ram_pico_prom_mb": mean(ram_peaks),
                "vram_pico_prom_mb": mean(gpu_peaks),
                "gpu_util_prom": mean(gpu_utils)
            }

            rows.append(row)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "repositorio",
            "modo",
            "precision",
            "size",
            "cols",
            "repeticiones",
            "tiempo_prom_ms",
            "tiempo_std_ms",
            "ram_pico_prom_mb",
            "vram_pico_prom_mb",
            "gpu_util_prom"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResultados guardados en: {args.out}")


if __name__ == "__main__":
    main()