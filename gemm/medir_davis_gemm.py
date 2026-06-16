import argparse
import csv
import subprocess
import time
import threading
from pathlib import Path
from statistics import mean, stdev

import psutil


# ============================================================
# Monitoreo de RAM del proceso
# ============================================================

def monitor_ram(proc, stop_event, samples):
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
# Monitoreo de GPU
# ============================================================

def monitor_gpu(stop_event, samples):
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
                    "vram_mb": float(mem.strip()),
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
        target=monitor_ram,
        args=(proc, stop_event, ram_samples)
    )

    gpu_thread = threading.Thread(
        target=monitor_gpu,
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

    vram_peak_mb = max([x["vram_mb"] for x in gpu_samples]) if gpu_samples else 0.0
    gpu_util_avg = mean([x["gpu_util"] for x in gpu_samples]) if gpu_samples else 0.0

    return {
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "wall_time_ms": wall_time_ms,
        "ram_peak_mb": ram_peak_mb,
        "ram_avg_mb": ram_avg_mb,
        "vram_peak_mb": vram_peak_mb,
        "gpu_util_avg": gpu_util_avg
    }


# ============================================================
# Cálculo de GFLOPS
# ============================================================

def calcular_gflops(n, tiempo_ms):
    segundos = tiempo_ms / 1000.0

    if segundos <= 0:
        return 0.0

    operaciones = (2 * (n ** 3)) - (n ** 2)
    gflops = operaciones / segundos / 1e9

    return gflops


# ============================================================
# Compilación
# ============================================================

def compilar(repo, size):
    command = f"./compile.sh {size}"

    result = subprocess.run(
        command,
        cwd=repo,
        shell=True,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Error al compilar con tamaño {size}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Medición de rendimiento para davisethan/gemm."
    )

    parser.add_argument(
        "--repo",
        default=".",
        help="Ruta del repositorio gemm"
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1000, 2000, 3000, 4000],
        help="Tamaños de matriz N x N"
    )

    parser.add_argument(
        "--programs",
        nargs="+",
        default=["mainCuda"],
        help="Programas a ejecutar: mainCuda, main"
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Número de repeticiones"
    )

    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="No compilar antes de ejecutar"
    )

    parser.add_argument(
        "--out",
        default="resultados_davis_gemm.csv",
        help="Archivo CSV de salida"
    )

    args = parser.parse_args()

    repo = Path(args.repo).resolve()

    if not repo.exists():
        raise FileNotFoundError(f"No existe el repositorio: {repo}")

    compile_script = repo / "compile.sh"

    if not compile_script.exists():
        raise FileNotFoundError("No se encontró compile.sh")

    rows = []

    for size in args.sizes:
        print(f"\nTamaño de matriz: {size}x{size}")

        if not args.skip_compile:
            print("Compilando...")
            compilar(repo, size)

        for program in args.programs:
            program_path = repo / program

            if not program_path.exists():
                print(f"Omitido: no existe ./{program}")
                continue

            tiempos = []
            gflops = []
            ram_peaks = []
            vram_peaks = []
            gpu_utils = []

            for rep in range(1, args.reps + 1):
                command = f"./{program}"

                print(f"  Ejecutando {program} | repetición {rep}/{args.reps}")

                result = run_measured(command, repo)

                tiempo_ms = result["wall_time_ms"]
                gflops_calc = calcular_gflops(size, tiempo_ms)

                tiempos.append(tiempo_ms)
                gflops.append(gflops_calc)
                ram_peaks.append(result["ram_peak_mb"])
                vram_peaks.append(result["vram_peak_mb"])
                gpu_utils.append(result["gpu_util_avg"])

                print(
                    f"    Tiempo: {tiempo_ms:.3f} ms | "
                    f"GFLOPS externo: {gflops_calc:.3f} | "
                    f"RAM pico: {result['ram_peak_mb']:.2f} MB | "
                    f"VRAM pico: {result['vram_peak_mb']:.2f} MB | "
                    f"GPU uso prom.: {result['gpu_util_avg']:.2f}%"
                )

                if result["returncode"] != 0:
                    print("    Error:")
                    print(result["stderr"][:500])

            rows.append({
                "repositorio": "davisethan_gemm",
                "programa": program,
                "size": size,
                "repeticiones": args.reps,
                "tiempo_prom_ms": mean(tiempos),
                "tiempo_std_ms": stdev(tiempos) if len(tiempos) > 1 else 0.0,
                "gflops_prom_externo": mean(gflops),
                "gflops_std_externo": stdev(gflops) if len(gflops) > 1 else 0.0,
                "ram_pico_prom_mb": mean(ram_peaks),
                "vram_pico_prom_mb": mean(vram_peaks),
                "gpu_util_prom": mean(gpu_utils)
            })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "repositorio",
            "programa",
            "size",
            "repeticiones",
            "tiempo_prom_ms",
            "tiempo_std_ms",
            "gflops_prom_externo",
            "gflops_std_externo",
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