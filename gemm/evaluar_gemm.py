#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
import time
import re
import threading
from pathlib import Path
from statistics import mean, stdev

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None
    print("Advertencia: psutil no instalado. No se medirá RAM.", file=sys.stderr)


def monitor_ram(proc, stop_event, samples):
    if psutil is None:
        return
    while not stop_event.is_set():
        try:
            p = psutil.Process(proc.pid)
            total_mb = 0.0
            for proceso in [p] + p.children(recursive=True):
                try:
                    total_mb += proceso.memory_info().rss / (1024 * 1024)
                except psutil.NoSuchProcess:
                    pass
            samples.append(total_mb)
        except psutil.NoSuchProcess:
            pass
        time.sleep(0.05)


def monitor_gpu(stop_event, samples):
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
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


def run_measured(command, cwd):
    ram_samples = []
    gpu_samples = []
    stop_event = threading.Event()

    start = time.perf_counter()
    proc = subprocess.Popen(
        command, cwd=cwd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    ram_thread = threading.Thread(target=monitor_ram, args=(proc, stop_event, ram_samples))
    gpu_thread = threading.Thread(target=monitor_gpu, args=(stop_event, gpu_samples))
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
    vram_peak_mb = max(x["vram_mb"] for x in gpu_samples) if gpu_samples else 0.0
    gpu_util_avg = mean(x["gpu_util"] for x in gpu_samples) if gpu_samples else 0.0

    return {
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "wall_time_ms": wall_time_ms,
        "ram_peak_mb": ram_peak_mb,
        "ram_avg_mb": ram_avg_mb,
        "vram_peak_mb": vram_peak_mb,
        "gpu_util_avg": gpu_util_avg,
    }


def parse_cpu_output(stdout):
    data = {}
    m = re.search(r"Matrix size = (\d+)", stdout)
    data["size"] = int(m.group(1)) if m else None
    m = re.search(r"BLAS elapsed time \(ms\) = ([\d.]+)", stdout)
    data["blas_ms"] = float(m.group(1)) if m else None
    m = re.search(r"OpenMP elapsed time \(ms\) = ([\d.]+)", stdout)
    data["openmp_ms"] = float(m.group(1)) if m else None
    m = re.search(r"C\+\+ threads elapsed time \(ms\) = ([\d.]+)", stdout)
    data["cpp_threads_ms"] = float(m.group(1)) if m else None
    m = re.search(r"BLAS and OpenMP residual.* = ([\d.eE+\-]+)", stdout)
    data["residual_openmp"] = float(m.group(1)) if m else None
    m = re.search(r"BLAS and C\+\+ threads residual.* = ([\d.eE+\-]+)", stdout)
    data["residual_cpp"] = float(m.group(1)) if m else None
    return data


def parse_gpu_output(stdout):
    data = {}
    m = re.search(r"Matrix size = (\d+)", stdout)
    data["size"] = int(m.group(1)) if m else None
    m = re.search(r"CuBLAS execution elapsed time \(ms\) = ([\d.]+)", stdout)
    data["cublas_ms"] = float(m.group(1)) if m else None
    m = re.search(r"CuBLAS copy elapsed time \(ms\) = ([\d.]+)", stdout)
    data["cublas_copy_ms"] = float(m.group(1)) if m else None
    m = re.search(r"CUDA execution elapsed time \(ms\) = ([\d.]+)", stdout)
    data["cuda_ms"] = float(m.group(1)) if m else None
    m = re.search(r"CUDA copy elapsed time \(ms\) = ([\d.]+)", stdout)
    data["cuda_copy_ms"] = float(m.group(1)) if m else None
    m = re.search(r"CuBLAS and CUDA residual.* = ([\d.eE+\-]+)", stdout)
    data["residual"] = float(m.group(1)) if m else None
    return data


def gflops(n, ms):
    if ms is None or ms <= 0:
        return 0.0
    return (2.0 * n ** 3 - n ** 2) / (ms / 1000.0) / 1e9


def bootstrap_ci(data, n_resamples=10000):
    arr = np.array(data, dtype=float)
    if len(arr) < 2:
        return float(arr[0]), float(arr[0]), float(arr[0])
    means = np.mean(np.random.choice(arr, size=(n_resamples, len(arr)), replace=True), axis=1)
    return float(np.mean(means)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación completa de rendimiento para gemm (davisethan/gemm)"
    )
    parser.add_argument("--repo", default=".",
                        help="Ruta del repositorio gemm")
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[64, 128, 256, 512, 1024],
                        help="Tamaños de matriz N x N")
    parser.add_argument("--reps", type=int, default=30,
                        help="Número de repeticiones por tamaño")
    parser.add_argument("--skip-compile", action="store_true",
                        help="No compilar antes de ejecutar")
    parser.add_argument("--out", default="evaluacion_gemm.csv",
                        help="Archivo CSV de salida")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not repo.exists():
        print(f"Error: no existe el directorio {repo}", file=sys.stderr)
        sys.exit(1)

    sizes = args.sizes
    reps = args.reps
    rows = []

    for size in sizes:
        print(f"\n{'='*60}")
        print(f"  Tamaño de matriz: {size}x{size}")
        print(f"{'='*60}")

        if not args.skip_compile:
            print("  Compilando...", flush=True)
            r = subprocess.run(f"./compile.sh {size}", cwd=repo, shell=True,
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  ERROR de compilación:\n{r.stderr}", file=sys.stderr)
                continue
            print("  Compilación exitosa.")

        gpu_path = repo / "mainCuda"
        if gpu_path.exists():
            print(f"\n  --- GPU (mainCuda) ---")
            gpu_timings = {"cublas": [], "cublas_copy": [], "cuda": [], "cuda_copy": []}
            gpu_residuals = []
            gpu_wall_times = []
            gpu_ram_peaks = []
            gpu_vram_peaks = []
            gpu_utils = []

            for rep in range(1, reps + 1):
                print(f"    Rep {rep:>2}/{reps}", end=" ", flush=True)
                result = run_measured("./mainCuda", repo)
                if result["returncode"] != 0:
                    print("[ERROR]")
                    continue

                parsed = parse_gpu_output(result["stdout"])
                gpu_timings["cublas"].append(parsed.get("cublas_ms"))
                gpu_timings["cublas_copy"].append(parsed.get("cublas_copy_ms"))
                gpu_timings["cuda"].append(parsed.get("cuda_ms"))
                gpu_timings["cuda_copy"].append(parsed.get("cuda_copy_ms"))
                if parsed.get("residual") is not None:
                    gpu_residuals.append(parsed["residual"])
                gpu_wall_times.append(result["wall_time_ms"])
                gpu_ram_peaks.append(result["ram_peak_mb"])
                gpu_vram_peaks.append(result["vram_peak_mb"])
                gpu_utils.append(result["gpu_util_avg"])
                print("OK")

            for alg_key, alg_label in [("cublas", "cuBLAS"), ("cuda", "CUDA")]:
                times = [t for t in gpu_timings[alg_key] if t is not None]
                if not times:
                    continue
                mt = mean(times)
                st = stdev(times) if len(times) > 1 else 0.0
                _, cl, ch = bootstrap_ci(times)
                gf = gflops(size, mt)
                rows.append({
                    "size": size, "programa": "mainCuda", "algoritmo": alg_label,
                    "reps": len(times),
                    "tiempo_mean_ms": round(mt, 4), "tiempo_std_ms": round(st, 4),
                    "tiempo_ci95_low_ms": round(cl, 4), "tiempo_ci95_high_ms": round(ch, 4),
                    "gflops_mean": round(gf, 4),
                    "gflops_ci95_low": round(gflops(size, ch), 4),
                    "gflops_ci95_high": round(gflops(size, cl), 4),
                    "residual_frobenius": round(mean(gpu_residuals), 6) if gpu_residuals else "",
                    "wall_time_mean_ms": round(mean(gpu_wall_times), 2),
                    "ram_peak_mean_mb": round(mean(gpu_ram_peaks), 2),
                    "vram_peak_mean_mb": round(mean(gpu_vram_peaks), 2),
                    "gpu_util_mean_pct": round(mean(gpu_utils), 2),
                })

            for ck, cl in [("cublas_copy", "cuBLAS copy"), ("cuda_copy", "CUDA copy")]:
                times = [t for t in gpu_timings[ck] if t is not None]
                if not times:
                    continue
                rows.append({
                    "size": size, "programa": "mainCuda", "algoritmo": cl,
                    "reps": len(times),
                    "tiempo_mean_ms": round(mean(times), 4),
                    "tiempo_std_ms": round(stdev(times), 4) if len(times) > 1 else 0.0,
                    "gflops_mean": "", "residual_frobenius": "",
                    "wall_time_mean_ms": "", "ram_peak_mean_mb": "",
                    "vram_peak_mean_mb": "", "gpu_util_mean_pct": "",
                })

        cpu_path = repo / "main"
        if cpu_path.exists():
            print(f"\n  --- CPU (main) ---")
            cpu_timings = {"blas": [], "openmp": [], "cpp_threads": []}
            cpu_residuals = {"openmp": [], "cpp": []}
            cpu_wall_times = []
            cpu_ram_peaks = []

            for rep in range(1, reps + 1):
                print(f"    Rep {rep:>2}/{reps}", end=" ", flush=True)
                result = run_measured("./main", repo)
                if result["returncode"] != 0:
                    print("[ERROR]")
                    continue

                parsed = parse_cpu_output(result["stdout"])
                cpu_timings["blas"].append(parsed.get("blas_ms"))
                cpu_timings["openmp"].append(parsed.get("openmp_ms"))
                cpu_timings["cpp_threads"].append(parsed.get("cpp_threads_ms"))
                if parsed.get("residual_openmp") is not None:
                    cpu_residuals["openmp"].append(parsed["residual_openmp"])
                if parsed.get("residual_cpp") is not None:
                    cpu_residuals["cpp"].append(parsed["residual_cpp"])
                cpu_wall_times.append(result["wall_time_ms"])
                cpu_ram_peaks.append(result["ram_peak_mb"])
                print("OK")

            alg_map = [
                ("blas", "BLAS (OpenBLAS)", None),
                ("openmp", "OpenMP", "openmp"),
                ("cpp_threads", "C++ Threads", "cpp"),
            ]
            for ak, al, rk in alg_map:
                times = [t for t in cpu_timings[ak] if t is not None]
                if not times:
                    continue
                mt = mean(times)
                st = stdev(times) if len(times) > 1 else 0.0
                _, cl, ch = bootstrap_ci(times)
                gf = gflops(size, mt)
                residual = mean(cpu_residuals[rk]) if rk and cpu_residuals.get(rk) else ""
                rows.append({
                    "size": size, "programa": "main", "algoritmo": al,
                    "reps": len(times),
                    "tiempo_mean_ms": round(mt, 4), "tiempo_std_ms": round(st, 4),
                    "tiempo_ci95_low_ms": round(cl, 4), "tiempo_ci95_high_ms": round(ch, 4),
                    "gflops_mean": round(gf, 4),
                    "gflops_ci95_low": round(gflops(size, ch), 4),
                    "gflops_ci95_high": round(gflops(size, cl), 4),
                    "residual_frobenius": round(residual, 6) if residual != "" else "",
                    "wall_time_mean_ms": round(mean(cpu_wall_times), 2),
                    "ram_peak_mean_mb": round(mean(cpu_ram_peaks), 2),
                    "vram_peak_mean_mb": "",
                    "gpu_util_mean_pct": "",
                })

    fieldnames = [
        "size", "programa", "algoritmo", "reps",
        "tiempo_mean_ms", "tiempo_std_ms",
        "tiempo_ci95_low_ms", "tiempo_ci95_high_ms",
        "gflops_mean", "gflops_ci95_low", "gflops_ci95_high",
        "residual_frobenius",
        "wall_time_mean_ms", "ram_peak_mean_mb",
        "vram_peak_mean_mb", "gpu_util_mean_pct",
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"  Resultados guardados en: {args.out}")
    print(f"{'='*60}")

    print(f"\n{'='*110}")
    print(f"  RESUMEN DE RENDIMIENTO")
    print(f"{'='*110}")
    header = f"{'N':>6}  {'Programa':<10} {'Algoritmo':<18} {'Tiempo(ms)':>11} {'GFLOPS':>10} {'RAM(MB)':>9} {'VRAM(MB)':>9} {'GPU(%)':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.get("gflops_mean") not in (None, ""):
            print(
                f"{r['size']:>6}  {r['programa']:<10} {r['algoritmo']:<18} "
                f"{r['tiempo_mean_ms']:>10.2f}  {r['gflops_mean']:>8.2f}  "
                f"{str(r.get('ram_peak_mean_mb', '')):>7}  "
                f"{str(r.get('vram_peak_mean_mb', '')):>7}  "
                f"{str(r.get('gpu_util_mean_pct', '')):>6}"
            )


if __name__ == "__main__":
    main()
