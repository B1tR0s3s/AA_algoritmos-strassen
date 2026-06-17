import csv
import subprocess
import time
import threading
from pathlib import Path
from statistics import mean, stdev
import psutil

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

def memory_monitor_gpu(stop_event, samples):
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            line = result.stdout.strip().split("\n")[0]
            if line:
                mem, util = line.split(",")
                samples.append({"gpu_memory_mb": float(mem.strip()), "gpu_util": float(util.strip())})
        except Exception:
            pass
        time.sleep(0.1)

def run_measured(command, cwd="."):
    ram_samples = []
    gpu_samples = []
    stop_event = threading.Event()
    start = time.perf_counter()
    proc = subprocess.Popen(command, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ram_thread = threading.Thread(target=memory_monitor_process, args=(proc, stop_event, ram_samples))
    gpu_thread = threading.Thread(target=memory_monitor_gpu, args=(stop_event, gpu_samples))
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
        "returncode": proc.returncode, "stdout": stdout, "stderr": stderr,
        "wall_time_ms": wall_time_ms, "ram_peak_mb": ram_peak_mb, "ram_avg_mb": ram_avg_mb,
        "gpu_mem_peak_mb": gpu_mem_peak_mb, "gpu_util_avg": gpu_util_avg
    }

repo = Path(".").resolve()
gen_path = repo / "gen"
multiply_path = repo / "multiply"
mat_dir = repo / "matrices_generadas"
mat_dir.mkdir(exist_ok=True)

# Shapes: (m, k, n)  →  A[m×k] × B[k×n] = C[m×n]
shapes = [
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    # Rectangulares
    (256, 512, 256),
    (512, 256, 1024),
    (1024, 512, 512),
    (512, 1024, 2048),
    (2048, 1024, 512),
]

reps = 3
rows = []

for m, k, n in shapes:
    print(f"\n{'='*60}")
    print(f"Tamaño: A[{m}×{k}] × B[{k}×{n}] → C[{m}×{n}]")
    print(f"{'='*60}")

    a_file = mat_dir / f"A_{m}x{k}.mtx"
    b_file = mat_dir / f"B_{k}x{n}.mtx"
    c_file = mat_dir / f"C_{m}x{n}.mtx"

    if not a_file.exists():
        subprocess.run([str(gen_path), "-r", str(m), "-c", str(k), str(a_file)], check=True)
    if not b_file.exists():
        subprocess.run([str(gen_path), "-r", str(k), "-c", str(n), str(b_file)], check=True)

    tiempos = []
    ram_peaks = []
    gpu_peaks = []
    gpu_utils = []

    for rep in range(1, reps + 1):
        command = f"./multiply {a_file} {b_file} {c_file}"
        print(f"\n  Repetición {rep}/{reps}")
        print(f"  Comando: {command}")

        result = run_measured(command, repo)

        tiempos.append(result["wall_time_ms"])
        ram_peaks.append(result["ram_peak_mb"])
        gpu_peaks.append(result["gpu_mem_peak_mb"])
        gpu_utils.append(result["gpu_util_avg"])

        print(f"    Tiempo: {result['wall_time_ms']:.3f} ms | "
              f"RAM pico: {result['ram_peak_mb']:.2f} MB | "
              f"VRAM pico: {result['gpu_mem_peak_mb']:.2f} MB | "
              f"GPU uso: {result['gpu_util_avg']:.2f}%")

        if result["returncode"] != 0:
            print("    ERROR:")
            print(result["stderr"][:500])

    avg_time = mean(tiempos)
    ops = 2 * m * n * k
    gflops = ops / (avg_time / 1000) / 1e9 if avg_time > 0 else 0

    row = {
        "m": m, "k": k, "n": n,
        "tiempo_prom_ms": round(avg_time, 3),
        "tiempo_std_ms": round(stdev(tiempos) if len(tiempos) > 1 else 0.0, 3),
        "ram_pico_prom_mb": round(mean(ram_peaks), 2),
        "vram_pico_prom_mb": round(mean(gpu_peaks), 2),
        "gpu_util_prom": round(mean(gpu_utils), 2),
        "gflops": round(gflops, 2)
    }
    rows.append(row)

print(f"\n\n{'='*80}")
print("RESUMEN FINAL")
print(f"{'='*80}")
print(f"{'m':>5} {'k':>5} {'n':>5} | {'Tiempo(ms)':>10} | {'Std(ms)':>8} | {'RAM(MB)':>8} | {'VRAM(MB)':>9} | {'GPU%':>6} | {'GFLOPS':>8}")
print("-" * 80)
for r in rows:
    print(f"{r['m']:>5} {r['k']:>5} {r['n']:>5} | {r['tiempo_prom_ms']:>8.3f}  | {r['tiempo_std_ms']:>6.3f}  | {r['ram_pico_prom_mb']:>6.2f} | {r['vram_pico_prom_mb']:>7.2f} | {r['gpu_util_prom']:>4.2f} | {r['gflops']:>6.2f}")

csv_file = repo / "resultados.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["m", "k", "n", "tiempo_prom_ms", "tiempo_std_ms",
                   "ram_pico_prom_mb", "vram_pico_prom_mb",
                   "gpu_util_prom", "gflops"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"\nResultados guardados en: {csv_file}")
