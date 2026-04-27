"""
Benchmark skripta za TFLite modele na RPi-u.
Mjeri: FPS, latenciju, RAM, temperaturu CPU-a.
"""

from ai_edge_litert.interpreter import Interpreter
import numpy as np
import time
import psutil
import json
from pathlib import Path
from datetime import datetime

# Putanje
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_TFLITE = BASE_DIR / "models" / "tflite"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Postavke benchmarka
NUM_WARMUP = 10
NUM_ITERATIONS = 100
IMAGE_SIZE = 320  # TFLite modeli su exportani s 320


def get_cpu_temperature() -> float:
    try:
        temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if temp_path.exists():
            return float(temp_path.read_text().strip()) / 1000.0
    except Exception:
        pass
    return -1.0


def get_ram_usage_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def benchmark_model(model_path: Path) -> dict:
    print(f"\n{'='*50}")
    print(f"Model: {model_path.name}")
    print(f"{'='*50}")

    # Učitaj model
    print("Učitavam model...")
    ram_before = get_ram_usage_mb()

    interpreter = Interpreter(
        model_path=str(model_path),
        num_threads=4  # RPi 5 ima 4 jezgre
    )
    interpreter.allocate_tensors()

    ram_after_load = get_ram_usage_mb()
    ram_model_mb = ram_after_load - ram_before

    # Dohvati input/output detalje
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    print(f"Input shape: {input_shape}")
    print(f"Input dtype: {input_dtype}")
    print(f"RAM za učitavanje modela: {ram_model_mb:.1f} MB")

    # Pripremi dummy input
    dummy_input = np.random.rand(*input_shape).astype(input_dtype)

    # Warmup
    print(f"Warmup ({NUM_WARMUP} iteracija)...")
    for _ in range(NUM_WARMUP):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()

    # Benchmark
    print(f"Benchmark ({NUM_ITERATIONS} iteracija)...")
    temp_before = get_cpu_temperature()
    latencies = []

    for i in range(NUM_ITERATIONS):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{NUM_ITERATIONS} – trenutni FPS: {1000/latencies[-1]:.1f}")

    temp_after = get_cpu_temperature()
    ram_peak = get_ram_usage_mb()

    # Statistike
    latencies = np.array(latencies)
    avg_latency = float(np.mean(latencies))
    avg_fps = 1000.0 / avg_latency

    results = {
        "model": model_path.name,
        "format": "tflite",
        "image_size": IMAGE_SIZE,
        "num_iterations": NUM_ITERATIONS,
        "avg_latency_ms": round(avg_latency, 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "avg_fps": round(avg_fps, 2),
        "ram_model_mb": round(ram_model_mb, 1),
        "ram_peak_mb": round(ram_peak, 1),
        "temp_before_c": temp_before,
        "temp_after_c": temp_after,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n📊 Rezultati:")
    print(f"  Avg FPS:      {results['avg_fps']:.1f}")
    print(f"  Avg latency:  {results['avg_latency_ms']:.1f} ms")
    print(f"  P95 latency:  {results['p95_latency_ms']:.1f} ms")
    print(f"  RAM (model):  {results['ram_model_mb']:.1f} MB")
    print(f"  Temp prije:   {results['temp_before_c']:.1f}°C")
    print(f"  Temp poslije: {results['temp_after_c']:.1f}°C")

    return results


def main():
    print("🚀 TFLite Benchmark – RPi 5")
    print(f"Broj jezgri: {psutil.cpu_count()}")
    print(f"Ukupni RAM: {psutil.virtual_memory().total / 1024 / 1024:.0f} MB")

    models = sorted(MODELS_TFLITE.glob("*.tflite"))
    if not models:
        print(f"❌ Nema TFLite modela u {MODELS_TFLITE}")
        return

    print(f"\nPronađeni modeli: {[m.name for m in models]}")

    all_results = []
    for model_path in models:
        result = benchmark_model(model_path)
        all_results.append(result)

    # Spremi rezultate
    output_file = RESULTS_DIR / f"tflite_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Rezultati spremljeni: {output_file}")

    # Usporedna tablica
    print(f"\n{'='*60}")
    print(f"{'Model':<35} {'FPS':>8} {'Latency':>10} {'RAM':>8}")
    print(f"{'='*60}")
    for r in all_results:
        print(f"{r['model']:<35} {r['avg_fps']:>8.1f} {r['avg_latency_ms']:>9.1f}ms {r['ram_model_mb']:>7.1f}MB")


if __name__ == "__main__":
    main()