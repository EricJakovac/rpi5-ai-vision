"""
Inference s kamerom u realnom vremenu.
Mjeri realne FPS-ove na pravim slikama.
Pokreći na RPi-u.
"""

import onnxruntime as ort
import numpy as np
import time
import psutil
import json
from picamera2 import Picamera2
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
MODELS_ONNX = BASE_DIR / "models" / "onnx"
RESULTS_DIR = BASE_DIR / "benchmark" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_WARMUP = 10
NUM_ITERATIONS = 100
IMAGE_SIZE = 640
CONF_THRESHOLD = 0.5


def get_cpu_temperature() -> float:
    try:
        temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if temp_path.exists():
            return float(temp_path.read_text().strip()) / 1000.0
    except Exception:
        pass
    return -1.0


def preprocess_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """Pripremi frame za inference – resize, normalize, transpose."""
    # Resize
    from PIL import Image
    img = Image.fromarray(frame)
    img = img.resize((size, size))
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize 0-255 → 0-1
    img_array /= 255.0
    
    # HWC → CHW → BCHW
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def postprocess_output(output: np.ndarray, conf_threshold: float) -> list:
    """Izvuci detekcije osoba iz YOLO outputa."""
    predictions = output[0]  # (1, 84, 8400) → (84, 8400)
    if len(predictions.shape) == 3:
        predictions = predictions[0]
    
    predictions = predictions.T  # (8400, 84)
    
    detections = []
    for pred in predictions:
        scores = pred[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        # Class 0 = person u COCO
        if class_id == 0 and confidence > conf_threshold:
            cx, cy, w, h = pred[:4]
            detections.append({
                "confidence": float(confidence),
                "bbox": [float(cx), float(cy), float(w), float(h)]
            })
    
    return detections


def benchmark_with_camera(model_path: Path, picam2: Picamera2) -> dict:
    print(f"\n{'='*50}")
    print(f"Model: {model_path.name}")
    print(f"{'='*50}")

    ram_before = get_ram_usage_mb() if hasattr(psutil.Process(), 'memory_info') else 0

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4

    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    ram_after = get_ram_usage_mb() if hasattr(psutil.Process(), 'memory_info') else 0

    # Warmup
    print(f"Warmup ({NUM_WARMUP} iteracija)...")
    for _ in range(NUM_WARMUP):
        frame = picam2.capture_array()
        inp = preprocess_frame(frame, IMAGE_SIZE)
        session.run(None, {input_name: inp})

    # Benchmark
    print(f"Benchmark ({NUM_ITERATIONS} iteracija)...")
    temp_before = get_cpu_temperature()
    
    latencies_capture = []
    latencies_preprocess = []
    latencies_inference = []
    total_persons_detected = 0

    for i in range(NUM_ITERATIONS):
        # Capture
        t0 = time.perf_counter()
        frame = picam2.capture_array()
        t1 = time.perf_counter()

        # Preprocess
        inp = preprocess_frame(frame, IMAGE_SIZE)
        t2 = time.perf_counter()

        # Inference
        output = session.run(None, {input_name: inp})
        t3 = time.perf_counter()

        latencies_capture.append((t1 - t0) * 1000)
        latencies_preprocess.append((t2 - t1) * 1000)
        latencies_inference.append((t3 - t2) * 1000)

        # Detekcije
        detections = postprocess_output(output[0], CONF_THRESHOLD)
        total_persons_detected += len(detections)

        if (i + 1) % 25 == 0:
            avg_inf = np.mean(latencies_inference[-25:])
            print(f"  {i+1}/{NUM_ITERATIONS} – inference FPS: {1000/avg_inf:.1f}")

    temp_after = get_cpu_temperature()

    # Ukupna latencija = capture + preprocess + inference
    total_latencies = [
        latencies_capture[i] + latencies_preprocess[i] + latencies_inference[i]
        for i in range(NUM_ITERATIONS)
    ]

    results = {
        "model": model_path.name,
        "format": "onnx",
        "image_size": IMAGE_SIZE,
        "num_iterations": NUM_ITERATIONS,
        "source": "camera",
        # Inference samo
        "avg_inference_ms": round(float(np.mean(latencies_inference)), 2),
        "avg_inference_fps": round(1000 / float(np.mean(latencies_inference)), 2),
        # Ukupno (capture + preprocess + inference)
        "avg_total_ms": round(float(np.mean(total_latencies)), 2),
        "avg_total_fps": round(1000 / float(np.mean(total_latencies)), 2),
        "p95_total_ms": round(float(np.percentile(total_latencies, 95)), 2),
        # Ostalo
        "avg_capture_ms": round(float(np.mean(latencies_capture)), 2),
        "avg_preprocess_ms": round(float(np.mean(latencies_preprocess)), 2),
        "ram_model_mb": round(ram_after - ram_before, 1),
        "avg_persons_detected": round(total_persons_detected / NUM_ITERATIONS, 2),
        "temp_before_c": temp_before,
        "temp_after_c": temp_after,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n📊 Rezultati:")
    print(f"  Inference FPS:  {results['avg_inference_fps']:.1f}")
    print(f"  Ukupni FPS:     {results['avg_total_fps']:.1f}")
    print(f"  Capture:        {results['avg_capture_ms']:.1f}ms")
    print(f"  Preprocess:     {results['avg_preprocess_ms']:.1f}ms")
    print(f"  Inference:      {results['avg_inference_ms']:.1f}ms")
    print(f"  Avg osoba:      {results['avg_persons_detected']:.1f}")

    return results


def get_ram_usage_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def main():
    print("🚀 Camera Inference Benchmark – RPi 5")
    print(f"ORT verzija: {ort.__version__}")

    # Inicijaliziraj kameru
    print("\nPokrećem kameru...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (IMAGE_SIZE, IMAGE_SIZE), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # AWB warmup
    print("✅ Kamera pokrenuta")

    models = sorted(MODELS_ONNX.glob("*.onnx"))
    if not models:
        print(f"❌ Nema ONNX modela u {MODELS_ONNX}")
        picam2.stop()
        return

    print(f"Modeli: {[m.name for m in models]}")

    all_results = []
    for model_path in models:
        result = benchmark_with_camera(model_path, picam2)
        all_results.append(result)

    picam2.stop()

    # Spremi rezultate
    output_file = RESULTS_DIR / f"camera_onnx_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Rezultati spremljeni: {output_file}")

    # Usporedna tablica
    print(f"\n{'='*65}")
    print(f"{'Model':<30} {'Inf.FPS':>8} {'Tot.FPS':>8} {'Osoba':>6}")
    print(f"{'='*65}")
    for r in all_results:
        print(f"{r['model']:<30} {r['avg_inference_fps']:>8.1f} {r['avg_total_fps']:>8.1f} {r['avg_persons_detected']:>6.1f}")


if __name__ == "__main__":
    main()