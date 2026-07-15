"""
mAP + FPS + RAM + Temp evaluacija na RPi 5.

Korištenje:
    python3 evaluate_map_rpi.py --type pretrained
    python3 evaluate_map_rpi.py --type finetuned
"""

import argparse
import json
import time
import numpy as np
import psutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ─── Putanje ─────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "benchmark" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DIR = (
    Path(__file__).parent.parent.parent.parent / "datasets" / "Rpi5-ai-vision.yolov8"
)
TEST_IMAGES = DATASET_DIR / "test" / "images"
TEST_LABELS = DATASET_DIR / "test" / "labels"

ALL_MODELS = [
    # TFLite FP32
    {"filename": "yolov8n_fp32.tflite", "format": "tflite", "quantization": "fp32"},
    {"filename": "yolov8s_fp32.tflite", "format": "tflite", "quantization": "fp32"},
    {"filename": "yolov10n_fp32.tflite", "format": "tflite", "quantization": "fp32"},
    {"filename": "yolo11n_fp32.tflite", "format": "tflite", "quantization": "fp32"},
    # TFLite INT8
    {"filename": "yolov8n_int8.tflite", "format": "tflite", "quantization": "int8"},
    {"filename": "yolov8s_int8.tflite", "format": "tflite", "quantization": "int8"},
    {"filename": "yolov10n_int8.tflite", "format": "tflite", "quantization": "int8"},
    {"filename": "yolo11n_int8.tflite", "format": "tflite", "quantization": "int8"},
    # ONNX FP32
    {"filename": "yolov8n_fp32.onnx", "format": "onnx", "quantization": "fp32"},
    {"filename": "yolov8s_fp32.onnx", "format": "onnx", "quantization": "fp32"},
    {"filename": "yolov10n_fp32.onnx", "format": "onnx", "quantization": "fp32"},
    {"filename": "yolo11n_fp32.onnx", "format": "onnx", "quantization": "fp32"},
    {"filename": "rtdetr-l_fp32.onnx", "format": "onnx", "quantization": "fp32"},
    # ONNX INT8
    {"filename": "yolov8n_int8.onnx", "format": "onnx", "quantization": "int8"},
    {"filename": "yolov8s_int8.onnx", "format": "onnx", "quantization": "int8"},
    {"filename": "yolov10n_int8.onnx", "format": "onnx", "quantization": "int8"},
    {"filename": "yolo11n_int8.onnx", "format": "onnx", "quantization": "int8"},
    {"filename": "rtdetr-l_int8.onnx", "format": "onnx", "quantization": "int8"},
]

IMAGE_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
NUM_WARMUP = 5


# ─── RPi metrike ─────────────────────────────────────────────────────────────


def get_temperature() -> float:
    try:
        temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if temp_path.exists():
            return float(temp_path.read_text().strip()) / 1000.0
    except Exception:
        pass
    return -1.0


def get_ram_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


# ─── Dataset ─────────────────────────────────────────────────────────────────


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list:
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append({"class": cls, "bbox": [x1, y1, x2, y2]})
    return boxes


def load_test_set() -> list:
    images = sorted(list(TEST_IMAGES.glob("*.jpg")) + list(TEST_IMAGES.glob("*.png")))
    dataset = []
    for img_path in images:
        lbl_path = TEST_LABELS / (img_path.stem + ".txt")
        dataset.append({"image_path": img_path, "label_path": lbl_path})
    print(f"✅ Test set: {len(dataset)} slika")
    return dataset


# ─── Preprocess ──────────────────────────────────────────────────────────────


def preprocess_image(img_path: Path) -> tuple:
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    arr = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255.0
    return arr, orig_w, orig_h


# ─── NMS ─────────────────────────────────────────────────────────────────────


def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


# ─── Postprocess ─────────────────────────────────────────────────────────────


def postprocess_yolov8(output, orig_w, orig_h):
    """
    Output shape: (1, 84, 8400) → transponiramo u (8400, 84)
    Stupci: [cx, cy, w, h, cls0, cls1, ..., cls79]
    cls0 = person (COCO)
    """
    pred = output[0]  # (84, 8400)

    # Ako je (84, 8400) transponiramo
    if pred.shape[0] == 84:
        pred = pred.T  # (8400, 84)

    # person score je stupac 4 (index 4)
    person_scores = pred[:, 4]
    mask = person_scores > CONF_THRESHOLD
    pred = pred[mask]

    if len(pred) == 0:
        return []

    boxes = pred[:, :4]  # cx, cy, w, h (normalizirano 0-IMAGE_SIZE)
    scores = pred[:, 4]

    keep = nms(boxes, scores, IOU_THRESHOLD)
    results = []
    for i in keep:
        cx, cy, w, h = boxes[i]
        x1 = (cx - w / 2) / IMAGE_SIZE * orig_w
        y1 = (cy - h / 2) / IMAGE_SIZE * orig_h
        x2 = (cx + w / 2) / IMAGE_SIZE * orig_w
        y2 = (cy + h / 2) / IMAGE_SIZE * orig_h
        results.append(
            {"class": 0, "bbox": [x1, y1, x2, y2], "score": float(scores[i])}
        )
    return results


def postprocess_rtdetr(output, orig_w, orig_h):
    """RT-DETR: output shape (1, 300, 6) → [x1,y1,x2,y2,score,class]"""
    preds = output[0][0]  # (300, 6)
    results = []
    for pred in preds:
        x1, y1, x2, y2, score, cls = pred
        if score < CONF_THRESHOLD or int(cls) != 0:
            continue
        results.append(
            {
                "class": 0,
                "bbox": [
                    float(x1) * orig_w,
                    float(y1) * orig_h,
                    float(x2) * orig_w,
                    float(y2) * orig_h,
                ],
                "score": float(score),
            }
        )
    return results


# ─── Inference ───────────────────────────────────────────────────────────────


def run_tflite_inference(
    interpreter, input_details, output_details, img_arr, orig_w, orig_h
):
    input_dtype = input_details[0]["dtype"]
    inp = np.expand_dims(img_arr, 0)  # BHWC (1, 640, 640, 3)

    if input_dtype == np.int8:
        scale, zero_point = input_details[0]["quantization"]
        if scale != 0:
            inp = (inp / scale + zero_point).astype(np.int8)

    t0 = time.perf_counter()
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if input_dtype == np.int8:
        out_scale, out_zp = output_details[0]["quantization"]
        if out_scale != 0:
            output = (output.astype(np.float32) - out_zp) * out_scale

    return postprocess_yolov8(output, orig_w, orig_h), elapsed_ms


def run_onnx_inference(session, img_arr, orig_w, orig_h, is_rtdetr=False):
    input_name = session.get_inputs()[0].name
    inp = np.expand_dims(img_arr.transpose(2, 0, 1), 0)  # BCHW

    t0 = time.perf_counter()
    output = session.run(None, {input_name: inp})
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if is_rtdetr:
        return postprocess_rtdetr(output, orig_w, orig_h), elapsed_ms
    return postprocess_yolov8(output[0], orig_w, orig_h), elapsed_ms


# ─── mAP ─────────────────────────────────────────────────────────────────────


def compute_iou(box1, box2):
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-10)


def compute_map(all_detections, all_ground_truths, iou_thresholds):
    aps = []
    for iou_thresh in iou_thresholds:
        tp_list, fp_list, scores_list = [], [], []
        n_gt = sum(len(gt) for gt in all_ground_truths)

        for dets, gts in zip(all_detections, all_ground_truths):
            matched = [False] * len(gts)
            for det in sorted(dets, key=lambda x: x["score"], reverse=True):
                scores_list.append(det["score"])
                best_iou, best_j = 0, -1
                for j, gt in enumerate(gts):
                    if matched[j]:
                        continue
                    iou = compute_iou(det["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= iou_thresh and best_j >= 0:
                    tp_list.append(1)
                    fp_list.append(0)
                    matched[best_j] = True
                else:
                    tp_list.append(0)
                    fp_list.append(1)

        if not scores_list:
            aps.append(0.0)
            continue

        idx = np.argsort(scores_list)[::-1]
        tp_cum = np.cumsum(np.array(tp_list)[idx])
        fp_cum = np.cumsum(np.array(fp_list)[idx])
        precision = tp_cum / (tp_cum + fp_cum + 1e-10)
        recall = tp_cum / (n_gt + 1e-10)

        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = precision[recall >= t]
            ap += np.max(p) if len(p) > 0 else 0.0
        aps.append(ap / 11)

    return float(aps[0]) if aps else 0.0, float(np.mean(aps)) if aps else 0.0


# ─── Evaluacija modela ───────────────────────────────────────────────────────


def evaluate_model(model_info: dict, dataset: list, model_type: str) -> dict:
    filename = model_info["filename"]
    fmt = model_info["format"]
    quant = model_info["quantization"]
    is_rtdetr = "rtdetr" in filename.lower()

    model_path = MODELS_DIR / fmt / quant / filename
    if not model_path.exists():
        print(f"⚠️  Model ne postoji: {model_path}")
        return None

    print(f"\n{'='*60}")
    print(f"Evaluiram: {filename} [{model_type}]")
    print(f"{'='*60}")

    # Učitaj model
    ram_before = get_ram_mb()
    if fmt == "tflite":
        from ai_edge_litert.interpreter import Interpreter

        interpreter = Interpreter(model_path=str(model_path), num_threads=4)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    else:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

    ram_after_load = get_ram_mb()
    ram_model_mb = ram_after_load - ram_before

    # Warmup
    print(f"Warmup ({NUM_WARMUP} iteracija)...")
    dummy_arr = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
    for _ in range(NUM_WARMUP):
        if fmt == "tflite":
            run_tflite_inference(
                interpreter,
                input_details,
                output_details,
                dummy_arr,
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        else:
            run_onnx_inference(session, dummy_arr, IMAGE_SIZE, IMAGE_SIZE, is_rtdetr)

    # Inference na test setu
    all_detections = []
    all_ground_truths = []
    inference_times = []
    temp_before = get_temperature()

    for item in tqdm(dataset, desc=filename):
        img_arr, orig_w, orig_h = preprocess_image(item["image_path"])
        gt_boxes = load_yolo_labels(item["label_path"], orig_w, orig_h)

        if fmt == "tflite":
            dets, ms = run_tflite_inference(
                interpreter, input_details, output_details, img_arr, orig_w, orig_h
            )
        else:
            dets, ms = run_onnx_inference(session, img_arr, orig_w, orig_h, is_rtdetr)

        all_detections.append(dets)
        all_ground_truths.append(gt_boxes)
        inference_times.append(ms)

    temp_after = get_temperature()
    ram_peak = get_ram_mb()
    avg_ms = float(np.mean(inference_times))

    # mAP
    map_50, map_5095 = compute_map(all_detections, all_ground_truths, [0.5])
    _, map_5095 = compute_map(
        all_detections, all_ground_truths, np.arange(0.5, 1.0, 0.05).tolist()
    )

    # Precision i Recall
    all_tp, all_fp = [], []
    n_gt_total = sum(len(gt) for gt in all_ground_truths)
    for dets, gts in zip(all_detections, all_ground_truths):
        matched = [False] * len(gts)
        for det in sorted(dets, key=lambda x: x["score"], reverse=True):
            best_iou, best_j = 0, -1
            for j, gt in enumerate(gts):
                if matched[j]:
                    continue
                iou = compute_iou(det["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.5 and best_j >= 0:
                all_tp.append(1)
                all_fp.append(0)
                matched[best_j] = True
            else:
                all_tp.append(0)
                all_fp.append(1)

    tp_sum = sum(all_tp)
    fp_sum = sum(all_fp)
    avg_precision = tp_sum / (tp_sum + fp_sum + 1e-10)
    avg_recall = tp_sum / (n_gt_total + 1e-10)

    result = {
        "model": filename,
        "format": fmt,
        "quantization": quant,
        "type": model_type,
        "image_size": IMAGE_SIZE,
        "num_images": len(dataset),
        "mAP_0.5": round(map_50, 4),
        "mAP_0.5_0.95": round(map_5095, 4),
        "avg_precision": round(float(avg_precision), 4),
        "avg_recall": round(float(avg_recall), 4),
        "avg_inference_ms": round(avg_ms, 2),
        "avg_fps": round(1000.0 / avg_ms, 2),
        "ram_model_mb": round(ram_model_mb, 1),
        "ram_peak_mb": round(ram_peak, 1),
        "temp_before_c": round(temp_before, 1),
        "temp_after_c": round(temp_after, 1),
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n📊 Rezultati:")
    print(f"  mAP@0.5:      {result['mAP_0.5']:.4f}")
    print(f"  mAP@0.5:0.95: {result['mAP_0.5_0.95']:.4f}")
    print(f"  Precision:    {result['avg_precision']:.4f}")
    print(f"  Recall:       {result['avg_recall']:.4f}")
    print(f"  Avg FPS:      {result['avg_fps']:.1f}")
    print(f"  RAM model:    {result['ram_model_mb']:.1f} MB")
    print(
        f"  Temp:         {result['temp_before_c']:.1f}°C → {result['temp_after_c']:.1f}°C"
    )

    return result


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type", default="pretrained", choices=["pretrained", "finetuned"]
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"🚀 Evaluacija na RPi 5 – {args.type.upper()}")
    print(f"Test set: {TEST_IMAGES}")
    print(f"RAM ukupno: {psutil.virtual_memory().total / 1024 / 1024:.0f} MB")

    dataset = load_test_set()

    models_to_eval = ALL_MODELS
    if args.models:
        models_to_eval = [m for m in ALL_MODELS if m["filename"] in args.models]

    output_file = (
        Path(args.output) if args.output else RESULTS_DIR / "benchmark_results.json"
    )

    existing_results = []
    if output_file.exists():
        with open(output_file) as f:
            existing_results = json.load(f)
        print(f"📂 Učitano {len(existing_results)} postojećih rezultata")

    new_results = []
    for model_info in models_to_eval:
        result = evaluate_model(model_info, dataset, args.type)
        if result:
            new_results.append(result)
            all_results = existing_results + new_results
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"💾 Spremljeno: {output_file}")

    print(f"\n✅ Gotovo! Evaluirano: {len(new_results)} modela")
    print(f"\n{'='*80}")
    print(f"{'Model':<35} {'Type':>10} {'mAP@0.5':>8} {'FPS':>7} {'RAM':>8}")
    print(f"{'='*80}")
    for r in new_results:
        print(
            f"{r['model']:<35} {r['type']:>10} "
            f"{r['mAP_0.5']:>8.4f} {r['avg_fps']:>7.1f} "
            f"{r['ram_model_mb']:>7.1f}MB"
        )


if __name__ == "__main__":
    main()
