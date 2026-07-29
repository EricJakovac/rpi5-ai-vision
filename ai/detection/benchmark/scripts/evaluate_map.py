"""
mAP + FPS + RAM + Temp evaluacija na RPi 5.
Korištenje:
    python3 evaluate_map.py --type pretrained
    python3 evaluate_map.py --type finetuned
    python3 evaluate_map.py --type pretrained --models yolov8n_fp32.tflite yolov8n_fp32.onnx
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

BASE_DIR = Path.home() / "Desktop" / "rpi5-ai-vision"
MODELS_DIR = BASE_DIR / "ai" / "detection" / "models"
RESULTS_DIR = BASE_DIR / "ai" / "detection" / "benchmark" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DIR = BASE_DIR / "datasets" / "Rpi5-ai-vision.yolov8"
TEST_IMAGES = DATASET_DIR / "test" / "images"
TEST_LABELS = DATASET_DIR / "test" / "labels"

ALL_MODELS = [
    # ─── PRETRAINED ───────────────────────────────────────────────
    {"filename": "yolov8n_fp32.tflite",  "format": "tflite", "quantization": "fp32", "arch": "yolov8"},
    {"filename": "yolov8s_fp32.tflite",  "format": "tflite", "quantization": "fp32", "arch": "yolov8"},
    {"filename": "yolov10n_fp32.tflite", "format": "tflite", "quantization": "fp32", "arch": "yolov10"},
    {"filename": "yolo11n_fp32.tflite",  "format": "tflite", "quantization": "fp32", "arch": "yolo11"},
    {"filename": "yolov8n_int8.tflite",  "format": "tflite", "quantization": "int8", "arch": "yolov8"},
    {"filename": "yolov8s_int8.tflite",  "format": "tflite", "quantization": "int8", "arch": "yolov8"},
    {"filename": "yolov10n_int8.tflite", "format": "tflite", "quantization": "int8", "arch": "yolov10"},
    {"filename": "yolo11n_int8.tflite",  "format": "tflite", "quantization": "int8", "arch": "yolo11"},
    {"filename": "yolov8n_fp32.onnx",    "format": "onnx",   "quantization": "fp32", "arch": "yolov8"},
    {"filename": "yolov8s_fp32.onnx",    "format": "onnx",   "quantization": "fp32", "arch": "yolov8"},
    {"filename": "yolov10n_fp32.onnx",   "format": "onnx",   "quantization": "fp32", "arch": "yolov10"},
    {"filename": "yolo11n_fp32.onnx",    "format": "onnx",   "quantization": "fp32", "arch": "yolo11"},
    {"filename": "rtdetr-l_fp32.onnx",   "format": "onnx",   "quantization": "fp32", "arch": "rtdetr"},
    {"filename": "yolov8n_int8.onnx",    "format": "onnx",   "quantization": "int8", "arch": "yolov8"},
    {"filename": "yolov8s_int8.onnx",    "format": "onnx",   "quantization": "int8", "arch": "yolov8"},
    {"filename": "yolov10n_int8.onnx",   "format": "onnx",   "quantization": "int8", "arch": "yolov10"},
    {"filename": "yolo11n_int8.onnx",    "format": "onnx",   "quantization": "int8", "arch": "yolo11"},
    # ─── FINE-TUNED ───────────────────────────────────────────────
    {"filename": "yolov8n_ft_fp32.tflite",  "format": "tflite", "quantization": "fp32", "arch": "yolov8"},
    {"filename": "yolov8s_ft_fp32.tflite",  "format": "tflite", "quantization": "fp32", "arch": "yolov8"},
    {"filename": "yolov10n_ft_fp32.tflite", "format": "tflite", "quantization": "fp32", "arch": "yolov10"},
    {"filename": "yolo11n_ft_fp32.tflite",  "format": "tflite", "quantization": "fp32", "arch": "yolo11"},
    {"filename": "yolov8n_ft_int8.tflite",  "format": "tflite", "quantization": "int8", "arch": "yolov8"},
    {"filename": "yolov8s_ft_int8.tflite",  "format": "tflite", "quantization": "int8", "arch": "yolov8"},
    {"filename": "yolov10n_ft_int8.tflite", "format": "tflite", "quantization": "int8", "arch": "yolov10"},
    {"filename": "yolo11n_ft_int8.tflite",  "format": "tflite", "quantization": "int8", "arch": "yolo11"},
    {"filename": "yolov8n_ft_fp32.onnx",    "format": "onnx",   "quantization": "fp32", "arch": "yolov8"},
    {"filename": "yolov8s_ft_fp32.onnx",    "format": "onnx",   "quantization": "fp32", "arch": "yolov8"},
    {"filename": "yolov10n_ft_fp32.onnx",   "format": "onnx",   "quantization": "fp32", "arch": "yolov10"},
    {"filename": "yolo11n_ft_fp32.onnx",    "format": "onnx",   "quantization": "fp32", "arch": "yolo11"},
    {"filename": "rtdetr-l_ft_fp32.onnx",   "format": "onnx",   "quantization": "fp32", "arch": "rtdetr"},
    {"filename": "yolov8n_ft_int8.onnx",    "format": "onnx",   "quantization": "int8", "arch": "yolov8"},
    {"filename": "yolov8s_ft_int8.onnx",    "format": "onnx",   "quantization": "int8", "arch": "yolov8"},
    {"filename": "yolov10n_ft_int8.onnx",   "format": "onnx",   "quantization": "int8", "arch": "yolov10"},
    {"filename": "yolo11n_ft_int8.onnx",    "format": "onnx",   "quantization": "int8", "arch": "yolo11"},
    # ─── FINE-TUNED ───────────────────────────────────────────────
    {"filename": "yolov8n_v2_ft_fp32.tflite",  "format": "tflite", "quantization": "fp32", "arch": "yolov8"},
    {"filename": "yolov8s_v2_ft_fp32.tflite",  "format": "tflite", "quantization": "fp32", "arch": "yolov8"},
    {"filename": "yolov10n_v2_ft_fp32.tflite", "format": "tflite", "quantization": "fp32", "arch": "yolov10"},
    {"filename": "yolo11n_v2_ft_fp32.tflite",  "format": "tflite", "quantization": "fp32", "arch": "yolo11"},
    {"filename": "yolov8n_v2_ft_int8.tflite",  "format": "tflite", "quantization": "int8", "arch": "yolov8"},
    {"filename": "yolov8s_v2_ft_int8.tflite",  "format": "tflite", "quantization": "int8", "arch": "yolov8"},
    {"filename": "yolov10n_v2_ft_int8.tflite", "format": "tflite", "quantization": "int8", "arch": "yolov10"},
    {"filename": "yolo11n_v2_ft_int8.tflite",  "format": "tflite", "quantization": "int8", "arch": "yolo11"},
    {"filename": "yolov8n_v2_ft_fp32.onnx",    "format": "onnx",   "quantization": "fp32", "arch": "yolov8"},
    {"filename": "yolov8s_v2_ft_fp32.onnx",    "format": "onnx",   "quantization": "fp32", "arch": "yolov8"},
    {"filename": "yolov10n_v2_ft_fp32.onnx",   "format": "onnx",   "quantization": "fp32", "arch": "yolov10"},
    {"filename": "yolo11n_v2_ft_fp32.onnx",    "format": "onnx",   "quantization": "fp32", "arch": "yolo11"},
    {"filename": "rtdetr-l_v2_ft_fp32.onnx",   "format": "onnx",   "quantization": "fp32", "arch": "rtdetr"},
    {"filename": "yolov8n_v2_ft_int8.onnx",    "format": "onnx",   "quantization": "int8", "arch": "yolov8"},
    {"filename": "yolov8s_v2_ft_int8.onnx",    "format": "onnx",   "quantization": "int8", "arch": "yolov8"},
    {"filename": "yolov10n_v2_ft_int8.onnx",   "format": "onnx",   "quantization": "int8", "arch": "yolov10"},
    {"filename": "yolo11n_v2_ft_int8.onnx",    "format": "onnx",   "quantization": "int8", "arch": "yolo11"},
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
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append({"class": cls, "bbox": [x1, y1, x2, y2]})
    return boxes


def load_test_set() -> list:
    images = sorted(
        list(TEST_IMAGES.glob("*.jpg")) +
        list(TEST_IMAGES.glob("*.png"))
    )
    dataset = []
    for img_path in images:
        lbl_path = TEST_LABELS / (img_path.stem + ".txt")
        dataset.append({"image_path": img_path, "label_path": lbl_path})
    print(f"✅ Test set: {len(dataset)} slika")
    return dataset


# ─── Preprocess – Letterbox ───────────────────────────────────────────────────

def preprocess_image(img_path: Path) -> tuple:
    """Letterbox resize – čuva aspect ratio kao Ultralytics."""
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    scale = IMAGE_SIZE / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    pad_img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (114, 114, 114))
    pad_x = (IMAGE_SIZE - new_w) // 2
    pad_y = (IMAGE_SIZE - new_h) // 2
    pad_img.paste(img_resized, (pad_x, pad_y))

    arr = np.array(pad_img, dtype=np.float32) / 255.0
    return arr, orig_w, orig_h, scale, pad_x, pad_y


# ─── Decode bbox ─────────────────────────────────────────────────────────────

def decode_bbox(x1_or_cx, y1_or_cy, x2_or_w, y2_or_h,
                scale, pad_x, pad_y,
                normalized=False, is_cxcywh=False):
    """Konvertira bbox iz 640x640 letterbox prostora u originalne dimenzije."""
    if is_cxcywh:
        cx, cy, w, h = x1_or_cx, y1_or_cy, x2_or_w, y2_or_h
        if normalized:
            cx *= IMAGE_SIZE
            cy *= IMAGE_SIZE
            w  *= IMAGE_SIZE
            h  *= IMAGE_SIZE
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
    else:
        x1, y1, x2, y2 = x1_or_cx, y1_or_cy, x2_or_w, y2_or_h
        if normalized:
            x1 *= IMAGE_SIZE
            y1 *= IMAGE_SIZE
            x2 *= IMAGE_SIZE
            y2 *= IMAGE_SIZE

    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    return x1, y1, x2, y2


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

def postprocess_yolov8(output, orig_w, orig_h, scale, pad_x, pad_y):
    """Pretrained YOLOv8/v11: (1, 84, 8400)"""
    pred = output[0]
    if pred.shape[0] == 84:
        pred = pred.T

    person_scores = pred[:, 4]
    mask = person_scores > CONF_THRESHOLD
    pred = pred[mask]
    if len(pred) == 0:
        return []

    boxes = pred[:, :4]
    scores = pred[:, 4]
    keep = nms(boxes, scores, IOU_THRESHOLD)

    results = []
    for i in keep:
        cx, cy, w, h = boxes[i]
        normalized = not (cx > 1.0 or cy > 1.0 or w > 1.0 or h > 1.0)
        x1, y1, x2, y2 = decode_bbox(
            cx, cy, w, h, scale, pad_x, pad_y,
            normalized=normalized, is_cxcywh=True
        )
        results.append({"class": 0, "bbox": [x1, y1, x2, y2], "score": float(scores[i])})
    return results


def postprocess_yolov8_ft(output, orig_w, orig_h, scale, pad_x, pad_y):
    """Fine-tuned YOLOv8/v11: (1, 5, 8400), nc=1"""
    pred = output[0]
    if pred.shape[0] == 5:
        pred = pred.T

    scores = pred[:, 4]
    mask = scores > CONF_THRESHOLD
    pred = pred[mask]
    if len(pred) == 0:
        return []

    # Auto-detektiraj normalizirano ili pikseli
    normalized = pred[:, :4].max() <= 1.0

    boxes_640 = pred[:, :4].copy()
    if normalized:
        boxes_640[:, 0] = pred[:, 0] * IMAGE_SIZE
        boxes_640[:, 1] = pred[:, 1] * IMAGE_SIZE
        boxes_640[:, 2] = pred[:, 2] * IMAGE_SIZE
        boxes_640[:, 3] = pred[:, 3] * IMAGE_SIZE

    scores = pred[:, 4]
    keep = nms(boxes_640, scores, IOU_THRESHOLD)

    results = []
    for i in keep:
        cx, cy, w, h = boxes_640[i]
        x1, y1, x2, y2 = decode_bbox(
            cx, cy, w, h, scale, pad_x, pad_y,
            normalized=False, is_cxcywh=True
        )
        results.append({"class": 0, "bbox": [x1, y1, x2, y2], "score": float(scores[i])})
    return results


def postprocess_yolov10(output, orig_w, orig_h, scale, pad_x, pad_y, is_tflite=False):
    """Pretrained YOLOv10: (1, 300, 6) → [x1,y1,x2,y2,score,class]"""
    preds = output[0][0]
    results = []
    for pred in preds:
        x1, y1, x2, y2 = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
        score = float(pred[4])
        cls = int(pred[5])
        if score < CONF_THRESHOLD or cls != 0:
            continue
        x1, y1, x2, y2 = decode_bbox(
            x1, y1, x2, y2, scale, pad_x, pad_y,
            normalized=is_tflite, is_cxcywh=False
        )
        results.append({"class": 0, "bbox": [x1, y1, x2, y2], "score": score})
    return results


def postprocess_yolov10_ft(output, orig_w, orig_h, scale, pad_x, pad_y):
    """Fine-tuned YOLOv10/RT-DETR: (1, 300, 6), auto-detektiraj koordinate"""
    preds = output[0][0]
    if len(preds) == 0:
        return []

    max_coord = float(max(preds[0][0], preds[0][2]))
    normalized = max_coord <= 1.0

    results = []
    for pred in preds:
        x1, y1, x2, y2 = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
        score = float(pred[4])
        if score < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = decode_bbox(
            x1, y1, x2, y2, scale, pad_x, pad_y,
            normalized=normalized, is_cxcywh=False
        )
        results.append({"class": 0, "bbox": [x1, y1, x2, y2], "score": score})
    return results


def postprocess_rtdetr(output, orig_w, orig_h, scale, pad_x, pad_y):
    """Pretrained RT-DETR: (1, 300, N) → [cx,cy,w,h,cls0_score,...]"""
    preds = output[0][0]
    results = []
    for pred in preds:
        if len(pred) < 5:
            continue
        cx, cy, w, h = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
        score = float(pred[4])
        if score < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = decode_bbox(
            cx, cy, w, h, scale, pad_x, pad_y,
            normalized=True, is_cxcywh=True
        )
        results.append({"class": 0, "bbox": [x1, y1, x2, y2], "score": score})
    return results


# ─── Helpers ─────────────────────────────────────────────────────────────────

def is_finetuned(filename: str) -> bool:
    return "_ft_" in filename or "_v2_ft_" in filename


# ─── Inference ───────────────────────────────────────────────────────────────

def run_tflite_inference(interpreter, input_details, output_details,
                         img_arr, orig_w, orig_h, scale, pad_x, pad_y,
                         arch="yolov8", filename=""):
    input_dtype = input_details[0]["dtype"]
    input_shape = input_details[0]["shape"]
    output_shape = output_details[0]["shape"]

    if input_shape[1] == 3:
        inp = np.expand_dims(img_arr.transpose(2, 0, 1), 0)  # BCHW
    else:
        inp = np.expand_dims(img_arr, 0)  # BHWC

    if input_dtype == np.int8:
        s, zp = input_details[0]["quantization"]
        if s != 0:
            inp = (inp / s + zp).astype(np.int8)

    t0 = time.perf_counter()
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if input_dtype == np.int8:
        os_, ozp = output_details[0]["quantization"]
        if os_ != 0:
            output = (output.astype(np.float32) - ozp) * os_

    ft = is_finetuned(filename)

    if len(output_shape) == 3 and output_shape[2] == 6:
        if ft:
            return postprocess_yolov10_ft([output], orig_w, orig_h, scale, pad_x, pad_y), elapsed_ms
        return postprocess_yolov10([output], orig_w, orig_h, scale, pad_x, pad_y, is_tflite=True), elapsed_ms
    elif len(output_shape) == 3 and output_shape[1] == 5:
        return postprocess_yolov8_ft(output, orig_w, orig_h, scale, pad_x, pad_y), elapsed_ms
    else:
        return postprocess_yolov8(output, orig_w, orig_h, scale, pad_x, pad_y), elapsed_ms


def run_onnx_inference(session, img_arr, orig_w, orig_h, scale, pad_x, pad_y,
                       arch="yolov8", filename=""):
    input_name = session.get_inputs()[0].name
    inp = np.expand_dims(img_arr.transpose(2, 0, 1), 0)  # BCHW

    t0 = time.perf_counter()
    output = session.run(None, {input_name: inp})
    elapsed_ms = (time.perf_counter() - t0) * 1000

    ft = is_finetuned(filename)
    out_shape = output[0].shape

    if len(out_shape) == 3 and out_shape[2] == 6:
        if arch == "rtdetr" and not ft:
            return postprocess_rtdetr(output, orig_w, orig_h, scale, pad_x, pad_y), elapsed_ms
        return postprocess_yolov10_ft(output, orig_w, orig_h, scale, pad_x, pad_y), elapsed_ms
    elif len(out_shape) == 3 and out_shape[1] == 5:
        return postprocess_yolov8_ft(output[0], orig_w, orig_h, scale, pad_x, pad_y), elapsed_ms
    else:
        return postprocess_yolov8(output[0], orig_w, orig_h, scale, pad_x, pad_y), elapsed_ms


# ─── mAP računanje ───────────────────────────────────────────────────────────

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
        all_scores, all_tp, all_fp = [], [], []
        n_gt = sum(len(gt) for gt in all_ground_truths)

        for dets, gts in zip(all_detections, all_ground_truths):
            matched = [False] * len(gts)
            for det in sorted(dets, key=lambda x: x["score"], reverse=True):
                all_scores.append(det["score"])
                best_iou, best_j = 0, -1
                for j, gt in enumerate(gts):
                    if matched[j]:
                        continue
                    iou = compute_iou(det["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= iou_thresh and best_j >= 0:
                    all_tp.append(1)
                    all_fp.append(0)
                    matched[best_j] = True
                else:
                    all_tp.append(0)
                    all_fp.append(1)

        if not all_scores:
            aps.append(0.0)
            continue

        idx = np.argsort(all_scores)[::-1]
        tp_cum = np.cumsum(np.array(all_tp)[idx])
        fp_cum = np.cumsum(np.array(all_fp)[idx])
        precision = tp_cum / (tp_cum + fp_cum + 1e-10)
        recall = tp_cum / (n_gt + 1e-10)

        precision = np.concatenate(([1.0], precision, [0.0]))
        recall = np.concatenate(([0.0], recall, [1.0]))

        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])

        ap = 0.0
        for r in np.linspace(0, 1, 101):
            p = precision[recall >= r]
            ap += (np.max(p) if len(p) > 0 else 0.0)
        aps.append(ap / 101)

    return float(aps[0]) if aps else 0.0, float(np.mean(aps)) if aps else 0.0


# ─── Evaluacija jednog modela ─────────────────────────────────────────────────

def evaluate_model(model_info: dict, dataset: list, model_type: str) -> dict:
    filename = model_info["filename"]
    fmt = model_info["format"]
    quant = model_info["quantization"]
    arch = model_info.get("arch", "yolov8")

    model_path = MODELS_DIR / fmt / quant / filename

    print(f"\n{'='*60}")
    print(f"Evaluiram: {filename} [{model_type}] ft={is_finetuned(filename)}")
    print(f"{'='*60}")

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
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

    ram_after_load = get_ram_mb()
    ram_model_mb = ram_after_load - ram_before

    # Warmup
    dummy_arr = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
    dummy_scale, dummy_pad_x, dummy_pad_y = 1.0, 0, 0
    print(f"Warmup ({NUM_WARMUP} iteracija)...")
    for _ in range(NUM_WARMUP):
        if fmt == "tflite":
            run_tflite_inference(interpreter, input_details, output_details,
                                dummy_arr, IMAGE_SIZE, IMAGE_SIZE,
                                dummy_scale, dummy_pad_x, dummy_pad_y, arch, filename)
        else:
            run_onnx_inference(session, dummy_arr, IMAGE_SIZE, IMAGE_SIZE,
                              dummy_scale, dummy_pad_x, dummy_pad_y, arch, filename)

    # Inference na test setu
    all_detections = []
    all_ground_truths = []
    inference_times = []
    temp_before = get_temperature()

    for item in tqdm(dataset, desc=filename):
        img_arr, orig_w, orig_h, scale, pad_x, pad_y = preprocess_image(item["image_path"])
        gt_boxes = load_yolo_labels(item["label_path"], orig_w, orig_h)

        if fmt == "tflite":
            dets, ms = run_tflite_inference(
                interpreter, input_details, output_details,
                img_arr, orig_w, orig_h, scale, pad_x, pad_y, arch, filename
            )
        else:
            dets, ms = run_onnx_inference(
                session, img_arr, orig_w, orig_h, scale, pad_x, pad_y, arch, filename
            )

        all_detections.append(dets)
        all_ground_truths.append(gt_boxes)
        inference_times.append(ms)

    temp_after = get_temperature()
    ram_peak = get_ram_mb()
    avg_ms = float(np.mean(inference_times))

    map_50, _ = compute_map(all_detections, all_ground_truths, [0.5])
    _, map_5095 = compute_map(
        all_detections, all_ground_truths,
        np.arange(0.5, 1.0, 0.05).tolist()
    )

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
    print(f"  Temp:         {result['temp_before_c']:.1f}°C → {result['temp_after_c']:.1f}°C")

    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="pretrained",
                        choices=["pretrained", "finetuned", "finetuned-v2"])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"🚀 Evaluacija na RPi 5 – {args.type.upper()}")
    print(f"Test set: {TEST_IMAGES}")
    print(f"RAM ukupno: {psutil.virtual_memory().total / 1024 / 1024:.0f} MB")

    if not TEST_IMAGES.exists():
        print(f"❌ Test images folder ne postoji: {TEST_IMAGES}")
        return
    if not TEST_LABELS.exists():
        print(f"❌ Test labels folder ne postoji: {TEST_LABELS}")
        return

    dataset = load_test_set()
    if len(dataset) == 0:
        print(f"❌ Nema slika u: {TEST_IMAGES}")
        return

    models_to_eval = ALL_MODELS
    if args.models:
        models_to_eval = [m for m in ALL_MODELS if m["filename"] in args.models]

    available_models = []
    missing_models = []
    for m in models_to_eval:
        path = MODELS_DIR / m["format"] / m["quantization"] / m["filename"]
        if path.exists():
            available_models.append(m)
        else:
            missing_models.append(m["filename"])

    if missing_models:
        print(f"\n⚠️  Nedostaju modeli ({len(missing_models)}):")
        for name in missing_models:
            print(f"   - {name}")

    if not available_models:
        print("❌ Nema dostupnih modela!")
        return

    print(f"\n✅ Evaluiram {len(available_models)} modela:")
    for m in available_models:
        print(f"   - {m['filename']} (ft={is_finetuned(m['filename'])})")

    output_file = Path(args.output) if args.output else \
        RESULTS_DIR / "benchmark_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    existing_results = []
    if output_file.exists():
        with open(output_file) as f:
            existing_results = json.load(f)
        print(f"📂 Učitano {len(existing_results)} postojećih rezultata")

    new_results = []
    for model_info in available_models:
        result = evaluate_model(model_info, dataset, args.type)
        if result:
            new_results.append(result)
            all_results = existing_results + new_results
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"💾 Spremljeno: {output_file}")

    if not new_results:
        print("❌ Nijedan model nije uspješno evaluiran!")
        return

    print(f"\n✅ Gotovo! Evaluirano: {len(new_results)} modela")
    print(f"\n{'='*80}")
    print(f"{'Model':<40} {'Type':>10} {'mAP@0.5':>8} {'FPS':>7} {'RAM':>8}")
    print(f"{'='*80}")
    for r in new_results:
        print(f"{r['model']:<40} {r['type']:>10} "
              f"{r['mAP_0.5']:>8.4f} {r['avg_fps']:>7.1f} "
              f"{r['ram_model_mb']:>7.1f}MB")


if __name__ == "__main__":
    main()