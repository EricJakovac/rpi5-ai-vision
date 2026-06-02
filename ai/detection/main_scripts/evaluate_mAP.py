"""
mAP evaluacija SVIH modela (ONNX + TFLite) na COCO val2017 datasetu.
Podržava: YOLOv8, YOLOv10, RT-DETR
"""

import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image as PILImage

# Putanje
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = Path.home() / "Desktop" / "rpi5-ai-vision" / "datasets" / "coco"
IMAGES_DIR = DATASET_DIR / "val2017"
ANNOTATIONS_FILE = DATASET_DIR / "annotations" / "instances_val2017.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_IMAGES = 500
PERSON_CLASS_COCO = 1


# ─── Provjere ────────────────────────────────────────────────────────────────

def check_dataset() -> bool:
    print("\n📂 Provjera dataseta:")
    print(f"   Slike: {IMAGES_DIR}")
    print(f"   Anotacije: {ANNOTATIONS_FILE}")
    if not IMAGES_DIR.exists():
        print("   ❌ Folder sa slikama ne postoji!")
        return False
    img_count = len(list(IMAGES_DIR.glob("*.jpg")))
    if img_count == 0:
        print("   ❌ Nema slika!")
        return False
    if not ANNOTATIONS_FILE.exists():
        print("   ❌ Anotacije ne postoje!")
        return False
    print(f"   ✅ Dataset OK – {img_count} slika")
    return True


def check_models() -> dict:
    models = {"onnx": [], "tflite": []}

    print("\n📂 Provjera ONNX modela:")
    for subfolder in ["fp32", "int8"]:
        folder = BASE_DIR / "models" / "onnx" / subfolder
        if folder.exists():
            models["onnx"].extend(sorted(folder.glob("*.onnx")))
    if models["onnx"]:
        print(f"   ✅ Pronađeno {len(models['onnx'])} modela:")
        for m in models["onnx"]:
            print(f"      - {m.name} ({m.stat().st_size/1024/1024:.1f} MB)")
    else:
        print("   ❌ Nema ONNX modela!")

    print("\n📂 Provjera TFLite modela:")
    for subfolder in ["fp32", "int8"]:
        folder = BASE_DIR / "models" / "tflite" / subfolder
        if folder.exists():
            models["tflite"].extend(sorted(folder.glob("*.tflite")))
    if models["tflite"]:
        print(f"   ✅ Pronađeno {len(models['tflite'])} modela:")
        for m in models["tflite"]:
            print(f"      - {m.name} ({m.stat().st_size/1024/1024:.1f} MB)")
    else:
        print("   ❌ Nema TFLite modela!")

    return models


# ─── Preprocessing ───────────────────────────────────────────────────────────

def preprocess_onnx(image_path: Path, size: int):
    """BCHW format za ONNX."""
    img = PILImage.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_array = np.array(img.resize((size, size)), dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array.transpose(2, 0, 1), axis=0)
    return img_array, orig_w, orig_h


def preprocess_tflite(image_path: Path, size: int, input_dtype, input_details):
    """BHWC format za TFLite."""
    img = PILImage.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_array = np.array(img.resize((size, size)), dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # BHWC

    if input_dtype == np.int8:
        scale, zero_point = input_details[0]['quantization']
        if scale != 0:
            img_array = (img_array / scale + zero_point).astype(np.int8)
        else:
            img_array = img_array.astype(np.int8)

    return img_array, orig_w, orig_h


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


# ─── Postprocessing ──────────────────────────────────────────────────────────

def postprocess_yolov8(output, orig_w, orig_h, conf_threshold, iou_threshold, tflite=False):
    """
    YOLOv8 output: (1, 84, 8400)
    Format: [cx, cy, w, h, 80 class scores]
    ONNX:   koordinate u pikselima 0-640
    TFLite: koordinate normalizirane 0-1
    """
    predictions = output[0].T  # (8400, 84)

    persons_mask = (
        (np.argmax(predictions[:, 4:], axis=1) == 0) &
        (np.max(predictions[:, 4:], axis=1) > conf_threshold)
    )
    persons = predictions[persons_mask]
    if len(persons) == 0:
        return []

    boxes = persons[:, :4]
    scores = np.max(persons[:, 4:], axis=1)
    keep = nms(boxes, scores, iou_threshold)

    detections = []
    for i in keep:
        cx, cy, w, h = persons[i, :4]
        if tflite:
            cx = float(cx) * orig_w
            cy = float(cy) * orig_h
            w  = float(w)  * orig_w
            h  = float(h)  * orig_h
        else:
            cx = float(cx) / IMAGE_SIZE * orig_w
            cy = float(cy) / IMAGE_SIZE * orig_h
            w  = float(w)  / IMAGE_SIZE * orig_w
            h  = float(h)  / IMAGE_SIZE * orig_h
        x1 = cx - w / 2
        y1 = cy - h / 2
        if w <= 0 or h <= 0:
            continue
        detections.append({
            "bbox": [x1, y1, w, h],
            "score": float(scores[i]),
            "category_id": PERSON_CLASS_COCO
        })
    return detections


def postprocess_yolov10(output, orig_w, orig_h, conf_threshold, tflite=False):
    """
    YOLOv10 output: (1, 300, 6) – već NMS procesiran
    Format: [x1, y1, x2, y2, score, class_id]
    ONNX:   koordinate u pikselima 0-640
    TFLite: koordinate normalizirane 0-1
    """
    predictions = output[0]  # (300, 6)
    detections = []
    for pred in predictions:
        if len(pred) < 6:
            continue
        x1, y1, x2, y2, score, class_id = pred[:6]
        if int(class_id) != 0 or float(score) < conf_threshold:
            continue

        if tflite:
            x1 = float(x1) * orig_w
            y1 = float(y1) * orig_h
            x2 = float(x2) * orig_w
            y2 = float(y2) * orig_h
        else:
            x1 = float(x1) / IMAGE_SIZE * orig_w
            y1 = float(y1) / IMAGE_SIZE * orig_h
            x2 = float(x2) / IMAGE_SIZE * orig_w
            y2 = float(y2) / IMAGE_SIZE * orig_h

        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        detections.append({
            "bbox": [x1, y1, w, h],
            "score": float(score),
            "category_id": PERSON_CLASS_COCO
        })
    return detections


def postprocess_rtdetr(output, orig_w, orig_h, conf_threshold):
    """
    RT-DETR output: (1, 300, 84)
    Format: [cx, cy, w, h, class_scores...]
    Koordinate normalizirane 0-1
    """
    predictions = output[0]  # (300, 84)
    detections = []
    for pred in predictions:
        scores = pred[4:]
        class_id = int(np.argmax(scores))
        score = float(scores[class_id])
        if class_id != 0 or score < conf_threshold:
            continue
        cx = float(pred[0]) * orig_w
        cy = float(pred[1]) * orig_h
        w  = float(pred[2]) * orig_w
        h  = float(pred[3]) * orig_h
        x1 = cx - w / 2
        y1 = cy - h / 2
        if w <= 0 or h <= 0:
            continue
        detections.append({
            "bbox": [x1, y1, w, h],
            "score": score,
            "category_id": PERSON_CLASS_COCO
        })
    return detections


def detect_model_type(model_name: str) -> str:
    name = model_name.lower()
    if "rtdetr" in name or "rt-detr" in name:
        return "rtdetr"
    elif "v10" in name:
        return "yolov10"
    else:
        return "yolov8"


# ─── COCO evaluacija ─────────────────────────────────────────────────────────

def run_coco_eval(coco_gt, coco_results, image_ids):
    if not coco_results:
        return None
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.catIds = [PERSON_CLASS_COCO]
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def _build_results(model, fmt, quant, coco_eval, inference_times, num_images):
    return {
        "model": model,
        "format": fmt,
        "quantization": quant,
        "image_size": IMAGE_SIZE,
        "num_images": num_images,
        "mAP_0.5":       round(float(coco_eval.stats[1]), 4),
        "mAP_0.5_0.95":  round(float(coco_eval.stats[0]), 4),
        "avg_precision":  round(float(coco_eval.stats[8]), 4),
        "avg_recall":     round(float(coco_eval.stats[6]), 4),
        "avg_inference_ms": round(float(np.mean(inference_times)), 2),
        "avg_fps":          round(1000 / float(np.mean(inference_times)), 2),
        "timestamp": datetime.now().isoformat(),
    }


def _print_results(r):
    print(f"\n📊 Rezultati:")
    print(f"  mAP@0.5:       {r['mAP_0.5']:.4f}")
    print(f"  mAP@0.5:0.95:  {r['mAP_0.5_0.95']:.4f}")
    print(f"  Precision:     {r['avg_precision']:.4f}")
    print(f"  Recall:        {r['avg_recall']:.4f}")
    print(f"  Avg FPS:       {r['avg_fps']:.1f}")


# ─── ONNX evaluacija ─────────────────────────────────────────────────────────

def evaluate_onnx(model_path: Path, coco_gt, image_ids) -> dict:
    print(f"\n{'='*55}")
    print(f"[ONNX] {model_path.name}")
    print(f"{'='*55}")

    model_type = detect_model_type(model_path.name)
    print(f"Tip modela: {model_type}")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    test_out = session.run(None, {input_name: np.random.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)})
    print(f"Output shape: {test_out[0].shape}")

    coco_results = []
    inference_times = []

    for idx, img_id in enumerate(image_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = IMAGES_DIR / img_info["file_name"]
        if not img_path.exists():
            continue

        img_array, orig_w, orig_h = preprocess_onnx(img_path, IMAGE_SIZE)

        t0 = time.perf_counter()
        output = session.run(None, {input_name: img_array})
        inference_times.append((time.perf_counter() - t0) * 1000)

        if model_type == "yolov10":
            detections = postprocess_yolov10(
                output[0], orig_w, orig_h, CONF_THRESHOLD, tflite=False
            )
        elif model_type == "rtdetr":
            detections = postprocess_rtdetr(
                output[0], orig_w, orig_h, CONF_THRESHOLD
            )
        else:
            detections = postprocess_yolov8(
                output[0], orig_w, orig_h, CONF_THRESHOLD, IOU_THRESHOLD, tflite=False
            )

        for det in detections:
            coco_results.append({
                "image_id": img_id,
                "category_id": det["category_id"],
                "bbox": det["bbox"],
                "score": det["score"]
            })

        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(image_ids)} – avg: {np.mean(inference_times):.1f}ms | detekcija: {len(coco_results)}")

    coco_eval = run_coco_eval(coco_gt, coco_results, image_ids)
    if coco_eval is None:
        print("❌ Nema detekcija!")
        return None

    results = _build_results(
        model_path.name, "onnx",
        "int8" if "int8" in model_path.name else "fp32",
        coco_eval, inference_times, len(image_ids)
    )
    _print_results(results)
    return results


# ─── TFLite evaluacija ───────────────────────────────────────────────────────

def evaluate_tflite(model_path: Path, coco_gt, image_ids) -> dict:
    print(f"\n{'='*55}")
    print(f"[TFLite] {model_path.name}")
    print(f"{'='*55}")

    model_type = detect_model_type(model_path.name)
    print(f"Tip modela: {model_type}")

    interpreter = Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype    = input_details[0]['dtype']

    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_dtype}")

    dummy = np.zeros(input_details[0]['shape'], dtype=input_dtype)
    interpreter.set_tensor(input_details[0]['index'], dummy)
    interpreter.invoke()
    test_out = interpreter.get_tensor(output_details[0]['index'])
    print(f"Output shape: {test_out.shape}")

    coco_results = []
    inference_times = []

    for idx, img_id in enumerate(image_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = IMAGES_DIR / img_info["file_name"]
        if not img_path.exists():
            continue

        img_array, orig_w, orig_h = preprocess_tflite(
            img_path, IMAGE_SIZE, input_dtype, input_details
        )

        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        inference_times.append((time.perf_counter() - t0) * 1000)

        # Dequantize INT8 output
        if input_dtype == np.int8:
            out_scale, out_zp = output_details[0]['quantization']
            if out_scale != 0:
                output = (output.astype(np.float32) - out_zp) * out_scale

        if model_type == "yolov10":
            detections = postprocess_yolov10(
                output, orig_w, orig_h, CONF_THRESHOLD, tflite=True
            )
        else:
            detections = postprocess_yolov8(
                output, orig_w, orig_h, CONF_THRESHOLD, IOU_THRESHOLD, tflite=True
            )

        for det in detections:
            coco_results.append({
                "image_id": img_id,
                "category_id": det["category_id"],
                "bbox": det["bbox"],
                "score": det["score"]
            })

        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(image_ids)} – avg: {np.mean(inference_times):.1f}ms | detekcija: {len(coco_results)}")

    coco_eval = run_coco_eval(coco_gt, coco_results, image_ids)
    if coco_eval is None:
        print("❌ Nema detekcija!")
        return None

    results = _build_results(
        model_path.name, "tflite",
        "int8" if "int8" in model_path.name else "fp32",
        coco_eval, inference_times, len(image_ids)
    )
    _print_results(results)
    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("🚀 mAP Evaluacija (ONNX + TFLite) – RPi 5")

    if not check_dataset():
        print("\n❌ Dataset nije pronađen – prekidam.")
        return

    models = check_models()
    total = len(models["onnx"]) + len(models["tflite"])
    if total == 0:
        print("\n❌ Nema modela – prekidam.")
        return

    print(f"\n✅ Sve provjere prošle – krećem s evaluacijom")
    print(f"   ONNX modela:   {len(models['onnx'])}")
    print(f"   TFLite modela: {len(models['tflite'])}")
    print(f"   Broj slika:    {MAX_IMAGES}")
    print(f"   Procjena:      ~{total * MAX_IMAGES * 0.2 / 60:.0f} minuta\n")

    print("Učitavam COCO anotacije...")
    coco_gt = COCO(str(ANNOTATIONS_FILE))
    person_img_ids = coco_gt.getImgIds(catIds=[PERSON_CLASS_COCO])
    image_ids = sorted(person_img_ids)[:MAX_IMAGES]
    print(f"Evaluacija na {len(image_ids)} slika s osobama\n")

    all_results = []

    print("\n" + "█"*55)
    print("  ONNX EVALUACIJA")
    print("█"*55)
    for model_path in models["onnx"]:
        result = evaluate_onnx(model_path, coco_gt, image_ids)
        if result:
            all_results.append(result)

    print("\n" + "█"*55)
    print("  TFLite EVALUACIJA")
    print("█"*55)
    for model_path in models["tflite"]:
        result = evaluate_tflite(model_path, coco_gt, image_ids)
        if result:
            all_results.append(result)

    if not all_results:
        print("\n❌ Nema rezultata – prekidam.")
        return

    output_file = RESULTS_DIR / f"map_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Rezultati spremljeni: {output_file}")

    print(f"\n{'='*90}")
    print(f"{'Model':<28} {'Format':>7} {'Quant':>6} {'mAP@.5':>8} {'mAP@.5:.95':>12} {'Prec':>6} {'Recall':>8} {'FPS':>6}")
    print(f"{'='*90}")
    for r in all_results:
        print(f"{r['model']:<28} {r['format']:>7} {r['quantization']:>6} "
              f"{r['mAP_0.5']:>8.4f} {r['mAP_0.5_0.95']:>12.4f} "
              f"{r['avg_precision']:>6.4f} {r['avg_recall']:>8.4f} "
              f"{r['avg_fps']:>6.1f}")


if __name__ == "__main__":
    main()
