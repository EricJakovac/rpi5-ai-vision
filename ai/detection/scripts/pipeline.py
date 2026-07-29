"""
Integrirani pipeline:
PIR senzor → Detekcija osobe (YOLOv8n TFLite INT8) → Prepoznavanje lica (InsightFace)

Box boje:
  Plavi  → osoba bez lica (leđa, daleko, itd.)
  Crveni → nepoznata osoba (lice pronađeno ali nije u bazi)
  Zeleni → poznata osoba
"""

import RPi.GPIO as GPIO
from ai_edge_litert.interpreter import Interpreter
import insightface
from insightface.app import FaceAnalysis
from picamera2 import Picamera2, Preview
from PIL import Image as PILImage, ImageDraw, ImageFont
import numpy as np
import json
import time
from pathlib import Path

# ─── Konfiguracija ───────────────────────────────────────────────────────────

PIR_PIN = 17

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "tflite" / "int8" / "yolov8n_int8.tflite"
DB_PATH = Path(__file__).parent.parent.parent / "recognition" / "face_database.json"

IMAGE_SIZE = 640
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
SIMILARITY_THRESHOLD = 0.5

CAM_WIDTH = 1280
CAM_HEIGHT = 720

PIR_STABILIZATION = 20
CAMERA_START_OFFSET = 10
IGNORE_AFTER_STABILIZATION = 3
MIN_TRIGGER_TIME = 0.8
POST_MOTION_DELAY = 5


# ─── Face database ───────────────────────────────────────────────────────────

def load_database() -> dict:
    if not DB_PATH.exists():
        print(f"⚠️  Baza lica ne postoji: {DB_PATH}")
        return {}
    with open(DB_PATH, "r") as f:
        db = json.load(f)
    persons = {}
    for name, data in db["persons"].items():
        persons[name] = np.array(data["embedding"])
    print(f"✅ Baza lica učitana – {len(persons)} osoba: {list(persons.keys())}")
    return persons


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


# ─── Detekcija osobe (TFLite INT8) ───────────────────────────────────────────

def detect_persons(interpreter, input_details, output_details, frame) -> list:
    input_dtype = input_details[0]['dtype']

    img = np.array(
        PILImage.fromarray(frame).resize((IMAGE_SIZE, IMAGE_SIZE)),
        dtype=np.float32
    ) / 255.0
    img = np.expand_dims(img, axis=0)  # BHWC: (1, 640, 640, 3)

    if input_dtype == np.int8:
        scale, zero_point = input_details[0]['quantization']
        if scale != 0:
            img = (img / scale + zero_point).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    if input_dtype == np.int8:
        out_scale, out_zp = output_details[0]['quantization']
        if out_scale != 0:
            output = (output.astype(np.float32) - out_zp) * out_scale

    predictions = output[0].T  # (8400, 84)
    persons_mask = (
        (np.argmax(predictions[:, 4:], axis=1) == 0) &
        (np.max(predictions[:, 4:], axis=1) > CONF_THRESHOLD)
    )
    persons = predictions[persons_mask]
    if len(persons) == 0:
        return []

    boxes = persons[:, :4]
    scores = np.max(persons[:, 4:], axis=1)
    keep = nms(boxes, scores, IOU_THRESHOLD)
    return [(persons[i, :4], float(scores[i])) for i in keep]


# ─── Prepoznavanje lica (InsightFace) ────────────────────────────────────────

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def identify_face(embedding: np.ndarray, persons: dict) -> tuple:
    best_name = None
    best_score = -1.0
    for name, db_embedding in persons.items():
        score = cosine_similarity(embedding, db_embedding)
        if score > best_score:
            best_score = score
            best_name = name
    if best_score >= SIMILARITY_THRESHOLD:
        return best_name, best_score
    return None, best_score


def recognize_in_person_bbox(face_app, frame, person_bbox, persons: dict) -> tuple:
    """
    Vraća:
      (ime, score)  → poznata osoba
      (None, score) → nepoznata osoba, lice pronađeno
      (None, -1.0)  → nema lica u bbox-u
    """
    if not persons:
        return None, -1.0

    faces = face_app.get(frame)
    if not faces:
        return None, -1.0  # nema lica u sceni

    # Pronađi lice koje se preklapa s YOLO person bbox-om
    cx, cy, w, h = person_bbox
    px1 = (cx - w / 2) * CAM_WIDTH
    py1 = (cy - h / 2) * CAM_HEIGHT
    px2 = (cx + w / 2) * CAM_WIDTH
    py2 = (cy + h / 2) * CAM_HEIGHT

    best_face = None
    best_overlap = 0.0

    for face in faces:
        fx1, fy1, fx2, fy2 = face.bbox
        ix1 = max(px1, fx1)
        iy1 = max(py1, fy1)
        ix2 = min(px2, fx2)
        iy2 = min(py2, fy2)
        if ix2 > ix1 and iy2 > iy1:
            overlap = (ix2 - ix1) * (iy2 - iy1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_face = face

    if best_face is None:
        return None, -1.0  # lice nije unutar person bbox-a

    return identify_face(best_face.embedding, persons)


# ─── Overlay ─────────────────────────────────────────────────────────────────

def draw_overlay(picam2, detections_with_identity, pir_active, inference_ms):
    """
    detections_with_identity: lista (bbox, conf, ime, face_score)
    
    Logika boja:
      face_score == -1.0 → nema lica → PLAVI box "Osoba"
      face_score >= 0.0, name=None → nepoznato → CRVENI box "Nepoznata osoba"
      name postoji → poznato → ZELENI box "Ime (score)"
    """
    overlay = PILImage.new("RGBA", (CAM_WIDTH, CAM_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18
        )
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    for bbox, conf, name, face_score in detections_with_identity:
        cx, cy, w, h = bbox
        x1 = int((cx - w / 2) * CAM_WIDTH)
        y1 = int((cy - h / 2) * CAM_HEIGHT)
        x2 = int((cx + w / 2) * CAM_WIDTH)
        y2 = int((cy + h / 2) * CAM_HEIGHT)

        if name:
            # Poznata osoba → zeleni
            color = (0, 255, 0, 255)
            label = f"{name} ({face_score:.2f})"
        elif face_score >= 0.0:
            # Nepoznata osoba (lice pronađeno) → crveni
            color = (255, 0, 0, 255)
            label = f"Nepoznata osoba ({face_score:.2f})"
        else:
            # Osoba bez lica → plavi
            color = (0, 150, 255, 255)
            label = f"Osoba ({conf:.2f})"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        bbox_text = draw.textbbox((x1, y1 - 30), label, font=font)
        draw.rectangle(bbox_text, fill=(0, 0, 0, 180))
        draw.text((x1, y1 - 30), label, fill=color, font=font)

    # Info overlay
    pir_status = "AKTIVAN" if pir_active else "CEKAM"
    info_lines = [
        f"PIR: {pir_status}",
        f"Inference: {inference_ms:.0f}ms",
        f"Osobe: {len(detections_with_identity)}",
    ]
    y_offset = 10
    for line in info_lines:
        bbox_info = draw.textbbox((10, y_offset), line, font=font_small)
        draw.rectangle(
            (bbox_info[0]-5, bbox_info[1]-3, bbox_info[2]+5, bbox_info[3]+3),
            fill=(0, 0, 0, 160)
        )
        draw.text((10, y_offset), line, fill=(0, 255, 0, 255), font=font_small)
        y_offset += 28

    picam2.set_overlay(np.array(overlay))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=== Integrirani Pipeline: PIR + Detekcija + Prepoznavanje ===")
    print("Boje: 🔵 Osoba | 🔴 Nepoznata osoba | 🟢 Poznata osoba\n")

    persons = load_database()

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIR_PIN, GPIO.IN)

    print(f"Učitavam detekcijski model: {MODEL_PATH.name}")
    interpreter = Interpreter(model_path=str(MODEL_PATH), num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Detekcijski model učitan")

    print("Učitavam InsightFace model...")
    face_app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ InsightFace učitan")

    print(f"\n⏳ PIR stabilizacija ({PIR_STABILIZATION}s)...")
    time.sleep(CAMERA_START_OFFSET)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start_preview(Preview.QT)
    picam2.start()

    remaining = PIR_STABILIZATION - CAMERA_START_OFFSET
    print(f"✅ Kamera pokrenuta (PIR završava za {remaining}s)")
    time.sleep(remaining)
    print("✅ PIR stabiliziran!")

    ignore_until = time.time() + IGNORE_AFTER_STABILIZATION
    while time.time() < ignore_until:
        time.sleep(0.1)

    print("💤 Spreman! Čekam pokret...\n")

    is_active = False
    last_motion_time = 0
    trigger_start = None

    try:
        while True:
            pir_state = GPIO.input(PIR_PIN)
            now = time.time()

            if pir_state == 1:
                last_motion_time = now
                if not is_active:
                    if trigger_start is None:
                        trigger_start = now
                    elif now - trigger_start >= MIN_TRIGGER_TIME:
                        print("🟢 Pokret detektiran – aktiviram pipeline!")
                        is_active = True
                        trigger_start = None
            else:
                if not is_active:
                    trigger_start = None

            if is_active:
                time_since_motion = now - last_motion_time

                if time_since_motion <= POST_MOTION_DELAY:
                    frame = picam2.capture_array()
                    t0 = time.perf_counter()

                    person_detections = detect_persons(
                        interpreter, input_details, output_details, frame
                    )

                    detections_with_identity = []
                    for bbox, conf in person_detections:
                        if persons:
                            name, face_score = recognize_in_person_bbox(
                                face_app, frame, bbox, persons
                            )
                        else:
                            name, face_score = None, -1.0

                        detections_with_identity.append((bbox, conf, name, face_score))

                    inference_ms = (time.perf_counter() - t0) * 1000

                    draw_overlay(picam2, detections_with_identity, True, inference_ms)

                    # Terminal ispis – usklađen s prikazom
                    if detections_with_identity:
                        for bbox, conf, name, face_score in detections_with_identity:
                            if name:
                                print(f"[{inference_ms:.0f}ms] 🟢 {name} ({face_score:.3f})")
                            elif face_score >= 0.0:
                                print(f"[{inference_ms:.0f}ms] 🔴 Nepoznata osoba ({face_score:.3f})")
                            else:
                                print(f"[{inference_ms:.0f}ms] 🔵 Osoba ({conf:.2f}) – nema lica")
                    else:
                        print(f"[{inference_ms:.0f}ms] Nema osoba")

                else:
                    draw_overlay(picam2, [], False, 0)
                    print("💤 Nema pokreta – pipeline se gasi\n")
                    is_active = False

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Gašenje...")
    finally:
        picam2.stop()
        GPIO.cleanup()


if __name__ == "__main__":
    main()