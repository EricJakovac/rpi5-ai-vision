"""
PIR + AI kontinuirana detekcija.
"""

import RPi.GPIO as GPIO
import onnxruntime as ort
import numpy as np
import time
from picamera2 import Picamera2, Preview
from PIL import Image as PILImage, ImageDraw, ImageFont
from pathlib import Path

PIR_PIN = 17
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "onnx" / "yolov8n_fp32.onnx"
IMAGE_SIZE = 640
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

PIR_STABILIZATION = 20         # sekundi za PIR stabilizaciju
CAMERA_START_OFFSET = 10       # sekundi od starta kada se pali kamera
IGNORE_AFTER_STABILIZATION = 3 # sekundi ignoriranja PIR-a odmah nakon stabilizacije (lažni okidač)
MIN_TRIGGER_TIME = 0.8         # signal mora trajati barem ovako dugo da se smatra pravim pokretom
POST_MOTION_DELAY = 2          # sekundi bez pokreta prije gašenja AI


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


def detect_persons(session, frame):
    input_name = session.get_inputs()[0].name
    img = np.array(
        PILImage.fromarray(frame).resize((IMAGE_SIZE, IMAGE_SIZE)),
        dtype=np.float32
    ) / 255.0
    inp = np.expand_dims(img.transpose(2, 0, 1), 0)
    output = session.run(None, {input_name: inp})[0]
    predictions = output[0].T
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
    return [(persons[i, :4], float(np.max(persons[i, 4:]))) for i in keep]


def draw_overlay(picam2, detections, cam_width=1280, cam_height=720):
    overlay = PILImage.new("RGBA", (cam_width, cam_height), (0, 0, 0, 0))
    if detections:
        draw = ImageDraw.Draw(overlay)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
        for box, conf in detections:
            cx, cy, w, h = box
            x1 = int((cx - w / 2) * cam_width / IMAGE_SIZE)
            y1 = int((cy - h / 2) * cam_height / IMAGE_SIZE)
            x2 = int((cx + w / 2) * cam_width / IMAGE_SIZE)
            y2 = int((cy + h / 2) * cam_height / IMAGE_SIZE)
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=3)
            draw.text((x1 + 4, y1 + 4), f"osoba {conf:.2f}", fill=(0, 255, 0, 255), font=font)
    picam2.set_overlay(np.array(overlay))


def main():
    print("=== PIR + AI Kontinuirana Detekcija ===\n")

    # GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIR_PIN, GPIO.IN)

    # Model učitaj odmah (brzo, ne čeka stabilizaciju)
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )
    print(f"✅ Model učitan: {MODEL_PATH.name}")

    # PIR stabilizacija — kamera se pali u sredini da ima vremena zagrijati se
    print(f"⏳ PIR stabilizacija ({PIR_STABILIZATION}s) — kamera se pali za {CAMERA_START_OFFSET}s...")
    time.sleep(CAMERA_START_OFFSET)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start_preview(Preview.QT)
    picam2.start()
    remaining = PIR_STABILIZATION - CAMERA_START_OFFSET
    print(f"✅ Kamera pokrenuta (PIR završava za {remaining}s)")

    time.sleep(remaining)
    print("✅ PIR stabiliziran!")

    # Ignoriraj lažni okidač odmah nakon stabilizacije
    print(f"⏳ Ignoriram lažne okidače ({IGNORE_AFTER_STABILIZATION}s)...")
    ignore_until = time.time() + IGNORE_AFTER_STABILIZATION
    while time.time() < ignore_until:
        time.sleep(0.1)

    print("💤 Spreman! Čekam prvi pokret...\n")

    # State machine
    is_active = False
    last_motion_time = 0
    trigger_start = None  # za MIN_TRIGGER_TIME debounce

    try:
        while True:
            pir_state = GPIO.input(PIR_PIN)
            now = time.time()

            if pir_state == 1:
                last_motion_time = now

                if not is_active:
                    # Debounce: čekaj da signal traje barem MIN_TRIGGER_TIME
                    if trigger_start is None:
                        trigger_start = now
                    elif now - trigger_start >= MIN_TRIGGER_TIME:
                        print("🟢 Pokret detektiran – aktiviram AI!")
                        is_active = True
                        trigger_start = None
            else:
                # Signal pao — resetiraj debounce ako AI još nije aktivan
                if not is_active:
                    trigger_start = None

            if is_active:
                time_since_motion = now - last_motion_time

                if time_since_motion <= POST_MOTION_DELAY:
                    frame = picam2.capture_array()
                    detections = detect_persons(session, frame)
                    draw_overlay(picam2, detections)

                    if detections:
                        count = len(detections)
                        conf = max(d[1] for d in detections)
                        print(f"👤 {count} osoba detektirana | conf: {conf:.2f}")
                else:
                    draw_overlay(picam2, [])  # očisti overlay
                    print("💤 Nema pokreta – AI se gasi, kamera ostaje aktivna\n")
                    is_active = False

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n🛑 Gašenje...")
    finally:
        picam2.stop()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
