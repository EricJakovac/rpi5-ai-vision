"""
Live preview inference s kamerom i prikazom detekcija.
Pokreći na RPi-u s aktivnim displayem.
"""

from picamera2 import Picamera2
import onnxruntime as ort
import numpy as np
import cv2
import time
from pathlib import Path
from PIL import Image as PILImage

MODEL_PATH = Path("ai/detection/models/onnx/yolov8n_fp32.onnx")
IMAGE_SIZE = 640
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45


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


def detect_persons(output, conf_threshold, iou_threshold):
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
    return [(persons[i, :4], float(np.max(persons[i, 4:]))) for i in keep]


def draw_detections(frame, detections, fps, inference_ms):
    h, w = frame.shape[:2]

    for bbox, conf in detections:
        cx, cy, bw, bh = bbox
        x1 = int((cx - bw / 2) / IMAGE_SIZE * w)
        y1 = int((cy - bh / 2) / IMAGE_SIZE * h)
        x2 = int((cx + bw / 2) / IMAGE_SIZE * w)
        y2 = int((cy + bh / 2) / IMAGE_SIZE * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"Person {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.rectangle(frame, (0, 0), (280, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Inference: {inference_ms:.0f}ms", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Osobe: {len(detections)}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame


def main():
    print("=== Live Inference Preview ===")
    print("Pritisni 'q' za izlaz")

    # Kamera – puna rezolucija bez zumiranja
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    # Zamrzni AWB gains za konzistentne boje
    metadata = picam2.capture_metadata()
    gains = metadata["ColourGains"]
    picam2.set_controls({
        "AwbEnable": False,
        "ColourGains": gains
    })
    time.sleep(0.5)
    print("✅ Kamera pokrenuta")

    # Model
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    print(f"✅ Model učitan: {MODEL_PATH.name}")

    fps = 0
    frame_count = 0
    fps_start = time.time()

    while True:
        # Capture originalni frame za prikaz
        frame = picam2.capture_array()
        display_frame = frame.copy()

        # Resize samo za inference
        img_resized = np.array(
            PILImage.fromarray(frame).resize((IMAGE_SIZE, IMAGE_SIZE)),
            dtype=np.float32
        )
        img_resized /= 255.0
        inp = np.expand_dims(img_resized.transpose(2, 0, 1), 0)

        # Inference
        t0 = time.perf_counter()
        output = session.run(None, {input_name: inp})
        inference_ms = (time.perf_counter() - t0) * 1000

        # Detekcije
        detections = detect_persons(output[0], CONF_THRESHOLD, IOU_THRESHOLD)

        # FPS
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - fps_start)
            fps_start = time.time()

        # Prikaz
        display = draw_detections(display_frame, detections, fps, inference_ms)
        display_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        cv2.imshow("RPi5 - Person Detection", display_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()
    print("Završeno.")


if __name__ == "__main__":
    main()