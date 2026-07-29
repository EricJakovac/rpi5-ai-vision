"""
Inference pipeline – background thread.
Koordinira kameru, detekciju i prepoznavanje lica.
"""

import sys
import threading
import time
import numpy as np
import psutil
import json
from pathlib import Path
from PIL import Image as PILImage
from datetime import datetime

from camera import CameraManager
from models import ModelManager
from schemas import Detection, Metrics

# Dodaj putanju do ai/recognition
sys.path.append(str(Path(__file__).parent.parent.parent / "ai" / "recognition"))
from clustering import UnknownPersonClustering

# ─── Konstante ───────────────────────────────────────────────────────────────

IMAGE_SIZE = 640
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
SIMILARITY_THRESHOLD = 0.5
CAM_WIDTH = 1280
CAM_HEIGHT = 720

DB_PATH = (
    Path(__file__).parent.parent.parent / "ai" / "recognition" / "face_database.json"
)


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


# ─── InferencePipeline ───────────────────────────────────────────────────────


class InferencePipeline:

    def __init__(self, camera: CameraManager, model_manager: ModelManager):
        self.camera = camera
        self.model_manager = model_manager

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        self._latest_frame = None
        self._latest_detections = []
        self._latest_metrics = None

        self._fps_times = []
        self._fps = 0.0
        self._inference_ms = 0.0
        self._detection_ms = 0.0
        self._recognition_ms = 0.0

        self.pir_active = False
        self._pir_triggers = 0

        # Face database
        self._persons = {}
        self._load_face_db()

        # InsightFace
        self._face_app = None
        self._init_insightface()

        # DBSCAN clustering
        self._clustering = UnknownPersonClustering(eps=0.4, min_samples=2)
        print("✅ DBSCAN clustering inicijaliziran")

        self._start_time = time.time()

    def _load_face_db(self):
        if DB_PATH.exists():
            with open(DB_PATH, "r") as f:
                db = json.load(f)
            for name, data in db["persons"].items():
                self._persons[name] = np.array(data["embedding"])
            print(f"✅ Baza lica: {len(self._persons)} osoba")

    def _init_insightface(self):
        try:
            from insightface.app import FaceAnalysis

            self._face_app = FaceAnalysis(
                name="buffalo_sc", providers=["CPUExecutionProvider"]
            )
            self._face_app.prepare(ctx_id=0, det_size=(640, 640))
            print("✅ InsightFace učitan")
        except Exception as e:
            print(f"⚠️  InsightFace nije dostupan: {e}")

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        print("✅ Inference pipeline pokrenut")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    # ─── Inference loop ──────────────────────────────────────────────────────

    def _inference_loop(self):
        while self._running:
            if not self.pir_active:
                frame = self.camera.get_frame()
                if frame is not None:
                    with self._lock:
                        self._latest_frame = frame
                        self._latest_detections = []
                time.sleep(0.1)
                continue

            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            t_total = time.perf_counter()

            t_det = time.perf_counter()
            person_detections = self._detect_persons(frame)
            detection_ms = (time.perf_counter() - t_det) * 1000

            t_rec = time.perf_counter()
            detections = self._recognize_persons(frame, person_detections)
            recognition_ms = (time.perf_counter() - t_rec) * 1000

            total_ms = (time.perf_counter() - t_total) * 1000

            now = time.time()
            self._fps_times.append(now)
            self._fps_times = [t for t in self._fps_times if now - t < 1.0]
            fps = len(self._fps_times)

            with self._lock:
                self._latest_frame = frame
                self._latest_detections = detections
                self._inference_ms = total_ms
                self._detection_ms = detection_ms
                self._recognition_ms = recognition_ms
                self._fps = fps
                self._latest_metrics = self._build_metrics(
                    fps, total_ms, detection_ms, recognition_ms
                )

    # ─── Detekcija ───────────────────────────────────────────────────────────

    def _detect_persons(self, frame) -> list:
        fmt = self.model_manager.get_format()
        if fmt == "tflite":
            return self._detect_tflite(frame)
        elif fmt == "onnx":
            return self._detect_onnx(frame)
        return []

    def _detect_tflite(self, frame) -> list:
        interpreter, input_details, output_details = (
            self.model_manager.get_interpreter()
        )
        if interpreter is None:
            return []

        input_dtype = input_details[0]["dtype"]
        img = (
            np.array(
                PILImage.fromarray(frame).resize((IMAGE_SIZE, IMAGE_SIZE)),
                dtype=np.float32,
            )
            / 255.0
        )
        img = np.expand_dims(img, axis=0)

        if input_dtype == np.int8:
            scale, zero_point = input_details[0]["quantization"]
            if scale != 0:
                img = (img / scale + zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        if input_dtype == np.int8:
            out_scale, out_zp = output_details[0]["quantization"]
            if out_scale != 0:
                output = (output.astype(np.float32) - out_zp) * out_scale

        return self._postprocess_yolov8(output, tflite=True)

    def _detect_onnx(self, frame) -> list:
        session = self.model_manager.get_session()
        if session is None:
            return []

        input_name = session.get_inputs()[0].name
        img = (
            np.array(
                PILImage.fromarray(frame).resize((IMAGE_SIZE, IMAGE_SIZE)),
                dtype=np.float32,
            )
            / 255.0
        )
        img = np.expand_dims(img.transpose(2, 0, 1), axis=0)

        output = session.run(None, {input_name: img})
        return self._postprocess_yolov8(output[0], tflite=False)

    def _postprocess_yolov8(self, output, tflite=False) -> list:
        predictions = output[0].T
        persons_mask = (np.argmax(predictions[:, 4:], axis=1) == 0) & (
            np.max(predictions[:, 4:], axis=1) > CONF_THRESHOLD
        )
        persons = predictions[persons_mask]
        if len(persons) == 0:
            return []

        boxes = persons[:, :4]
        scores = np.max(persons[:, 4:], axis=1)
        keep = nms(boxes, scores, IOU_THRESHOLD)
        return [(persons[i, :4], float(scores[i])) for i in keep]

    # ─── Prepoznavanje ───────────────────────────────────────────────────────

    def _recognize_persons(self, frame, person_detections) -> list:
        detections = []
        if not person_detections:
            return detections

        faces = []
        if self._face_app:
            try:
                faces = self._face_app.get(frame)
            except Exception:
                pass

        for bbox, conf in person_detections:
            name, face_score, cluster_id, face_obj = self._match_face(bbox, faces)

            if name:
                status = "known"
                cluster_label = None
            elif face_score >= 0.0:
                status = "unknown"
                # Pokušaj identificirati klaster
                if face_obj is not None:
                    cluster_id, cluster_score = self._clustering.identify_unknown(
                        face_obj.embedding
                    )
                    # Dodaj u clustering buffer ako prođe filtere
                    if self._clustering.should_add(
                        face_obj.embedding,
                        face_score,
                        float(face_obj.det_score),
                        self._persons,
                    ):
                        self._clustering.add_unknown(
                            face_obj.embedding, float(face_obj.det_score)
                        )
                cluster_label = cluster_id
            else:
                status = "no_face"
                cluster_label = None

            detections.append(
                Detection(
                    bbox=bbox.tolist(),
                    confidence=conf,
                    name=name,
                    face_score=face_score,
                    status=status,
                    cluster_id=cluster_label,
                )
            )

        return detections

    def _match_face(self, person_bbox, faces) -> tuple:
        """
        Pronađi lice unutar person bbox-a i identificiraj ga.
        Vraća: (name, face_score, cluster_id, face_obj)
        """
        if not faces:
            return None, -1.0, None, None

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
            return None, -1.0, None, None

        # Provjeri poznate osobe
        emb = best_face.embedding
        best_name = None
        best_score = -1.0

        for name, db_emb in self._persons.items():
            score = float(
                np.dot(emb, db_emb) / (np.linalg.norm(emb) * np.linalg.norm(db_emb))
            )
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= SIMILARITY_THRESHOLD:
            return best_name, best_score, None, best_face

        return None, best_score, None, best_face

    # ─── Metrike ─────────────────────────────────────────────────────────────

    def _get_temperature(self) -> float:
        try:
            temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
            if temp_path.exists():
                return float(temp_path.read_text().strip()) / 1000.0
        except Exception:
            pass
        return -1.0

    def _build_metrics(
        self, fps, inference_ms, detection_ms, recognition_ms
    ) -> Metrics:
        mem = psutil.virtual_memory()
        model_info = self.model_manager.get_current_info()

        return Metrics(
            fps=round(fps, 1),
            inference_ms=round(inference_ms, 1),
            detection_ms=round(detection_ms, 1),
            recognition_ms=round(recognition_ms, 1),
            cpu_percent=psutil.cpu_percent(),
            ram_used_mb=round(mem.used / 1024 / 1024, 1),
            ram_total_mb=round(mem.total / 1024 / 1024, 1),
            temperature_c=self._get_temperature(),
            pir_active=self.pir_active,
            active_model=model_info.get("name", "N/A"),
            num_persons=len(self._latest_detections),
            timestamp=datetime.now().isoformat(),
        )

    # ─── Getteri za API ──────────────────────────────────────────────────────

    def get_frame(self):
        with self._lock:
            return self._latest_frame

    def get_detections(self) -> list:
        with self._lock:
            return self._latest_detections.copy()

    def get_metrics(self) -> Metrics:
        with self._lock:
            if self._latest_metrics is None:
                return self._build_metrics(0, 0, 0, 0)
            return self._latest_metrics

    def get_clusters(self) -> list:
        """Vrati trenutne klastere nepoznatih osoba."""
        return self._clustering.get_clusters()

    def get_clustering_stats(self) -> dict:
        """Statistike clusteringa."""
        return self._clustering.get_stats()

    def reset_clustering(self):
        """Resetiraj clustering podatke."""
        self._clustering.reset()
