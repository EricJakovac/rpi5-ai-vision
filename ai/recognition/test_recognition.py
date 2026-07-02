"""
Test prepoznavanja lica iz kamere s live prikazom.
"""

import insightface
from insightface.app import FaceAnalysis
from picamera2 import Picamera2
import numpy as np
import json
import time
import cv2
from pathlib import Path

# Putanje
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "face_database.json"

SIMILARITY_THRESHOLD = 0.5


def load_database() -> dict:
    if not DB_PATH.exists():
        print(f"❌ Baza ne postoji: {DB_PATH}")
        return {}
    with open(DB_PATH, "r") as f:
        db = json.load(f)
    persons = {}
    for name, data in db["persons"].items():
        persons[name] = np.array(data["embedding"])
    print(f"✅ Baza učitana – {len(persons)} osoba: {list(persons.keys())}")
    return persons


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
    else:
        return None, best_score


def draw_results(
    frame: np.ndarray, faces: list, persons: dict, inference_ms: float
) -> np.ndarray:
    display = frame.copy()

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        name, score = identify_face(face.embedding, persons)

        if name:
            color = (0, 255, 0)  # zelena = poznata osoba
            label = f"{name} ({score:.2f})"
        else:
            color = (0, 0, 255)  # crvena = nepoznata osoba
            label = f"Nepoznato ({score:.2f})"

        # Bounding box
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        # Label pozadina
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(display, (x1, y1 - lh - 8), (x1 + lw, y1), color, -1)
        cv2.putText(
            display, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

        # Det score
        cv2.putText(
            display,
            f"det: {face.det_score:.2f}",
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    # Info overlay
    cv2.rectangle(display, (0, 0), (320, 55), (0, 0, 0), -1)
    cv2.putText(
        display,
        f"Inference: {inference_ms:.0f}ms",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        display,
        f"Lica: {len(faces)} | Threshold: {SIMILARITY_THRESHOLD}",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    return display


def main():
    print("🚀 Test prepoznavanja lica – RPi 5")
    print("Pritisni 'q' za izlaz\n")

    persons = load_database()
    if not persons:
        return

    print("Učitavam InsightFace model...")
    app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ Model učitan")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    print("✅ Kamera pokrenuta\n")

    try:
        while True:
            frame = picam2.capture_array()

            t0 = time.perf_counter()
            faces = app.get(frame)
            inference_ms = (time.perf_counter() - t0) * 1000

            # Terminal ispis
            if len(faces) == 0:
                print(f"[{inference_ms:.0f}ms] Nema lica")
            else:
                for face in faces:
                    name, score = identify_face(face.embedding, persons)
                    if name:
                        print(
                            f"[{inference_ms:.0f}ms] ✅ {name} (sličnost: {score:.3f})"
                        )
                    else:
                        print(
                            f"[{inference_ms:.0f}ms] ❓ Nepoznata osoba (max: {score:.3f})"
                        )

            # Prikaz
            display = draw_results(frame, faces, persons, inference_ms)
            display_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
            cv2.imshow("Face Recognition – RPi5", display_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n🛑 Gašenje...")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
