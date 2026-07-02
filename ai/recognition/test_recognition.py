"""
Test prepoznavanja lica iz kamere s live prikazom.
Koristi picamera2 QT preview + PIL overlay.
Pokreći na RPi-u.
"""

import insightface
from insightface.app import FaceAnalysis
from picamera2 import Picamera2, Preview
from PIL import Image as PILImage, ImageDraw, ImageFont
import numpy as np
import json
import time
from pathlib import Path

# Putanje
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "face_database.json"

SIMILARITY_THRESHOLD = 0.4
CAM_WIDTH = 1280
CAM_HEIGHT = 720


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
    return None, best_score


def draw_overlay(picam2: Picamera2, faces: list, persons: dict, inference_ms: float):
    """Crta overlay s bounding boxovima i imenima."""
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

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        name, score = identify_face(face.embedding, persons)

        if name:
            color = (0, 255, 0, 255)
            label = f"{name} ({score:.2f})"
        else:
            color = (255, 0, 0, 255)
            label = f"Nepoznato ({score:.2f})"

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label pozadina i tekst
        bbox_text = draw.textbbox((x1, y1 - 30), label, font=font)
        draw.rectangle(bbox_text, fill=(0, 0, 0, 180))
        draw.text((x1, y1 - 30), label, fill=color, font=font)

        # Det score ispod boxa
        draw.text(
            (x1, y2 + 5),
            f"det: {face.det_score:.2f}",
            fill=color,
            font=font_small
        )

    # Info overlay gore lijevo
    info = f"Inference: {inference_ms:.0f}ms | Lica: {len(faces)} | Threshold: {SIMILARITY_THRESHOLD}"
    bbox_info = draw.textbbox((10, 10), info, font=font_small)
    draw.rectangle(
        (bbox_info[0] - 5, bbox_info[1] - 5, bbox_info[2] + 5, bbox_info[3] + 5),
        fill=(0, 0, 0, 160)
    )
    draw.text((10, 10), info, fill=(0, 255, 0, 255), font=font_small)

    picam2.set_overlay(np.array(overlay))


def main():
    print("🚀 Test prepoznavanja lica – RPi 5")
    print("Pritisni Ctrl+C za izlaz\n")

    # Učitaj bazu
    persons = load_database()
    if not persons:
        return

    # Inicijaliziraj InsightFace
    print("Učitavam InsightFace model...")
    app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ Model učitan")

    # Inicijaliziraj kameru
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start_preview(Preview.QT)
    picam2.start()
    time.sleep(2)
    print("✅ Kamera pokrenuta")
    print(f"Threshold: {SIMILARITY_THRESHOLD}\n")

    try:
        while True:
            frame = picam2.capture_array()

            t0 = time.perf_counter()
            faces = app.get(frame)
            inference_ms = (time.perf_counter() - t0) * 1000

            # Crtaj overlay
            draw_overlay(picam2, faces, persons, inference_ms)

            # Terminal ispis
            if len(faces) == 0:
                print(f"[{inference_ms:.0f}ms] Nema lica")
            else:
                for face in faces:
                    name, score = identify_face(face.embedding, persons)
                    if name:
                        print(f"[{inference_ms:.0f}ms] ✅ {name} (sličnost: {score:.3f})")
                    else:
                        print(f"[{inference_ms:.0f}ms] ❓ Nepoznata osoba (max: {score:.3f})")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n🛑 Gašenje...")
    finally:
        picam2.stop()


if __name__ == "__main__":
    main()