"""
Test prepoznavanja lica iz kamere.
Uspoređuje lice s bazom i ispisuje rezultat.
"""

import insightface
from insightface.app import FaceAnalysis
from picamera2 import Picamera2
import numpy as np
import json
import time
from pathlib import Path

# Putanje
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "face_database.json"

# Threshold za prepoznavanje
SIMILARITY_THRESHOLD = 0.8  # iznad ovoga = poznata osoba


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
    """Vrati (ime, sličnost) najbliže osobe ili (None, score) ako nepoznata."""
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


def main():
    print("🚀 Test prepoznavanja lica – RPi 5")

    # Učitaj bazu
    persons = load_database()
    if not persons:
        return

    # Inicijaliziraj InsightFace
    print("\nUčitavam InsightFace model...")
    app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ Model učitan")

    # Inicijaliziraj kameru
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    print("✅ Kamera pokrenuta")
    print(f"\nThreshold: {SIMILARITY_THRESHOLD}")
    print("Pritisni Ctrl+C za izlaz\n")

    try:
        while True:
            frame = picam2.capture_array()

            t0 = time.perf_counter()
            faces = app.get(frame)
            elapsed = (time.perf_counter() - t0) * 1000

            if len(faces) == 0:
                print(f"[{elapsed:.0f}ms] Nema lica")
            else:
                for i, face in enumerate(faces):
                    name, score = identify_face(face.embedding, persons)

                    if name:
                        print(f"[{elapsed:.0f}ms] ✅ {name} (sličnost: {score:.3f})")
                    else:
                        print(
                            f"[{elapsed:.0f}ms] ❓ Nepoznata osoba (max sličnost: {score:.3f})"
                        )

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n🛑 Gašenje...")
    finally:
        picam2.stop()


if __name__ == "__main__":
    main()
