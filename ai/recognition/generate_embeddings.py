"""
Generira face embeddings iz slika i sprema u JSON bazu.
"""

import insightface
from insightface.app import FaceAnalysis
import numpy as np
import json
import cv2
from pathlib import Path
from datetime import datetime

# Putanje
BASE_DIR = Path(__file__).parent
PICTURES_DIR = BASE_DIR / "pictures"
DB_PATH = BASE_DIR / "face_database.json"

# Automatsko mapiranje foldera → ime osobe
FOLDER_TO_PERSON = {
    "eric_svjetlo": "Eric Jakovac",
    "eric_kapa": "Eric Jakovac",
    "eric_mrak": "Eric Jakovac",
    "eric_naocale": "Eric Jakovac",
    "eric_hod": "Eric Jakovac",
    "ana_svjetlo": "Anamaria Stefanac",
    "ana_kapa": "Anamaria Stefanac",
    "ana_mrak": "Anamaria Stefanac",
    "ana_naocale": "Anamaria Stefanac",
    "ana_hod": "Anamaria Stefanac",
}


def load_database() -> dict:
    if DB_PATH.exists():
        with open(DB_PATH, "r") as f:
            return json.load(f)
    return {"persons": {}, "created": datetime.now().isoformat()}


def save_database(db: dict):
    db["updated"] = datetime.now().isoformat()
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)
    print(f"✅ Baza spremljena: {DB_PATH}")


def process_folder(app: FaceAnalysis, folder: Path) -> list:
    """Generiraj embeddings za sve slike u folderu."""
    images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))

    if not images:
        print(f"  ❌ Nema slika u {folder}")
        return []

    print(f"  Obrađujem {len(images)} slika...")

    embeddings = []
    no_face = 0
    multi_face = 0

    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)

        if len(faces) == 0:
            no_face += 1
            continue

        if len(faces) > 1:
            # Uzmi najveće lice
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )
            multi_face += 1

        face = faces[0]

        if face.det_score < 0.5:
            no_face += 1
            continue

        embeddings.append(face.embedding)

        if (i + 1) % 100 == 0:
            print(f"    → {i+1}/{len(images)} | embeddings: {len(embeddings)}")

    print(f"  ✅ Embeddings: {len(embeddings)}")
    print(f"  ⚠️  Bez lica:  {no_face}")
    print(f"  ⚠️  Više lica: {multi_face}")

    return embeddings


def main():
    print("🚀 Generiranje face embeddings")
    print(f"Slike iz: {PICTURES_DIR}\n")

    if not PICTURES_DIR.exists():
        print(f"❌ Folder ne postoji: {PICTURES_DIR}")
        return

    # Pronađi sve foldere
    folders = sorted([f for f in PICTURES_DIR.iterdir() if f.is_dir()])
    if not folders:
        print("❌ Nema foldera sa slikama")
        return

    print("Pronađeni folderi i mapiranje:")
    for f in folders:
        img_count = len(list(f.glob("*.jpg")))
        person = FOLDER_TO_PERSON.get(f.name, "❌ NIJE MAPIRANO")
        print(f"  {f.name:<20} → {person} ({img_count} slika)")

    print("\nPokrećem generiranje embeddings...\n")

    # Inicijaliziraj InsightFace
    print("Učitavam InsightFace model...")
    app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ Model učitan\n")

    # Prikupi embeddings po osobi
    person_embeddings = {}

    for folder in folders:
        person_name = FOLDER_TO_PERSON.get(folder.name)

        if not person_name:
            print(f"⚠️  Preskačem {folder.name} – nije mapirano")
            continue

        print(f"\n{'='*55}")
        print(f"Folder: {folder.name} → {person_name}")
        print(f"{'='*55}")

        embeddings = process_folder(app, folder)

        if not embeddings:
            print(f"⚠️  Nema embeddings za {folder.name}")
            continue

        if person_name not in person_embeddings:
            person_embeddings[person_name] = []
        person_embeddings[person_name].extend(embeddings)

    # Spremi u bazu
    print(f"\n{'='*55}")
    print("SPREMANJE U BAZU")
    print(f"{'='*55}")

    db = load_database()

    for person_name, embeddings in person_embeddings.items():
        print(f"\nOsoba: {person_name}")
        print(f"  Ukupno embeddings: {len(embeddings)}")

        # Normaliziraj i usrjednji
        normalized = []
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            if norm > 0:
                normalized.append(emb / norm)

        avg_embedding = np.mean(normalized, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        db["persons"][person_name] = {
            "embedding": avg_embedding.tolist(),
            "num_images": len(embeddings),
            "registered": datetime.now().isoformat(),
        }
        print(f"  ✅ Prosječni embedding izračunat iz {len(embeddings)} slika")

    save_database(db)

    print(f"\n{'='*55}")
    print("SAŽETAK")
    print(f"{'='*55}")
    for name, data in db["persons"].items():
        print(f"  ✅ {name}: {data['num_images']} slika")
    print(f"\n✅ Baza gotova: {DB_PATH}")


if __name__ == "__main__":
    main()
