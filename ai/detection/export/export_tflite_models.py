"""
Export YOLOv8 modela u TFLite format.
"""

from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODELS_TFLITE = BASE_DIR / "models" / "tflite"
MODELS_TFLITE.mkdir(parents=True, exist_ok=True)

MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov10n.pt",
]

IMAGE_SIZE = 640

def export_tflite(model_name: str):
    print(f"\n{'='*50}")
    print(f"Exportam: {model_name}")
    print(f"{'='*50}")

    model = YOLO(model_name)

    print("\nExport TFLite FP32...")
    tflite_path = model.export(
        format="tflite",
        imgsz=IMAGE_SIZE,
        simplify=True,
    )

    src = Path(str(tflite_path))
    dst = MODELS_TFLITE / f"{model_name.replace('.pt', '')}_fp32.tflite"
    src.rename(dst)
    print(f"✅ Spremljeno: {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    for model_name in MODELS:
        export_tflite(model_name)

    print(f"\n✅ Export završen!")
    print(f"Modeli u: {MODELS_TFLITE}")
    print("\nFajlovi:")
    for f in MODELS_TFLITE.glob("*.tflite"):
        print(f"  {f.name} – {f.stat().st_size / 1024 / 1024:.1f} MB")