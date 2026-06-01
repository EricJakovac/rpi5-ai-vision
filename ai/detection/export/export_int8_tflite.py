"""
Export modela u TFLite INT8 format.
"""
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODELS_TFLITE_INT8 = BASE_DIR / "models" / "tflite" / "int8"
MODELS_TFLITE_INT8.mkdir(parents=True, exist_ok=True)

MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov10n.pt",
]

IMAGE_SIZE = 640


def export_tflite_int8(model_name: str):
    print(f"\n{'='*50}")
    print(f"Exportam TFLite INT8: {model_name}")
    print(f"{'='*50}")

    dst = MODELS_TFLITE_INT8 / f"{model_name.replace('.pt', '')}_int8.tflite"

    if dst.exists():
        print(f"⏭️  Već postoji: {dst.name} – preskačem")
        return

    model = YOLO(model_name)

    print("\nExport TFLite INT8...")
    tflite_path = model.export(
        format="tflite",
        imgsz=IMAGE_SIZE,
        simplify=True,
        int8=True,
    )

    src = Path(str(tflite_path))
    src.rename(dst)
    print(f"✅ Spremljeno: {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    for model_name in MODELS:
        export_tflite_int8(model_name)

    print(f"\n✅ INT8 TFLite Export završen!")
    print(f"\nFajlovi u {MODELS_TFLITE_INT8}:")
    for f in sorted(MODELS_TFLITE_INT8.glob("*.tflite")):
        print(f"  {f.name} – {f.stat().st_size / 1024 / 1024:.1f} MB")