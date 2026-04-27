"""
Export YOLOv8 modela u ONNX format.
Pokreći na Ubuntuu.
TFLite export se radi posebno na RPi-u.
"""

from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODELS_ONNX = BASE_DIR / "models" / "onnx"
MODELS_ONNX.mkdir(parents=True, exist_ok=True)

MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
]

IMAGE_SIZE = 640

def export_onnx(model_name: str):
    print(f"\n{'='*50}")
    print(f"Exportam: {model_name}")
    print(f"{'='*50}")

    model = YOLO(model_name)

    print("\nExport ONNX FP32...")
    onnx_path = model.export(
        format="onnx",
        imgsz=IMAGE_SIZE,
        simplify=True,
        dynamic=False,
    )

    src = Path(onnx_path)
    dst = MODELS_ONNX / f"{model_name.replace('.pt', '')}_fp32.onnx"
    src.rename(dst)
    print(f"✅ Spremljeno: {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    for model_name in MODELS:
        export_onnx(model_name)

    print(f"\n✅ Export završen!")
    print(f"Modeli u: {MODELS_ONNX}")
    print("\nFajlovi:")
    for f in MODELS_ONNX.glob("*.onnx"):
        print(f"  {f.name} – {f.stat().st_size / 1024 / 1024:.1f} MB")