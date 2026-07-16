"""
Export fine-tuned modela u TFLite INT8/FP32 i ONNX FP32/INT8 formate. U nazivu se nalazi _ft.
"""

from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path("/workspace")
TRAINING_DIR = BASE_DIR / "ai" / "detection" / "training"
MODELS_DIR = BASE_DIR / "ai" / "detection" / "models"
CALIB_DATA = str(BASE_DIR / "datasets" / "Rpi5-ai-vision.yolov8" / "data.yaml")

FINETUNED_MODELS = [
    {"name": "yolov8n", "arch": "yolov8"},
    {"name": "yolov8s", "arch": "yolov8"},
    {"name": "yolov10n", "arch": "yolov10"},
    {"name": "yolo11n", "arch": "yolo11"},
    {"name": "rtdetr-l", "arch": "rtdetr"},
]


def export_model(model_info: dict):
    name = model_info["name"]
    arch = model_info["arch"]
    best_pt = TRAINING_DIR / name / "weights" / "best.pt"

    if not best_pt.exists():
        print(f"⚠️  best.pt ne postoji za {name}: {best_pt}")
        return False

    print(f"\n{'='*60}")
    print(f"Export: {name} → {name}_ft_*.tflite / {name}_ft_*.onnx")
    print(f"{'='*60}")

    model = YOLO(str(best_pt))

    # 1. ONNX FP32
    try:
        print(f"  → ONNX FP32...")
        out = model.export(format="onnx", imgsz=640, simplify=True)
        dst = MODELS_DIR / "onnx" / "fp32" / f"{name}_ft_fp32.onnx"
        dst.parent.mkdir(parents=True, exist_ok=True)
        Path(out).rename(dst)
        print(f"  ✅ {dst.name}")
    except Exception as e:
        print(f"  ❌ ONNX FP32: {e}")

    # 2. ONNX INT8 (ne za rtdetr)
    if arch != "rtdetr":
        try:
            print(f"  → ONNX INT8...")
            out = model.export(
                format="onnx", imgsz=640, simplify=True, int8=True, data=CALIB_DATA
            )
            dst = MODELS_DIR / "onnx" / "int8" / f"{name}_ft_int8.onnx"
            dst.parent.mkdir(parents=True, exist_ok=True)
            Path(out).rename(dst)
            print(f"  ✅ {dst.name}")
        except Exception as e:
            print(f"  ❌ ONNX INT8: {e}")

    # 3. TFLite FP32 (ne za rtdetr)
    if arch != "rtdetr":
        try:
            print(f"  → TFLite FP32...")
            out = model.export(format="tflite", imgsz=640)
            out_path = Path(out)
            # Pronađi .tflite fajl
            if out_path.is_dir():
                tflite = next(out_path.glob("*.tflite"), None)
            else:
                tflite = out_path if out_path.suffix == ".tflite" else None

            if tflite:
                dst = MODELS_DIR / "tflite" / "fp32" / f"{name}_ft_fp32.tflite"
                dst.parent.mkdir(parents=True, exist_ok=True)
                tflite.rename(dst)
                print(f"  ✅ {dst.name}")
        except Exception as e:
            print(f"  ❌ TFLite FP32: {e}")

    # 4. TFLite INT8 (ne za rtdetr)
    if arch != "rtdetr":
        try:
            print(f"  → TFLite INT8...")
            out = model.export(format="tflite", imgsz=640, int8=True, data=CALIB_DATA)
            out_path = Path(out)
            if out_path.is_dir():
                tflite = next(out_path.glob("*int8*.tflite"), None) or next(
                    out_path.glob("*.tflite"), None
                )
            else:
                tflite = out_path if out_path.suffix == ".tflite" else None

            if tflite:
                dst = MODELS_DIR / "tflite" / "int8" / f"{name}_ft_int8.tflite"
                dst.parent.mkdir(parents=True, exist_ok=True)
                tflite.rename(dst)
                print(f"  ✅ {dst.name}")
        except Exception as e:
            print(f"  ❌ TFLite INT8: {e}")

    return True


def main():
    print("🚀 Export fine-tuned modela (_ft oznaka)")
    print(f"Training dir: {TRAINING_DIR}")
    print(f"Output dir:   {MODELS_DIR}")

    print("\n📋 Provjera best.pt fajlova:")
    available = []
    for m in FINETUNED_MODELS:
        best_pt = TRAINING_DIR / m["name"] / "weights" / "best.pt"
        if best_pt.exists():
            size_mb = best_pt.stat().st_size / 1024 / 1024
            print(f"  ✅ {m['name']:12} → {size_mb:.1f}MB")
            available.append(m)
        else:
            print(f"  ⚠️  {m['name']:12} → best.pt ne postoji")

    if not available:
        print("❌ Nema dostupnih modela!")
        return

    print(f"\nExportiram {len(available)} modela...")
    for model_info in available:
        export_model(model_info)

    print(f"\n{'='*60}")
    print("✅ Export završen! Fajlovi:")
    for fmt in ["tflite/fp32", "tflite/int8", "onnx/fp32", "onnx/int8"]:
        folder = MODELS_DIR / fmt
        if folder.exists():
            ft_files = list(folder.glob("*_ft_*"))
            for f in sorted(ft_files):
                print(f"  {f.name}")

    print(f"\nℹ️  Kopiraj na RPi:")
    print(f"  scp ai/detection/models/tflite/int8/*_ft_* \\")
    print(
        f"      ericjakovac@192.168.1.234:~/Desktop/rpi5-ai-vision/ai/detection/models/tflite/int8/"
    )


if __name__ == "__main__":
    main()
