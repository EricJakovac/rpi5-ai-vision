"""
ONNX INT8 kvantizacija pomoću onnxruntime.
"""
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODELS_FP32 = BASE_DIR / "models" / "onnx" / "fp32"
MODELS_INT8 = BASE_DIR / "models" / "onnx" / "int8"
MODELS_INT8.mkdir(parents=True, exist_ok=True)

MODELS = [
    "yolov8n_fp32.onnx",
    "yolov8s_fp32.onnx",
    "yolov10n_fp32.onnx",
    "rtdetr-l_fp32.onnx",
]


def quantize_to_int8(model_name: str):
    print(f"\n{'='*50}")
    print(f"Kvantiziram INT8: {model_name}")
    print(f"{'='*50}")

    src = MODELS_FP32 / model_name
    dst = MODELS_INT8 / model_name.replace("_fp32", "_int8")

    if dst.exists():
        print(f"⏭️  Već postoji: {dst.name} – preskačem")
        return

    if not src.exists():
        print(f"❌ Ne postoji FP32 model: {src}")
        return

    print(f"Ulaz:  {src.name} ({src.stat().st_size / 1024 / 1024:.1f} MB)")

    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QUInt8,
    )

    print(f"✅ Spremljeno: {dst.name} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")
    reduction = (1 - dst.stat().st_size / src.stat().st_size) * 100
    print(f"   Smanjenje veličine: {reduction:.1f}%")


if __name__ == "__main__":
    for model_name in MODELS:
        quantize_to_int8(model_name)

    print(f"\n✅ INT8 kvantizacija završena!")
    print(f"\nFajlovi u {MODELS_INT8}:")
    for f in sorted(MODELS_INT8.glob("*.onnx")):
        print(f"  {f.name} – {f.stat().st_size / 1024 / 1024:.1f} MB")