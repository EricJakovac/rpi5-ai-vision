"""
Fine-tuning svih YOLO modela na person datasetu.
Korištenje:
    python3 ai/detection/scripts/fine_tune.py
    python3 ai/detection/scripts/fine_tune.py --model yolov8n
    python3 ai/detection/scripts/fine_tune.py --epochs 50 --batch 64
"""

import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
import yaml

# ─── Putanje ─────────────────────────────────────────────────────────────────

BASE_DIR = Path("/workspace")
DATASET_YAML = BASE_DIR / "datasets" / "Rpi5-ai-vision.yolov8" / "data.yaml"
RESULTS_DIR = BASE_DIR / "ai" / "detection" / "training"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Modeli za fine-tuning ───────────────────────────────────────────────────

MODELS = [
    {"name": "yolov8n", "weights": "yolov8n.pt"},
    {"name": "yolov8s", "weights": "yolov8s.pt"},
    {"name": "yolov10n", "weights": "yolov10n.pt"},
    {"name": "yolo11n", "weights": "yolo11n.pt"},
    {"name": "rtdetr-l", "weights": "rtdetr-l.pt"},
]


# ─── Provjere ────────────────────────────────────────────────────────────────


def check_gpu() -> str:
    """Provjeri dostupnost GPU-a i vrati device string."""
    print("\n🔍 Provjera GPU-a...")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  ✅ GPU: {gpu_name}")
        print(f"  ✅ VRAM: {vram_gb:.1f} GB")
        print(f"  ✅ CUDA/ROCm: {torch.version.cuda or 'ROCm'}")
        return "0"
    else:
        print("  ⚠️  GPU nije dostupan – koristim CPU")
        print("  ℹ️  Treniranje će biti puno sporije!")
        return "cpu"


def suggest_batch_size(device: str) -> int:
    """Predloži batch size na temelju dostupnog VRAM-a."""
    if device == "cpu":
        return 8

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    if vram_gb >= 20:
        batch = 64
    elif vram_gb >= 12:
        batch = 32
    elif vram_gb >= 8:
        batch = 16
    else:
        batch = 8

    print(f"  ℹ️  Preporučeni batch size za {vram_gb:.0f}GB VRAM: {batch}")
    return batch


def check_dataset(yaml_path: Path) -> bool:
    if not yaml_path.exists():
        print(f"❌ Dataset YAML ne postoji: {yaml_path}")
        return False

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    dataset_dir = yaml_path.parent

    splits = {
        "train": "train",
        "valid": "valid",
        "test": "test",
    }

    for split_name, folder_name in splits.items():
        img_dir = dataset_dir / folder_name / "images"

        if img_dir.exists():
            n_imgs = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
            if n_imgs > 0:
                print(f"  ✅ {split_name}: {n_imgs} slika ({img_dir})")
            else:
                print(f"  ⚠️  {split_name}: folder postoji ali nema slika ({img_dir})")
        else:
            print(f"  ❌ {split_name} folder ne postoji: {img_dir}")
            return False

    return True


def fix_dataset_yaml() -> str:
    """Popravi putanje u data.yaml za Docker okruženje."""
    with open(DATASET_YAML) as f:
        data = yaml.safe_load(f)

    dataset_dir = DATASET_YAML.parent
    data["train"] = str(dataset_dir / "train" / "images")
    data["val"] = str(dataset_dir / "valid" / "images")
    data["test"] = str(dataset_dir / "test" / "images")
    data["nc"] = 1
    data["names"] = ["person"]

    fixed_yaml = RESULTS_DIR / "data_fixed.yaml"
    with open(fixed_yaml, "w") as f:
        yaml.dump(data, f)

    print(f"  ✅ Popravljen data.yaml: {fixed_yaml}")
    return str(fixed_yaml)


# ─── Treniranje ───────────────────────────────────────────────────────────────


def train_model(
    model_info: dict, data_yaml: str, epochs: int, batch: int, device: str
) -> bool:
    name = model_info["name"]
    weights = model_info["weights"]

    print(f"\n{'='*60}")
    print(f"🚀 Treniram: {name}")
    print(f"   Weights:  {weights}")
    print(f"   Epochs:   {epochs} (early stopping patience=20)")
    print(f"   Batch:    {batch}")
    print(f"   Device:   {device}")
    print(f"{'='*60}")

    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"❌ Ne mogu učitati model {weights}: {e}")
        return False

    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=batch,
            device=device,
            workers=8,
            patience=20,  # early stopping
            save=True,
            save_period=-1,  # spremi SAMO best.pt
            plots=True,
            verbose=True,
            exist_ok=True,
            project=str(RESULTS_DIR),
            name=name,
            # Augmentacija
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
        )

        best_map = results.results_dict.get("metrics/mAP50(B)", "N/A")
        print(f"\n✅ {name} završeno!")
        print(f"   Best mAP@0.5: {best_map}")
        print(f"   Best weights: {RESULTS_DIR / name / 'weights' / 'best.pt'}")
        return True

    except Exception as e:
        print(f"❌ Greška pri treniranju {name}: {e}")
        return False


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning YOLO modela")
    parser.add_argument(
        "--model",
        default=None,
        choices=["yolov8n", "yolov8s", "yolov10n", "yolo11n", "rtdetr-l"],
        help="Specifični model za treniranje",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Broj epoha (default: 100)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size (default: auto prema VRAM-u)",
    )
    parser.add_argument(
        "--device", default=None, help="Device: 0 za GPU, cpu za CPU (default: auto)"
    )
    args = parser.parse_args()

    print("🚀 Fine-tuning – Person Detection Dataset")
    print(f"   Dataset: {DATASET_YAML}")

    # ─── Provjera GPU ────────────────────────────────────────────────────────
    device = args.device if args.device else check_gpu()

    # ─── Batch size ──────────────────────────────────────────────────────────
    batch = args.batch if args.batch else suggest_batch_size(device)
    print(f"   Batch:   {batch}")
    print(f"   Epochs:  {args.epochs}")

    # ─── Provjera dataseta ───────────────────────────────────────────────────
    print("\n🔍 Provjera dataseta...")
    if not check_dataset(DATASET_YAML):
        return

    data_yaml = fix_dataset_yaml()

    # ─── Odabir modela ───────────────────────────────────────────────────────
    models_to_train = MODELS
    if args.model:
        models_to_train = [m for m in MODELS if m["name"] == args.model]

    print(f"\n📋 Modeli za treniranje ({len(models_to_train)}):")
    for m in models_to_train:
        print(f"   - {m['name']} ({m['weights']})")

    # ─── Treniranje ──────────────────────────────────────────────────────────
    successful = []
    failed = []

    for model_info in models_to_train:
        success = train_model(model_info, data_yaml, args.epochs, batch, device)
        if success:
            successful.append(model_info["name"])
        else:
            failed.append(model_info["name"])

    # ─── Sažetak ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ SAŽETAK TRENIRANJA")
    print(f"{'='*60}")
    print(f"Uspješno: {len(successful)} modela")
    for name in successful:
        best_pt = RESULTS_DIR / name / "weights" / "best.pt"
        print(f"  ✅ {name} → {best_pt}")

    if failed:
        print(f"\nNeuspješno: {len(failed)} modela")
        for name in failed:
            print(f"  ❌ {name}")

    print(f"\n📁 Rezultati: {RESULTS_DIR}")
    print("ℹ️  Sljedeći korak: export modela u TFLite i ONNX formate")


if __name__ == "__main__":
    main()
