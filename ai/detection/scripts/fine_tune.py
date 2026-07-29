"""
Fine-tuning svih YOLO modela na person datasetu.
Korištenje:
    python3 ai/detection/scripts/fine_tune.py
    python3 ai/detection/scripts/fine_tune.py --model yolov8n
    python3 ai/detection/scripts/fine_tune.py --epochs 100 --batch 64
"""

import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
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
    {"name": "yolov8n_v2",  "weights": "yolov8n.pt"},
    {"name": "yolov8s_v2", "weights": "yolov8s.pt"},
    {"name": "yolov10n_v2", "weights": "yolov10n.pt"},
    {"name": "yolo11n_v2",  "weights": "yolo11n.pt"},
    {"name": "rtdetr-l_v2", "weights": "rtdetr-l.pt"},
]


# ─── Provjere ────────────────────────────────────────────────────────────────

def check_gpu() -> str:
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
        return "cpu"


def suggest_batch_size(device: str) -> int:
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
    for split_name, folder_name in [("train", "train"), ("valid", "valid"), ("test", "test")]:
        img_dir = dataset_dir / folder_name / "images"
        if img_dir.exists():
            n_imgs = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
            if n_imgs > 0:
                print(f"  ✅ {split_name}: {n_imgs} slika")
            else:
                print(f"  ⚠️  {split_name}: nema slika")
        else:
            print(f"  ❌ {split_name} ne postoji: {img_dir}")
            return False
    return True


def fix_dataset_yaml() -> str:
    with open(DATASET_YAML) as f:
        data = yaml.safe_load(f)
    dataset_dir = DATASET_YAML.parent
    data["train"] = str(dataset_dir / "train" / "images")
    data["val"]   = str(dataset_dir / "valid" / "images")
    data["test"]  = str(dataset_dir / "test"  / "images")
    data["nc"]    = 1
    data["names"] = ["person"]
    fixed_yaml = RESULTS_DIR / "data_fixed.yaml"
    with open(fixed_yaml, "w") as f:
        yaml.dump(data, f)
    print(f"  ✅ Popravljen data.yaml: {fixed_yaml}")
    return str(fixed_yaml)


# ─── Generiranje statistike ───────────────────────────────────────────────────

def generate_results_png(name: str):
    """Generiraj results.png iz results.csv."""
    results_path = RESULTS_DIR / name
    csv_path = results_path / "results.csv"

    if not csv_path.exists():
        print(f"  ⚠️  results.csv ne postoji, preskačem results.png")
        return

    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'{name} Training Results', fontsize=14)

        metrics = [
            ('train/box_loss',        'Train Box Loss'),
            ('train/cls_loss',        'Train Cls Loss'),
            ('train/dfl_loss',        'Train DFL Loss'),
            ('metrics/precision(B)',  'Precision'),
            ('metrics/recall(B)',     'Recall'),
            ('val/box_loss',          'Val Box Loss'),
            ('val/cls_loss',          'Val Cls Loss'),
            ('val/dfl_loss',          'Val DFL Loss'),
            ('metrics/mAP50(B)',      'mAP@0.5'),
            ('metrics/mAP50-95(B)',   'mAP@0.5:0.95'),
        ]

        for ax, (col, title) in zip(axes.flat, metrics):
            if col in df.columns:
                ax.plot(df['epoch'], df[col], 'b-o', markersize=2, label='results')
                ax.plot(df['epoch'], df[col].rolling(5, min_periods=1).mean(),
                        'orange', linestyle='--', label='smooth')
                ax.set_title(title)
                ax.set_xlabel('Epoch')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(results_path / 'results.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ results.png sačuvan")
    except Exception as e:
        print(f"  ⚠️  Greška pri generiranju results.png: {e}")


def generate_val_plots(name: str, data_yaml: str, device: str):
    """Generiraj validacijske grafove (confusion matrix, PR curve itd.)."""
    best_pt = RESULTS_DIR / name / "weights" / "best.pt"

    if not best_pt.exists():
        print(f"  ⚠️  best.pt ne postoji, preskačem validaciju")
        return

    try:
        print(f"  🔍 Generiram validacijske grafove...")
        val_model = YOLO(str(best_pt))
        val_results = val_model.val(
            data=data_yaml,
            split='val',
            imgsz=640,
            conf=0.25,
            iou=0.45,
            device=device,
            plots=True,
            project=str(RESULTS_DIR),
            name=name,
            exist_ok=True,
            verbose=False,
        )
        map50    = val_results.results_dict.get('metrics/mAP50(B)', 0)
        map5095  = val_results.results_dict.get('metrics/mAP50-95(B)', 0)
        prec     = val_results.results_dict.get('metrics/precision(B)', 0)
        recall   = val_results.results_dict.get('metrics/recall(B)', 0)

        print(f"  ✅ Validacijski grafovi sačuvani")
        print(f"     mAP@0.5:      {map50:.4f}")
        print(f"     mAP@0.5:0.95: {map5095:.4f}")
        print(f"     Precision:    {prec:.4f}")
        print(f"     Recall:       {recall:.4f}")

    except Exception as e:
        print(f"  ⚠️  Greška pri validaciji: {e}")


# ─── Treniranje ───────────────────────────────────────────────────────────────

def train_model(model_info: dict, data_yaml: str, epochs: int,
                batch: int, device: str) -> bool:
    name    = model_info["name"]
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
            patience=20,
            save=True,
            save_period=-1,
            plots=True,
            verbose=True,
            exist_ok=True,
            project=str(RESULTS_DIR),
            name=name,
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
        print(f"\n✅ {name} treniranje završeno!")
        print(f"   Best mAP@0.5: {best_map}")

    except Exception as e:
        print(f"❌ Greška pri treniranju {name}: {e}")
        # Čak i ako je greška, pokušaj generirati statistiku
        print(f"   Pokušavam generirati statistiku iz dostupnih podataka...")

    # Uvijek generiraj statistiku nakon treniranja (ili greške)
    print(f"\n📊 Generiranje statistike za {name}...")
    generate_results_png(name)
    generate_val_plots(name, data_yaml, device)

    # Provjeri je li best.pt sačuvan
    best_pt = RESULTS_DIR / name / "weights" / "best.pt"
    if best_pt.exists():
        print(f"   ✅ Best weights: {best_pt}")
        return True
    else:
        print(f"   ❌ best.pt ne postoji!")
        return False


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning YOLO modela")
    parser.add_argument("--model", default=None,
                        help="Specifični model za treniranje")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    print("🚀 Fine-tuning – Person Detection Dataset")
    print(f"   Dataset: {DATASET_YAML}")

    device = args.device if args.device else check_gpu()
    batch  = args.batch  if args.batch  else suggest_batch_size(device)
    print(f"   Batch:   {batch}")
    print(f"   Epochs:  {args.epochs}")

    print("\n🔍 Provjera dataseta...")
    if not check_dataset(DATASET_YAML):
        return

    data_yaml = fix_dataset_yaml()

    models_to_train = MODELS
    if args.model:
        models_to_train = [m for m in MODELS if m["name"] == args.model]

    print(f"\n📋 Modeli za treniranje ({len(models_to_train)}):")
    for m in models_to_train:
        print(f"   - {m['name']} ({m['weights']})")

    successful, failed = [], []
    for model_info in models_to_train:
        success = train_model(model_info, data_yaml, args.epochs, batch, device)
        if success:
            successful.append(model_info["name"])
        else:
            failed.append(model_info["name"])

    print(f"\n{'='*60}")
    print(f"✅ SAŽETAK TRENIRANJA")
    print(f"{'='*60}")
    print(f"Uspješno: {len(successful)} modela")
    for name in successful:
        print(f"  ✅ {name}")
    if failed:
        print(f"\nNeuspješno: {len(failed)} modela")
        for name in failed:
            print(f"  ❌ {name}")
    print(f"\n📁 Rezultati: {RESULTS_DIR}")


if __name__ == "__main__":
    main()