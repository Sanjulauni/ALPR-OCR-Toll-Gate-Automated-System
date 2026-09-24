"""Fine-tune YOLOv8 on the prepared plate dataset and install the result as
models/plate_detector.pt.

Usage (from project root, after scripts/prepare_dataset.py):
    python scripts/train.py [--epochs 50] [--batch 8]
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT_DIR / "data" / "plates" / "dataset.yaml")
    parser.add_argument("--base", type=Path, default=ROOT_DIR / "models" / "yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found. Run scripts/prepare_dataset.py first.")

    model = YOLO(str(args.base))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(ROOT_DIR / "runs"),
        name="plate",
        exist_ok=True,
        workers=0,
    )

    best = ROOT_DIR / "runs" / "plate" / "weights" / "best.pt"
    target = ROOT_DIR / "models" / "plate_detector.pt"
    shutil.copy2(best, target)
    print(f"Saved fine-tuned weights to {target}")


if __name__ == "__main__":
    main()
