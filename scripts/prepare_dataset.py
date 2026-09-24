"""Convert the Kaggle "Sri Lanka vehicle number plates" dataset (Pascal VOC XML)
into YOLO format with a train/val split.

Usage (from project root):
    python scripts/prepare_dataset.py [--raw data/raw] [--val-split 0.2]

Expects the extracted Kaggle files (image .jpg + .xml pairs) anywhere under --raw.
Writes data/plates/{images,labels}/{train,val} and data/plates/dataset.yaml.
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
CLASS_NAME = "license_plate"


def voc_to_yolo_lines(xml_path: Path, image_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0
    if width <= 0 or height <= 0:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        height, width = img.shape[:2]

    lines = []
    for obj in root.iter("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = max(float(box.findtext("xmin")), 0)
        ymin = max(float(box.findtext("ymin")), 0)
        xmax = min(float(box.findtext("xmax")), width)
        ymax = min(float(box.findtext("ymax")), height)
        if xmax <= xmin or ymax <= ymin:
            continue
        cx = (xmin + xmax) / 2 / width
        cy = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT_DIR / "data" / "raw")
    parser.add_argument("--out", type=Path, default=ROOT_DIR / "data" / "plates")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.raw.exists():
        raise SystemExit(f"{args.raw} not found. Extract the Kaggle download there first.")

    pairs = []
    for xml_path in sorted(args.raw.rglob("*.xml")):
        image_path = next(
            (xml_path.with_suffix(ext) for ext in (".jpg", ".jpeg", ".png", ".JPG")
             if xml_path.with_suffix(ext).exists()),
            None,
        )
        if image_path is None:
            continue
        lines = voc_to_yolo_lines(xml_path, image_path)
        if lines:
            pairs.append((image_path, lines))

    if not pairs:
        raise SystemExit(f"No usable image/XML pairs found under {args.raw}.")

    random.Random(args.seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_split))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    if args.out.exists():
        shutil.rmtree(args.out)
    for split, items in splits.items():
        img_dir = args.out / "images" / split
        lbl_dir = args.out / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for image_path, lines in items:
            shutil.copy2(image_path, img_dir / image_path.name)
            (lbl_dir / f"{image_path.stem}.txt").write_text("\n".join(lines) + "\n")

    (args.out / "dataset.yaml").write_text(
        f"path: {args.out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )
    print(f"Prepared {len(splits['train'])} train / {len(splits['val'])} val images in {args.out}")


if __name__ == "__main__":
    main()
