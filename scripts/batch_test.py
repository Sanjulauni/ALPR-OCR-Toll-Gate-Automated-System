"""Run the ALPR pipeline over a folder of images and report how it did.

Usage (from project root):
    python scripts/batch_test.py [--dir data/test] [--labels data/test/labels.csv]

Optional labels.csv (header: filename,plate) gives the true plate text per
image, e.g. "img1.jpg,WKM5141". With labels, exact-match accuracy is reported;
without, only detection and format-validity rates are.
Writes annotated images to outputs/batch/ and a CSV to outputs/batch_results.csv.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import ALPRPipeline  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def load_labels(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {row["filename"].strip(): normalize(row["plate"]) for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=ROOT_DIR / "data" / "test")
    parser.add_argument("--labels", type=Path, default=None)
    args = parser.parse_args()

    images = sorted(p for p in args.dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No images found in {args.dir}.")
    labels = load_labels(args.labels or args.dir / "labels.csv")

    pipeline = ALPRPipeline()
    out_dir = ROOT_DIR / "outputs" / "batch"
    rows = []
    for image_path in images:
        try:
            results = pipeline.process_and_annotate(image_path, out_dir / image_path.name)
        except Exception as exc:
            rows.append({"filename": image_path.name, "status": f"ERROR: {exc}"})
            continue

        expected = labels.get(image_path.name)
        if not results:
            rows.append({"filename": image_path.name, "status": "NO_DETECTION", "expected": expected or ""})
            continue

        best = max(results, key=lambda r: r.detection.confidence)
        row = {
            "filename": image_path.name,
            "status": "OK",
            "plate": best.plate_number,
            "raw": best.raw_text,
            "valid_format": best.is_valid,
            "det_conf": round(best.detection.confidence, 2),
            "ocr_conf": round(best.ocr_confidence, 2),
            "expected": expected or "",
        }
        if expected:
            row["correct"] = best.plate_number == expected
        rows.append(row)

    fields = ["filename", "status", "plate", "raw", "valid_format", "det_conf", "ocr_conf", "expected", "correct"]
    csv_path = ROOT_DIR / "outputs" / "batch_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    detected = [r for r in rows if r["status"] == "OK"]
    valid = [r for r in detected if r["valid_format"]]
    print(f"\n{'file':<28}{'plate':<12}{'valid':<7}{'det':<6}{'ocr':<6}{'expected':<12}result")
    for r in rows:
        if r["status"] != "OK":
            print(f"{r['filename']:<28}{r['status']}")
            continue
        verdict = "" if "correct" not in r else ("PASS" if r["correct"] else "FAIL")
        print(f"{r['filename']:<28}{r['plate']:<12}{str(r['valid_format']):<7}"
              f"{r['det_conf']:<6}{r['ocr_conf']:<6}{r['expected']:<12}{verdict}")

    print(f"\nImages:            {total}")
    print(f"Plate detected:    {len(detected)}/{total} ({len(detected) / total:.0%})")
    print(f"Valid SL format:   {len(valid)}/{total} ({len(valid) / total:.0%})")
    graded = [r for r in detected if "correct" in r]
    if graded:
        correct = sum(r["correct"] for r in graded)
        print(f"Exact match:       {correct}/{len(labels)} labelled ({correct / len(labels):.0%})")
    print(f"Results CSV:       {csv_path}")
    print(f"Annotated images:  {out_dir}")


if __name__ == "__main__":
    main()
