"""Quick manual check that the YOLO model loads and runs on a sample image.

Run from the project root: python scripts/smoke_test_detector.py
"""
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
model = YOLO(str(ROOT_DIR / "models" / "yolov8n.pt"))

results = model(str(ROOT_DIR / "data" / "images" / "car.jpg"), save=True)
print(results)
