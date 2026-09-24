"""YOLOv8-based license plate detection."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from . import config


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str


class PlateDetector:
    """Wraps a YOLOv8 model and returns clean Detection objects for downstream stages."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        confidence_threshold: Optional[float] = None,
        target_classes: Optional[List[str]] = None,
    ):
        model_path = Path(model_path) if model_path else (
            config.PLATE_DETECTOR_MODEL
            if config.PLATE_DETECTOR_MODEL.exists()
            else config.FALLBACK_DETECTOR_MODEL
        )
        if not model_path.exists():
            raise FileNotFoundError(
                f"No detection model found at {model_path}. Train/download a "
                "YOLOv8 license-plate model to models/plate_detector.pt, or place "
                "a base model at models/yolov8n.pt for smoke testing."
            )
        self.model_path = model_path
        self.model = YOLO(str(model_path))
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else config.DETECTION_CONFIDENCE_THRESHOLD
        )
        self.target_classes = target_classes

    def detect(self, image: np.ndarray) -> List[Detection]:
        prediction = self.model(image, verbose=False)[0]
        detections: List[Detection] = []
        for box in prediction.boxes:
            confidence = float(box.conf[0])
            if confidence < self.confidence_threshold:
                continue
            class_id = int(box.cls[0])
            class_name = prediction.names.get(class_id, str(class_id))
            if self.target_classes and class_name not in self.target_classes:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            detections.append(Detection((x1, y1, x2, y2), confidence, class_id, class_name))
        return detections

    @staticmethod
    def crop(image: np.ndarray, detection: Detection) -> np.ndarray:
        x1, y1, x2, y2 = detection.bbox
        return image[y1:y2, x1:x2]
