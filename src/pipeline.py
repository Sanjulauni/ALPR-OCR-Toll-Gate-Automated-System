"""End-to-end ALPR pipeline: detection -> preprocessing -> OCR -> validation.

Usage:
    python -m src.pipeline data/images/car.jpg
    python -m src.pipeline data/images/car.jpg --output outputs/car_annotated.jpg
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from . import config
from .detector import Detection, PlateDetector
from .ocr_engine import PlateOCR
from .preprocess import preprocess_plate
from .utils import draw_detection, get_logger, save_image
from .validator import validate_plate

logger = get_logger(__name__)


@dataclass
class PlateReadResult:
    detection: Detection
    raw_text: str
    plate_number: str
    is_valid: bool
    ocr_confidence: float


class ALPRPipeline:
    def __init__(self, detector: Optional[PlateDetector] = None, ocr: Optional[PlateOCR] = None):
        self.detector = detector or PlateDetector()
        self.ocr = ocr or PlateOCR()

    def process_image(self, image_path: Path) -> List[PlateReadResult]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.process_frame(image)

    def process_frame(self, image: np.ndarray) -> List[PlateReadResult]:
        results = []
        for detection in self.detector.detect(image):
            crop = self.detector.crop(image, detection)
            processed = preprocess_plate(crop)
            ocr_result = self.ocr.read(processed)
            validation = validate_plate(ocr_result.text)
            results.append(
                PlateReadResult(
                    detection=detection,
                    raw_text=ocr_result.text,
                    plate_number=validation.normalized_text,
                    is_valid=validation.is_valid,
                    ocr_confidence=ocr_result.confidence,
                )
            )
        return results

    def process_and_annotate(self, image_path: Path, output_path: Optional[Path] = None) -> List[PlateReadResult]:
        results = self.process_image(image_path)

        annotated = cv2.imread(str(image_path))
        for result in results:
            label = result.plate_number if result.is_valid else f"{result.plate_number}?"
            annotated = draw_detection(annotated, result.detection.bbox, label)

        output_path = output_path or (config.OUTPUTS_DIR / Path(image_path).name)
        save_image(annotated, output_path)
        logger.info("Annotated output saved to %s", output_path)
        return results


def main():
    parser = argparse.ArgumentParser(description="Run the ALPR pipeline on a single image.")
    parser.add_argument("image", type=Path, help="Path to an input image.")
    parser.add_argument("--output", type=Path, default=None, help="Path for the annotated output image.")
    args = parser.parse_args()

    pipeline = ALPRPipeline()
    results = pipeline.process_and_annotate(args.image, args.output)

    if not results:
        logger.info("No plates detected.")
    for result in results:
        status = "VALID" if result.is_valid else "UNVERIFIED"
        logger.info(
            "[%s] plate=%s raw=%r confidence=%.2f bbox=%s",
            status,
            result.plate_number,
            result.raw_text,
            result.ocr_confidence,
            result.detection.bbox,
        )


if __name__ == "__main__":
    main()
