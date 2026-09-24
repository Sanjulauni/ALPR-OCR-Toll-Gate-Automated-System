"""OCR extraction from preprocessed plate crops."""
import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from paddleocr import PaddleOCR

from . import config


@dataclass
class OCRResult:
    text: str
    confidence: float


def clean_plate_text(text: str) -> str:
    """Strip everything but alphanumerics and uppercase the result."""
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


class PlateOCR:
    """Wraps PaddleOCR and reduces its output to a single cleaned string."""

    def __init__(self, lang: Optional[str] = None):
        # enable_mkldnn=False works around a PaddlePaddle/oneDNN PIR-executor bug
        # (NotImplementedError: ConvertPirAttribute2RuntimeAttribute ...) that
        # crashes text detection on CPU with paddlepaddle 3.3.1 on Windows.
        # Doc orientation/unwarping target scanned pages, not plate crops; they
        # add latency and can distort a tight crop.
        self._ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            lang=lang or config.OCR_LANGUAGE,
            enable_mkldnn=False,
        )

    def read(self, image: np.ndarray) -> OCRResult:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        prediction = self._ocr.predict(image)
        if not prediction:
            return OCRResult(text="", confidence=0.0)

        result = prediction[0]
        texts = result["rec_texts"]
        confidences = result["rec_scores"]

        combined = clean_plate_text("".join(texts))
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRResult(text=combined, confidence=avg_confidence)
