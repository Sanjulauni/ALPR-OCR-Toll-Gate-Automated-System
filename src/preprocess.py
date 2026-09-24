"""OpenCV preprocessing applied to cropped plate regions before OCR."""
import cv2
import numpy as np


def resize_for_ocr(image: np.ndarray, target_height: int = 64) -> np.ndarray:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image
    scale = target_height / h
    return cv2.resize(
        image, (max(int(w * scale), 1), target_height), interpolation=cv2.INTER_CUBIC
    )


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(image, 11, 17, 17)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Binarize a grayscale image. Useful for engines like Tesseract; PaddleOCR
    generally performs better on the grayscale (non-binarized) output instead."""
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )


def deskew(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    coords = cv2.findNonZero(cv2.bitwise_not(gray))
    if coords is None:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def preprocess_plate(image: np.ndarray) -> np.ndarray:
    """Upscale the crop and keep it in colour. Deskew/denoise/grayscale were
    tested on real detections and made PaddleOCR's output worse, so they are
    kept above as opt-in helpers rather than applied by default."""
    return resize_for_ocr(image)
