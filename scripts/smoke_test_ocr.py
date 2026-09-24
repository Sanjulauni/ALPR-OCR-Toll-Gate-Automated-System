"""Quick manual check that PaddleOCR loads and runs on a sample image.

Run from the project root: python scripts/smoke_test_ocr.py
"""
from pathlib import Path

from paddleocr import PaddleOCR

ROOT_DIR = Path(__file__).resolve().parent.parent
# enable_mkldnn=False works around a PaddlePaddle/oneDNN PIR-executor crash on
# CPU with paddlepaddle 3.3.1 on Windows (see src/ocr_engine.py).
ocr = PaddleOCR(use_textline_orientation=True, lang="en", enable_mkldnn=False)

result = ocr.predict(str(ROOT_DIR / "data" / "images" / "car.jpg"))

for res in result:
    print(res["rec_texts"])
