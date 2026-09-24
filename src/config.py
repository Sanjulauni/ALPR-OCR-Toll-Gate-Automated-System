"""Central configuration: paths and tunable thresholds for the ALPR pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# A model fine-tuned on license plates should be placed here. Until one is
# trained, the pipeline falls back to the base YOLOv8 COCO model, which can
# only localize vehicles (not plates) and is meant for wiring/smoke tests.
PLATE_DETECTOR_MODEL = MODELS_DIR / "plate_detector.pt"
FALLBACK_DETECTOR_MODEL = MODELS_DIR / "yolov8n.pt"

DETECTION_CONFIDENCE_THRESHOLD = 0.4
OCR_LANGUAGE = "en"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
