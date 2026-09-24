# ALPR OCR System for Toll Gate Automation

An end-to-end Automatic License Plate Recognition (ALPR) pipeline for
identifying Sri Lankan vehicles at toll gates, built with YOLOv8, OpenCV, and
OCR in Python.

## Architecture

The pipeline is split into independent, swappable stages:

```
Image / Frame
     │
     ▼
┌─────────────┐   src/detector.py    YOLOv8 locates candidate license-plate
│  Detection  │                      regions and returns bounding boxes.
└─────────────┘
     │
     ▼
┌─────────────┐   src/preprocess.py  OpenCV upscales the cropped plate region
│Preprocessing│                      (deskew/denoise helpers are opt-in).
└─────────────┘
     │
     ▼
┌─────────────┐   src/ocr_engine.py  PaddleOCR extracts and cleans the raw
│     OCR     │                      text from the processed crop.
└─────────────┘
     │
     ▼
┌─────────────┐   src/validator.py   Corrects common OCR letter/digit
│ Validation  │                      confusions and checks the result against
└─────────────┘                      Sri Lankan plate formats.
     │
     ▼
Plate number + confidence + bounding box
```

`src/pipeline.py` (`ALPRPipeline`) wires the four stages together and exposes
a CLI entry point.

## Project layout

```
src/
  config.py       # paths and tunable thresholds
  detector.py      # PlateDetector — YOLOv8 wrapper
  preprocess.py     # OpenCV resize/cleanup for plate crops
  ocr_engine.py      # PlateOCR — PaddleOCR wrapper
  validator.py        # Sri Lankan plate format validation/correction
  pipeline.py           # ALPRPipeline orchestrator + CLI (images)
  video.py                # video file / webcam runner with plate tracking
  utils.py               # logging, drawing, image IO helpers
scripts/
  smoke_test_detector.py # manual check that YOLO loads and runs
  smoke_test_ocr.py       # manual check that PaddleOCR loads and runs
  prepare_dataset.py       # Kaggle VOC XML -> YOLO format + train/val split
  train.py                  # fine-tune YOLOv8 locally (Colab GPU recommended)
data/
  images/                  # sample input images
  test/                     # held-out test images
models/                       # model weights (.pt) — not committed, see below
outputs/                       # annotated pipeline output images
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Model weights

- `models/yolov8n.pt` is the stock Ultralytics COCO checkpoint, used only as
  the starting point for fine-tuning. It has no license-plate class.
- `models/plate_detector.pt` is the fine-tuned plate detector (single class,
  `license_plate`). `PlateDetector` prefers it automatically and falls back
  to `yolov8n.pt` only if it is missing. Weights are not committed.

### Training the plate detector

Dataset: [Sri Lanka vehicle number plates (Kaggle, CC0)](https://www.kaggle.com/datasets/rohankumara/sri-lanka-vehicle-number-plates)
— about 250 images with Pascal VOC XML plate boxes.

CPU training is slow, so use a Google Colab T4 GPU:

1. In Colab, install `ultralytics` and download the dataset (Kaggle API token,
   or upload the ZIP and confirm its size is about 21 MB before unzipping).
2. Convert VOC XML to YOLO format with an 80/20 train/val split (same logic as
   `scripts/prepare_dataset.py`), then run
   `YOLO("yolov8n.pt").train(data=..., epochs=50, imgsz=640, batch=16)`.
3. Download `runs/plate/weights/best.pt`, rename it `plate_detector.pt`, and
   place it in `models/`.

To train locally instead: extract the dataset into `data/raw/`, then run
`python scripts/prepare_dataset.py` and `python scripts/train.py`.

## Usage

Run the full pipeline on an image:

```bash
python -m src.pipeline data/images/car.jpg --output outputs/car_annotated.jpg
```

Run on a video file or webcam (plates are tracked across frames and logged
once each to `outputs/plates_log.csv`; press `q` to stop):

```bash
python -m src.video 0
python -m src.video path/to/video.mp4 --save outputs/out.mp4
python -m src.video path/to/video.mp4 --no-display
```

Options: `--stride N` (detect every Nth frame, default 3), `--min-hits N`
(reads needed to confirm a plate, default 3), `--max-age N` (frames without a
detection before a track ends, default 15). A plate is logged once it has been
read consistently as a valid format; the most-confident reading across frames
wins, which suppresses single-frame OCR noise.

Run a folder of test images and get accuracy stats:

```bash
python scripts/batch_test.py --dir data/test
```

Run the individual smoke tests:

```bash
python scripts/smoke_test_detector.py
python scripts/smoke_test_ocr.py
```

## Notes on OCR

- PaddleOCR runs with `enable_mkldnn=False`. With paddlepaddle 3.3.1 on
  Windows CPU, oneDNN raises `ConvertPirAttribute2RuntimeAttribute` errors.
- Document orientation/unwarping are disabled; they target scanned pages and
  break on tight plate crops.
- Grayscale/deskew/denoise preprocessing made OCR worse on real crops, so the
  default preprocessing only upscales the colour crop.
- Separators between letters and digits are dropped during text cleaning.

## Plate validation

Sri Lankan plates (post-2013 provincial format and the older format) reduce,
once punctuation is stripped, to 1–3 letters followed by 4 digits (e.g.
`CAB1234`, `KA1234`). `src/validator.py` normalizes OCR output against this
shape and corrects characters PaddleOCR commonly confuses on low-resolution
plate crops (`0`/`O`, `1`/`I`, `5`/`S`, `8`/`B`, `2`/`Z`). It does not check
issued plate letter-codes against the province/series registry — only the
general format.

Older plates carry a small province code before the registration (e.g.
`WP KM-5141`). When the OCR text does not fit the plate pattern but does once
a leading province code (`WP`, `CP`, `SP`, `NP`, `EP`, `NW`, `NC`, `UP`, `SG`)
is removed, the code is stripped (`WPKM5141` -> `KM5141`) and returned in the
validator's `province` field. Plates that already fit the pattern are left
alone.

## Known limitations

- Detection was fine-tuned on about 250 images; it has only been spot-checked
  on a single sample photo. Run a batch on varied Sri Lankan plates (angles,
  lighting, motorcycles) before trusting the accuracy.
- The validator checks plate *shape*, not whether a given letter-series code
  has actually been issued.
- Video mode has only been tested on a synthetic clip built from one photo,
  not real toll-gate footage or a live camera.
- Old-style plates carry a small province code (e.g. `WP`) beside the main
  registration. Full codes are stripped by the validator, but if OCR reads only
  one stray letter of it (e.g. `WKM5141`) that is indistinguishable from a
  3-letter series and is kept. Video mode's cross-frame voting mostly filters
  this out; single images may not.
