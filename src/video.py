"""ALPR on a video file or webcam, with cross-frame tracking so each plate is
logged once instead of on every frame.

Usage:
    python -m src.video 0                          # default webcam
    python -m src.video path/to/video.mp4          # video file
    python -m src.video video.mp4 --no-display --save outputs/out.mp4

Press 'q' in the preview window to stop.
"""
import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from . import config
from .pipeline import ALPRPipeline, PlateReadResult
from .utils import draw_detection, ensure_dir, get_logger

logger = get_logger(__name__)

BBox = Tuple[int, int, int, int]


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


@dataclass
class Track:
    bbox: BBox
    last_frame: int
    hits: int = 0
    scores: Dict[str, float] = field(default_factory=dict)
    valid_texts: set = field(default_factory=set)
    emitted: bool = False

    def add(self, result: PlateReadResult, frame_idx: int) -> None:
        self.bbox = result.detection.bbox
        self.last_frame = frame_idx
        if not result.plate_number:
            return
        self.hits += 1
        self.scores[result.plate_number] = (
            self.scores.get(result.plate_number, 0.0) + result.ocr_confidence
        )
        if result.is_valid:
            self.valid_texts.add(result.plate_number)

    def best(self) -> Optional[str]:
        """Highest cumulative-confidence text, restricted to valid formats if any."""
        candidates = {t: s for t, s in self.scores.items() if t in self.valid_texts} or self.scores
        return max(candidates, key=candidates.get) if candidates else None


class PlateTracker:
    """Associates detections across frames by box overlap and confirms a plate
    once it has been read consistently enough."""

    def __init__(self, min_hits: int = 3, max_age: int = 15, iou_threshold: float = 0.3):
        self.min_hits = min_hits
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []

    def update(self, results: List[PlateReadResult], frame_idx: int) -> List[Tuple[str, float]]:
        matched = set()
        for result in results:
            best_track, best_iou = None, self.iou_threshold
            for track in self.tracks:
                if id(track) in matched:
                    continue
                overlap = iou(track.bbox, result.detection.bbox)
                if overlap >= best_iou:
                    best_track, best_iou = track, overlap
            if best_track is None:
                best_track = Track(bbox=result.detection.bbox, last_frame=frame_idx)
                self.tracks.append(best_track)
            matched.add(id(best_track))
            best_track.add(result, frame_idx)

        confirmed = []
        for track in list(self.tracks):
            expired = frame_idx - track.last_frame > self.max_age
            text = track.best()
            ready = text in track.valid_texts and (track.hits >= self.min_hits or expired)
            if ready and not track.emitted:
                track.emitted = True
                confirmed.append((text, track.scores[text] / max(track.hits, 1)))
            if expired:
                self.tracks.remove(track)
        return confirmed

    def flush(self, frame_idx: int) -> List[Tuple[str, float]]:
        """Expire everything (end of stream) and return any unreported plates."""
        return self.update([], frame_idx + self.max_age + 1)

    def active(self, frame_idx: int) -> List[Track]:
        return [t for t in self.tracks if frame_idx - t.last_frame <= self.max_age]


def open_source(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    return cap


def append_log(log_path: Path, rows: List[dict]) -> None:
    ensure_dir(log_path.parent)
    is_new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "frame", "plate", "confidence"])
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def run(
    source: str,
    stride: int = 3,
    min_hits: int = 3,
    max_age: int = 15,
    display: bool = True,
    save_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
    pipeline: Optional[ALPRPipeline] = None,
) -> List[str]:
    pipeline = pipeline or ALPRPipeline()
    tracker = PlateTracker(min_hits=min_hits, max_age=max_age)
    log_path = log_path or config.OUTPUTS_DIR / "plates_log.csv"
    cap = open_source(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    frame_idx = 0
    seen: List[str] = []

    def record(events, frame):
        rows = []
        for text, conf in events:
            seen.append(text)
            rows.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "frame": frame,
                "plate": text,
                "confidence": round(conf, 2),
            })
            logger.info("Plate confirmed: %s (avg conf %.2f) at frame %d", text, conf, frame)
        if rows:
            append_log(log_path, rows)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % stride == 0:
                results = pipeline.process_frame(frame)
                record(tracker.update(results, frame_idx), frame_idx)

            for track in tracker.active(frame_idx):
                label = track.best() or "..."
                color = (0, 255, 0) if track.emitted else (0, 200, 255)
                draw_detection(frame, track.bbox, label, color)

            if save_path:
                if writer is None:
                    ensure_dir(save_path.parent)
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                    )
                writer.write(frame)

            if display:
                try:
                    cv2.imshow("ALPR", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    logger.warning("No display available; continuing without preview.")
                    display = False

            frame_idx += 1
    finally:
        record(tracker.flush(frame_idx), frame_idx)
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    logger.info("Done. %d frames, %d unique plate reads logged to %s", frame_idx, len(seen), log_path)
    return seen


def main():
    parser = argparse.ArgumentParser(description="Run ALPR on a video file or webcam.")
    parser.add_argument("source", help="Webcam index (e.g. 0) or path to a video file.")
    parser.add_argument("--stride", type=int, default=3, help="Run detection+OCR every Nth frame.")
    parser.add_argument("--min-hits", type=int, default=3, help="Reads needed to confirm a plate.")
    parser.add_argument("--max-age", type=int, default=15, help="Frames without a detection before a plate track ends.")
    parser.add_argument("--no-display", action="store_true", help="Do not open a preview window.")
    parser.add_argument("--save", type=Path, default=None, help="Save the annotated video here.")
    parser.add_argument("--log", type=Path, default=None, help="CSV of confirmed plates (default outputs/plates_log.csv).")
    args = parser.parse_args()

    run(
        args.source,
        stride=args.stride,
        min_hits=args.min_hits,
        max_age=args.max_age,
        display=not args.no_display,
        save_path=args.save,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()
