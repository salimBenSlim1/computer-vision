"""AI Coffee Detection Agent — single clean module.

This file implements a single `CoffeeDetectionAgent` with:
- Clean top-level imports
- `DetectionConfig` dataclass
- Robust model loading with graceful fallback
- Grid-based spatial deduplication + cooldown to avoid double-counting
- Simple dataset setup helper and a training stub
- JSON/CSV daily reporting

Designed to be easy to deploy on Jetson Nano / AWS / local PC.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import yaml

# Module logger
logger = logging.getLogger("coffee_agent")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)


@dataclass
class DetectionConfig:
    confidence_threshold: float = 0.25  # Very low detection threshold for instant pickup
    cooldown_seconds: float = 1.5  # Short window to allow quick re-detection after cooldown
    grid_cell_size: int = 100  # px for coarse spatial dedup (larger = less oversensitive)
    min_confidence_to_count: float = 0.3  # Very permissive counting threshold
    # When a track is first created, allow counting on a slightly lower confidence
    # to avoid waiting multiple frames. Set to None to compute automatically
    # as (min_confidence_to_count - 0.1) floored at 0.4.
    initial_count_conf: Optional[float] = 0.2  # Count immediately on first detection >20% confidence
    max_distance_pixels: float = 150  # Max pixel distance to consider same object
    reports_dir: str = "reports"
    device: Optional[str] = None  # e.g. 'cpu' or 'cuda'
    # Only track these classes (cups and coffee types)
    allowed_classes: set = None  # Will be set to {"cup", "espresso", "cappuccino"} in __init__


class CoffeeDetectionAgent:
    """Count coffees prepared in live camera streams.

    Public API:
      - `setup_dataset(dataset_path)` create minimal dataset structure
      - `train_model(...)` simple training stub (calls ultralytics training when dataset present)
      - `test_on_image(path)` run detection and return structured detections
      - `start_detection(source)` run live camera loop (press 'q' to quit)
      - `snapshot()` get current counters
      - `write_report()` persist JSON + CSV
    """

    def __init__(self, model_path: str = "coffee_model_final.pt", config: Optional[DetectionConfig] = None, data_yaml: str = "data.yaml"):
        self.model_path = Path(model_path)
        # store data_yaml path so model loader can use it to set model.names
        self.data_yaml = Path(data_yaml)
        self.config = config or DetectionConfig()
        # Load class names from `data.yaml` (preferred) and set both the
        # ordered `_data_names` and `config.allowed_classes`. Keep a safe
        # fallback to the original three classes if anything goes wrong.
        try:
            if self.data_yaml.exists():
                with open(self.data_yaml, "r", encoding="utf8") as f:
                    data = yaml.safe_load(f) or {}
                names = []
                names_raw = data.get("names")
                
                if isinstance(names_raw, dict):
                    # keys may be integers or strings; extract by numeric order
                    try:
                        # try int keys first
                        max_k = max((int(k) if isinstance(k, int) else int(k) for k in names_raw.keys()), default=-1)
                        names = [names_raw.get(i) or names_raw.get(str(i)) for i in range(max_k + 1)]
                        names = [n for n in names if n is not None]
                    except (ValueError, TypeError):
                        # fallback: just extract values
                        names = list(names_raw.values())
                elif isinstance(names_raw, list):
                    names = list(names_raw)

                if names:
                    # ordered list of class names from the dataset YAML
                    self._data_names = list(names)
                    # set allowed_classes to the full dataset set
                    self.config.allowed_classes = set(self._data_names)
                    logger.info("✓ Loaded %d classes from %s", len(self._data_names), str(self.data_yaml))
                else:
                    raise ValueError("no names found in data.yaml")
            else:
                raise FileNotFoundError("data.yaml not present")
        except Exception as e:
            # fallback to previous small set and ensure ordered list exists
            logger.warning("Falling back to default classes — %s: %s", type(e).__name__, str(e))
            self.config.allowed_classes = {"cup", "espresso", "cappuccino"}
            self._data_names = list(self.config.allowed_classes)

        # Compute derived initial_count_conf if not provided
        if self.config.initial_count_conf is None:
            self.config.initial_count_conf = max(0.4, self.config.min_confidence_to_count - 0.1)

        # Counting state — MUST initialize before _load_model
        self._counters: Dict[str, int] = {}
        # Track active objects: (label, center_x, center_y) -> last_seen_timestamp
        self._active_objects: Dict[Tuple[str, float, float], float] = {}
        self._lock = Lock()

        self._model: Optional[YOLO] = None
        self._load_model()

        # Ensure reports directory
        self.reports_dir = Path(self.config.reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Lightweight tracker for robust ID assignment and velocity (vitesse)
        # max_age uses cooldown_seconds to expire old tracks
        self.tracker = Tracker(max_age=self.config.cooldown_seconds * 3, iou_threshold=0.3)

    # ---------------------- Model Loading ----------------------
    def _load_model(self) -> None:
        """Try to load a custom model, else fall back to `yolov8n.pt`.

        Keeps counters initialized from model class names when available.
        """
        try:
            if self.model_path.exists():
                logger.info("Loading custom model from %s", self.model_path)
                self._model = YOLO(str(self.model_path))
            else:
                logger.warning("Custom model not found at %s — using pretrained yolov8n", self.model_path)
                self._model = YOLO("yolov8n.pt")

            # If we have names from data.yaml, override the model's names mapping
            try:
                if hasattr(self, '_data_names') and self._data_names:
                    # build dict mapping index->name
                    names_dict = {i: n for i, n in enumerate(self._data_names)}
                    # assign to model so subsequent code uses dataset names
                    try:
                        # override model names mapping
                        self._model.names = names_dict
                        logger.info("Model names overridden with %d dataset names", len(names_dict))
                    except Exception:
                        # best-effort, ignore if model doesn't allow assignment
                        pass

            except Exception:
                # ignore any failure in name overriding
                pass

            # Initialize counters for allowed classes
            for class_name in self.config.allowed_classes:
                self._counters.setdefault(class_name, 0)
        except Exception as exc:  # defensive: fallback
            logger.exception("Model load failed, falling back to yolov8n: %s", exc)
            self._model = YOLO("yolov8n.pt")

    # ---------------------- Dataset Helpers / Training Stub ----------------------
    def setup_dataset(self, dataset_path: str = "coffee_dataset") -> None:
        """Create minimal YOLOv8 dataset layout. Add 4-5 photos per class for quick prototyping."""
        base = Path(dataset_path)
        (base / "images" / "train").mkdir(parents=True, exist_ok=True)
        (base / "images" / "val").mkdir(parents=True, exist_ok=True)
        (base / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (base / "labels" / "val").mkdir(parents=True, exist_ok=True)
        logger.info("Dataset folders created under %s", str(base))

    def train_model(self, data_yaml: str, epochs: int = 10, imgsz: int = 640, resume: bool = False, device: Optional[str] = None) -> None:
        """Simple training helper. Requires a YOLO-compatible `data.yaml`.

        This is a convenience wrapper; use with care on edge devices.
        """
        if self._model is None:
            raise RuntimeError("Model not available for training")
        try:
            logger.info("Starting training: data=%s epochs=%d imgsz=%d resume=%s", data_yaml, epochs, imgsz, resume)
            # ultralytics YOLO object supports `.train(...)` in many versions
            # pass resume flag through to ultralytics trainer when requested
            # Allow explicit device selection (e.g., 'cpu' or '0' for GPU)
            train_kwargs = dict(data=data_yaml, epochs=epochs, imgsz=imgsz)
            if resume:
                train_kwargs['resume'] = True
            if device:
                train_kwargs['device'] = device

            self._model.train(**train_kwargs)
            logger.info("Training completed")
        except Exception as exc:
            logger.exception("Training failed: %s", exc)

    # ---------------------- Counting Helpers ----------------------
    def _calculate_iou(self, box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def _find_matching_object(self, label: str, center_x: float, center_y: float, bbox: Tuple[float, float, float, float], now_ts: float) -> Optional[Tuple[str, float, float]]:
        """Find if this detection matches an existing tracked object.
        
        Uses:
        1. Class name matching
        2. Spatial proximity (center distance)
        3. Cooldown period
        
        Returns the matched object key or None if new object.
        """
        for (tracked_label, tracked_cx, tracked_cy), last_seen in list(self._active_objects.items()):
            # Must be same class
            if tracked_label != label:
                continue
            
            # Must be within time window
            if now_ts - last_seen > self.config.cooldown_seconds:
                continue
            
            # Must be within spatial distance
            distance = ((center_x - tracked_cx) ** 2 + (center_y - tracked_cy) ** 2) ** 0.5
            if distance > self.config.max_distance_pixels:
                continue
            
            # Match found!
            return (tracked_label, tracked_cx, tracked_cy)
        
        return None

    def increment(self, label: str, x1: float, y1: float, x2: float, y2: float, confidence: float = 1.0) -> bool:
        """Thread-safe increment with improved object tracking to prevent duplicate counting.
        
        Returns True if object was counted (new), False if it was an existing object.
        """
        # Only count allowed classes
        if label not in self.config.allowed_classes:
            return False
        
        # Only count if confidence is high enough
        if confidence < self.config.min_confidence_to_count:
            return False
        
        now_ts = time.time()
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        bbox = (x1, y1, x2, y2)
        
        with self._lock:
            # Check if this is a new object or an existing one
            match = self._find_matching_object(label, center_x, center_y, bbox, now_ts)
            
            if match is None:
                # New object detected - count it!
                key = (label, center_x, center_y)
                self._active_objects[key] = now_ts
                self._counters[label] = self._counters.get(label, 0) + 1
                logger.info("✓ Counted new %s (conf=%.2f) — total %d", label, confidence, self._counters[label])
                return True
            else:
                # Existing object - just update its timestamp
                self._active_objects[match] = now_ts
                return False
            
            # Note: Cleanup happens at end of detection loop

    # ---------------------- Inference Helpers ----------------------
    def test_on_image(self, image_path: str, show: bool = False):
        """Run detection on a single image and return structured detections."""
        if self._model is None:
            raise RuntimeError("Model not loaded")

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Image not readable: {image_path}")

        results = self._model(img, conf=self.config.confidence_threshold)
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = self._model.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append({"label": name, "conf": conf, "bbox": (x1, y1, x2, y2)})

        if show:
            annotated = results[0].plot()
            cv2.imshow("Detections", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return detections

    def start_detection(self, source=0, save_video: Optional[str] = None):
        """Start live detection loop. `source` can be camera index or RTSP URL."""
        if self._model is None:
            raise RuntimeError("Model not loaded — cannot start detection")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error("Unable to open camera/source: %s", source)
            return

        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(save_video, fourcc, 15.0, (w, h))

        try:
            logger.info("Starting live detection on %s", source)
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame read failed, stopping")
                    break

                frame_count += 1
                results = self._model(frame, conf=self.config.confidence_threshold, verbose=False)

                # Build detections list and update tracker
                now_ts = time.time()
                detections = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self._model.names.get(cls_id, str(cls_id))
                    if label in self.config.allowed_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append({
                            "bbox": (float(x1), float(y1), float(x2), float(y2)),
                            "label": label,
                            "conf": float(conf),
                        })

                # Update tracker with detections
                try:
                    self.tracker.update(detections, timestamp=now_ts)
                except Exception:
                    logger.exception("Tracker update failed")

                # Count newly created tracks (first time seen with sufficient confidence)
                with self._lock:
                    for tr in self.tracker.get_active_tracks():
                        if getattr(tr, "counted", False):
                            continue
                        # Count immediately if detection confidence is high enough
                        now = now_ts
                        # Rule 1: standard counting threshold
                        if tr.last_conf >= self.config.min_confidence_to_count:
                            self._counters[tr.label] = self._counters.get(tr.label, 0) + 1
                            tr.counted = True
                            logger.info("Counted track %d label=%s conf=%.2f total=%d", tr.id, tr.label, tr.last_conf, self._counters[tr.label])
                            continue

                        # Rule 2: initial fast-count — if the track is newly created and has
                        # a slightly lower confidence but above initial_count_conf, count it
                        age = tr.time_since_creation(now)
                        if age <= 1.0 and tr.last_conf >= self.config.initial_count_conf:
                            self._counters[tr.label] = self._counters.get(tr.label, 0) + 1
                            tr.counted = True
                            logger.info("Fast-counted track %d label=%s conf=%.2f age=%.2f total=%d", tr.id, tr.label, tr.last_conf, age, self._counters[tr.label])
                            continue

                # Annotate and overlay counters - only show allowed classes
                annotated = results[0].plot()
                filtered_counters = {k: v for k, v in self._counters.items() if k in self.config.allowed_classes}
                cv2.rectangle(annotated, (10, 10), (320, 30 + len(filtered_counters) * 24), (0, 0, 0), -1)
                cv2.putText(annotated, "Today's Counts:", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                yy = 60
                for name, cnt in sorted(filtered_counters.items(), key=lambda x: (-x[1], x[0])):
                    cv2.putText(annotated, f"{name}: {cnt}", (20, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    yy += 24
                
                # Show active tracking info
                cv2.putText(annotated, f"Tracked: {len(self.tracker.get_active_tracks())}", (20, yy + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)

                cv2.imshow("Coffee Detection", annotated)
                if writer:
                    writer.write(annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    logger.info("Current counts: %s | Tracked objects: %d", self._counters, len(self.tracker.get_active_tracks()))

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.write_report()

    # ---------------------- Reporting ----------------------
    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def write_report(self) -> Tuple[str, str]:
        with self._lock:
            # Only include allowed classes in the report
            filtered_counts = {k: v for k, v in self._counters.items() if k in self.config.allowed_classes}
            report = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "generated_at": datetime.now().isoformat(),
                "counts": filtered_counts
            }

        json_path = self.reports_dir / f"coffee_report_{report['date']}.json"
        csv_path = self.reports_dir / f"coffee_report_{report['date']}.csv"

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2, ensure_ascii=False)

        with open(csv_path, "w", encoding="utf-8") as cf:
            cf.write("item,count\n")
            for item, cnt in sorted(report["counts"].items(), key=lambda x: (-x[1], x[0])):
                cf.write(f"{item},{cnt}\n")

        logger.info("Wrote reports: %s, %s", json_path, csv_path)
        return str(json_path), str(csv_path)


def quick_start():
    print("Import `CoffeeDetectionAgent` and call `.start_detection()` or `.test_on_image()`")


if __name__ == "__main__":
    agent = CoffeeDetectionAgent()
    try:
        agent.start_detection(0)
    except KeyboardInterrupt:
        logger.info("Interrupted by user — exiting")