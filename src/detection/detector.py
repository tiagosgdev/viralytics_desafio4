"""
src/detection/detector.py
─────────────────────────
Detection base class and YOLOv8 inference wrapper.
Returns structured DetectionResult objects used throughout the app.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO

from src.utils.color_utils import get_dominant_color_name

try:
    from ultralytics.nn.tasks import DetectionModel
except Exception:  # pragma: no cover - version-dependent import
    DetectionModel = None


def _prepare_trusted_yolo_checkpoint_loading() -> None:
    """
    PyTorch 2.6 changed torch.load(..., weights_only=True) to be the default.
    Older Ultralytics checkpoints may require either allow-listing model classes
    or forcing the legacy behavior for trusted local checkpoints.
    """
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if DetectionModel is not None:
        try:
            torch.serialization.add_safe_globals([DetectionModel])
        except Exception:
            pass


# ── Category definitions ───────────────────────────────────────────────────

CATEGORY_NAMES = {
    0: "short_sleeve_top",
    1: "long_sleeve_top",
    2: "short_sleeve_outwear",
    3: "long_sleeve_outwear",
    4: "vest",
    5: "sling",
    6: "shorts",
    7: "trousers",
    8: "skirt",
    9: "short_sleeve_dress",
    10: "long_sleeve_dress",
    11: "vest_dress",
    12: "sling_dress",
}

# edna/FashionNet trained on balanced dataset (11 classes — sling + short_sleeve_outwear dropped)
CATEGORY_NAMES_11 = {
    0: "short_sleeve_top",
    1: "long_sleeve_top",
    2: "long_sleeve_outwear",
    3: "vest",
    4: "shorts",
    5: "trousers",
    6: "skirt",
    7: "short_sleeve_dress",
    8: "long_sleeve_dress",
    9: "vest_dress",
    10: "sling_dress",
}

CATEGORY_COLORS = {
    0:  (255, 100, 100),   # short_sleeve_top    — coral
    1:  (255, 160,  60),   # long_sleeve_top     — amber
    2:  ( 80, 200, 120),   # short_sleeve_outwear— green
    3:  ( 40, 180, 200),   # long_sleeve_outwear — teal
    4:  (180,  80, 255),   # vest                — violet
    5:  (255,  80, 180),   # sling               — pink
    6:  (100, 180, 255),   # shorts              — sky
    7:  ( 60,  80, 200),   # trousers            — blue
    8:  (220, 180,  40),   # skirt               — gold
    9:  (255, 120,  80),   # short_sleeve_dress  — orange
    10: (140,  60, 200),   # long_sleeve_dress   — purple
    11: (200, 100, 150),   # vest_dress          — mauve
    12: (100, 200, 180),   # sling_dress         — mint
}


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class Detection:
    class_id:    int
    class_name:  str
    confidence:  float
    bbox:        List[int]          # [x1, y1, x2, y2] in pixels
    color:       tuple = field(default_factory=lambda: (0, 255, 0))
    color_name:  str = ""


@dataclass
class DetectionResult:
    detections:     List[Detection]
    frame_shape:    tuple           # (H, W, C)
    inference_ms:   float
    timestamp:      float


# ── Base detector ──────────────────────────────────────────────────────

class BaseDetector(ABC):
    """
    Abstract base class for all detectors.
    Subclasses must implement ``detect()``.
    """

    conf_thres: float
    iou_thres:  float

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run inference on a BGR frame. Returns DetectionResult."""

    def draw(
        self,
        frame:  np.ndarray,
        result: DetectionResult,
        show_conf: bool = True,
    ) -> np.ndarray:
        """Returns a copy of frame with bounding boxes drawn."""
        out = frame.copy()

        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = det.color

            # Box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = det.class_name.replace("_", " ")
            if show_conf:
                label += f"  {det.confidence:.0%}"
            if det.color_name:
                color_label = det.color_name.replace("_", " ")
                label += f" | {color_label}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            label_x1 = min(x1, max(0, out.shape[1] - tw - 4))
            label_x2 = min(out.shape[1] - 1, label_x1 + tw + 4)
            if y1 - th - baseline - 6 >= 0:
                label_y1 = y1 - th - baseline - 6
                label_y2 = y1
                text_y = y1 - baseline - 2
            else:
                label_y1 = y1
                label_y2 = min(out.shape[0] - 1, y1 + th + baseline + 6)
                text_y = min(out.shape[0] - 2, y1 + th + 2)
            cv2.rectangle(out, (label_x1, label_y1), (label_x2, label_y2), color, -1)

            # Label text
            cv2.putText(
                out, label, (label_x1 + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )

        # FPS / inference time overlay
        fps_text = f"{1000/result.inference_ms:.1f} FPS  ({result.inference_ms:.0f} ms)"
        cv2.putText(out, fps_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)

        return out


# ── YOLOv8 detector ───────────────────────────────────────────────────

class FashionDetector(BaseDetector):
    """
    Wraps YOLOv8 for clothing detection.

    Example
    -------
    detector = FashionDetector("models/weights/.../best.pt")
    result   = detector.detect(frame)          # frame = BGR numpy array
    annotated = detector.draw(frame, result)
    """

    def __init__(
        self,
        weights:    str   = "yolov8s.pt",   # swap for your best.pt after training
        conf_thres: float = 0.60,
        iou_thres:  float = 0.45,
        device:     str   = "",             # "" = auto
        imgsz:      int   = 640,
    ):
        self.conf_thres = conf_thres
        self.iou_thres  = iou_thres
        self.imgsz      = imgsz

        _prepare_trusted_yolo_checkpoint_loading()
        print(f"🔍  Loading model: {weights}")
        self.model = YOLO(weights)
        if device:
            self.model.to(device)
        print("✅  Model ready")

    # ── Public API ─────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run inference on a BGR frame. Returns DetectionResult."""
        t0 = time.perf_counter()

        results = self.model.predict(
            source  = frame,
            conf    = self.conf_thres,
            iou     = self.iou_thres,
            imgsz   = self.imgsz,
            verbose = False,
        )

        inference_ms = (time.perf_counter() - t0) * 1000

        # Parse detections and enrich each with dominant color (from the input frame)
        detections   = self._parse(results, frame)

        return DetectionResult(
            detections   = detections,
            frame_shape  = frame.shape,
            inference_ms = inference_ms,
            timestamp    = time.time(),
        )

    # ── Private ────────────────────────────────────────────────────────────

    def _parse(self, results, frame: np.ndarray) -> List[Detection]:
        """Parse raw YOLO results into Detection objects and compute dominant color per box.

        The frame must be the original BGR image used for inference so we can crop
        detection boxes and run the color pipeline (HSV -> KMeans -> map to name).
        """
        detections = []
        H, W = frame.shape[:2]

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls.item())
                bbox = [int(v) for v in box.xyxy[0].tolist()]

                # Clip bbox to frame
                x1 = max(0, min(W - 1, bbox[0]))
                y1 = max(0, min(H - 1, bbox[1]))
                x2 = max(0, min(W - 1, bbox[2]))
                y2 = max(0, min(H - 1, bbox[3]))

                det = Detection(
                    class_id   = class_id,
                    class_name = CATEGORY_NAMES.get(class_id, f"class_{class_id}"),
                    confidence = float(box.conf.item()),
                    bbox       = [x1, y1, x2, y2],
                    color      = CATEGORY_COLORS.get(class_id, (0, 255, 0)),
                )

                # Crop region and compute dominant color + name. Be defensive in case
                # the crop is empty or too small.
                try:
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        dom_rgb, dom_name = get_dominant_color_name(crop)
                        if dom_rgb is not None:
                            det.color = dom_rgb
                        det.color_name = dom_name or ""
                        # Print detected color info to the server terminal for debugging/visibility
                        try:
                            print(f"[DETECT] {det.class_name} conf={det.confidence:.3f} color={det.color} color_name='{det.color_name}' bbox={det.bbox}")
                        except Exception:
                            # Keep parsing robust even if print formatting fails
                            print(f"[DETECT] {det.class_name} conf={det.confidence:.3f} color=<unknown> bbox={det.bbox}")
                except Exception:
                    # If color extraction fails, keep default category color
                    det.color_name = ""
                    try:
                        print(f"[DETECT] {det.class_name} conf={det.confidence:.3f} color={det.color} color_name='<error>' bbox={det.bbox}")
                    except Exception:
                        print(f"[DETECT] {det.class_name} conf={det.confidence:.3f} color=<error> bbox={det.bbox}")

                detections.append(det)

        return detections
