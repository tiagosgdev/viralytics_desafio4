"""Lightweight human segmentation using the MediaPipe Tasks ImageSegmenter.

The legacy ``mp.solutions.selfie_segmentation`` API was removed in MediaPipe 0.10+,
so this uses the Tasks ImageSegmenter with the selfie-segmenter model
(``models/weights/mediapipe/selfie_segmenter.tflite``). The mask feeds the
silhouette width sampling (waist/chest), which has no landmark fallback — without
it those measurements collapse to zero.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


MASK_THRESHOLD = 0.45
MORPH_KERNEL_SIZE = 5
DEFAULT_MODEL_RELATIVE_PATH = Path("models") / "weights" / "mediapipe" / "selfie_segmenter.tflite"


class BodySegmenter:
    """Generate a cleaned binary body mask via the Tasks ImageSegmenter."""

    def __init__(self, model_path: str | os.PathLike[str] | None = None) -> None:
        self.model_path = Path(model_path or self._default_model_path())
        self._segmenter = None
        self._mp = None

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        segmenter = self._get_segmenter()
        mp = self._mp
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = segmenter.segment(mp_image)

        # The selfie model exposes a single foreground confidence mask (float 0..1).
        raw_mask = None
        confidence_masks = getattr(result, "confidence_masks", None)
        if confidence_masks:
            raw_mask = confidence_masks[0].numpy_view()
        else:
            category_mask = getattr(result, "category_mask", None)
            if category_mask is not None:
                # Category mask: 0 = background, non-zero = person foreground.
                raw_mask = (category_mask.numpy_view() > 0).astype(np.float32)
        if raw_mask is None:
            return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

        mask = (raw_mask >= MASK_THRESHOLD).astype(np.uint8) * 255
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return keep_largest_component(mask)

    def _get_segmenter(self):
        if self._segmenter is not None:
            return self._segmenter
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Selfie segmenter model not found at {self.model_path}. "
                "Download selfie_segmenter.tflite into models/weights/mediapipe/."
            )
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.ImageSegmenterOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)
        return self._segmenter

    @staticmethod
    def _default_model_path() -> str:
        env_path = os.getenv("SELFIE_SEGMENTER_MODEL_PATH")
        if env_path:
            return env_path
        project_root = Path(__file__).resolve().parents[2]
        return str(project_root / DEFAULT_MODEL_RELATIVE_PATH)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest foreground component in a binary mask."""
    if mask is None or mask.size == 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8) * 255


def overlay_body_mask(
    image_bgr: np.ndarray,
    body_mask: np.ndarray | None,
    *,
    color: tuple[int, int, int] = (40, 180, 200),
    alpha: float = 0.22,
) -> np.ndarray:
    """Blend a body mask over an image for debugging/visualization."""
    if body_mask is None or body_mask.size == 0:
        return image_bgr.copy()
    output = image_bgr.copy()
    color_layer = np.zeros_like(output)
    color_layer[:, :] = color
    blended = cv2.addWeighted(output, 1.0 - alpha, color_layer, alpha, 0)
    output[body_mask > 0] = blended[body_mask > 0]
    return output
