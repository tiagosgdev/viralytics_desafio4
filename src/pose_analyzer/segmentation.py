"""Lightweight human segmentation using MediaPipe Selfie Segmentation."""

from __future__ import annotations

import cv2
import numpy as np


MASK_THRESHOLD = 0.45
MORPH_KERNEL_SIZE = 5


class BodySegmenter:
    """Generate a cleaned binary body mask."""

    def __init__(self, model_selection: int = 1) -> None:
        self.model_selection = model_selection
        self._segmenter = None

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        segmenter = self._get_segmenter()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = segmenter.process(rgb)
        raw_mask = getattr(result, "segmentation_mask", None)
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
        import mediapipe as mp

        self._segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=self.model_selection
        )
        return self._segmenter


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
