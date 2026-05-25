"""Lightweight body segmentation helpers."""

from __future__ import annotations

import cv2
import numpy as np


MASK_THRESHOLD = 0.45
MORPH_KERNEL_SIZE = 5


class BodySegmenter:
    """Small wrapper around MediaPipe Selfie Segmentation."""

    def __init__(self, model_selection: int = 1) -> None:
        self.model_selection = model_selection
        self._segmenter = None

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a cleaned binary body mask with values 0 or 255."""
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
        return mask

    def _get_segmenter(self):
        if self._segmenter is not None:
            return self._segmenter
        import mediapipe as mp

        self._segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=self.model_selection
        )
        return self._segmenter


def estimate_mask_width(
    body_mask: np.ndarray | None,
    y: float,
    center_x: float,
    fallback_width: float,
    *,
    band: int = 8,
    max_width_multiplier: float = 1.8,
) -> float:
    """
    Estimate horizontal body width from the segmentation mask near a scanline.

    The search is constrained around the landmark center to avoid grabbing
    background fragments or another person.
    """
    if body_mask is None or body_mask.size == 0 or fallback_width <= 0:
        return fallback_width

    height, width = body_mask.shape[:2]
    y_i = int(round(y))
    if y_i < 0 or y_i >= height:
        return fallback_width

    y1 = max(0, y_i - band)
    y2 = min(height, y_i + band + 1)
    half_window = max(int(fallback_width * max_width_multiplier), 12)
    x1 = max(0, int(round(center_x - half_window)))
    x2 = min(width, int(round(center_x + half_window + 1)))
    if x2 <= x1:
        return fallback_width

    band_mask = body_mask[y1:y2, x1:x2]
    active_columns = np.where(np.any(band_mask > 0, axis=0))[0]
    if active_columns.size < 2:
        return fallback_width

    mask_width = float(active_columns[-1] - active_columns[0])
    lower_bound = fallback_width * 0.55
    upper_bound = fallback_width * max_width_multiplier
    if mask_width < lower_bound or mask_width > upper_bound:
        return fallback_width
    return mask_width


def overlay_body_mask(
    image_bgr: np.ndarray,
    body_mask: np.ndarray | None,
    color: tuple[int, int, int] = (40, 180, 200),
    alpha: float = 0.22,
) -> np.ndarray:
    """Blend a segmentation mask over an image for debugging/visualization."""
    if body_mask is None or body_mask.size == 0:
        return image_bgr.copy()
    output = image_bgr.copy()
    color_layer = np.zeros_like(output)
    color_layer[:, :] = color
    mask_bool = body_mask > 0
    output[mask_bool] = cv2.addWeighted(output, 1.0 - alpha, color_layer, alpha, 0)[mask_bool]
    return output
