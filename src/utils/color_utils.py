"""
Color utilities: dominant color extraction and mapping to human-friendly names.

Algorithm:
 - Input: BGR numpy image (crop of detected clothing)
 - Convert to HSV, optional median blur
 - Reshape pixels and run KMeans (k=3 default) on HSV
 - Pick largest cluster by pixel count as dominant
 - Convert centroid to RGB and map to the nearest name in the canonical palette.

The palette keys are locked to the DB item color vocabulary so that vision
output can be compared directly against item.color values and the ColourAgent
compatibility matrix.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple, Optional
import cv2
import numpy as np

# Canonical palette — keys exactly match the DB item color vocabulary and the
# ColourRecommenderAgent compatibility matrix.  Mapping vision output to this
# palette ensures detected_color always aligns with what the agent expects.
FALLBACK_PALETTE = {
    "black":    (10,  10,  10),
    "white":    (255, 255, 255),
    "gray":     (128, 128, 128),
    "red":      (220,  20,  60),
    "burgundy": (128,   0,  32),
    "pink":     (255, 105, 180),
    "orange":   (255, 165,   0),
    "yellow":   (255, 215,   0),
    "beige":    (245, 220, 180),
    "cream":    (255, 253, 208),
    "brown":    (150,  75,   0),
    "olive":    (107, 142,  35),
    "green":    ( 34, 139,  34),
    "teal":     (  0, 128, 128),
    "blue":     ( 30, 144, 255),
    "navy":     (  0,   0, 128),
    "purple":   (128,   0, 128),
    "coral":    (255, 127,  80),
}


@lru_cache(maxsize=1)
def _reference_palette() -> Dict[str, Tuple[int, int, int]]:
    """Return the canonical clothing colour palette (aligned with DB vocabulary)."""
    return dict(FALLBACK_PALETTE)


def _rgb_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return float((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _name_from_rgb(rgb: Tuple[int, int, int]) -> str:
    # Find nearest palette entry by Euclidean distance in RGB space.
    best_name = ""
    best_d = float("inf")
    for name, pal in _reference_palette().items():
        d = _rgb_distance(rgb, pal)
        if d < best_d:
            best_d = d
            best_name = name
    return best_name


def get_dominant_color_name(
    crop_bgr: np.ndarray,
    k: int = 3,
    blur_ksize: int = 5,
    resize_max: int = 200,
) -> Tuple[Optional[Tuple[int, int, int]], Optional[str]]:
    """Return (dominant_rgb, name) for the given BGR crop.

    dominant_rgb is an (R,G,B) tuple in 0-255 space. Returns (None, None)
    when crop is empty or invalid.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None, None

    h, w = crop_bgr.shape[:2]
    if h < 2 or w < 2:
        return None, None

    # Downscale to speed up KMeans while keeping aspect ratio
    scale = 1.0
    if max(h, w) > resize_max:
        scale = resize_max / float(max(h, w))
        crop = cv2.resize(crop_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        crop = crop_bgr.copy()

    # Optional blur to reduce noise
    if blur_ksize and blur_ksize > 1:
        crop = cv2.medianBlur(crop, blur_ksize)

    # Convert to HSV and use H,S,V as features (float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Reshape to Nx3
    pixels = hsv.reshape((-1, 3)).astype(np.float32)

    # Criteria for kmeans: (type, max_iter, epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
    flags = cv2.KMEANS_PP_CENTERS

    # If there are fewer pixels than k, reduce k
    k_used = max(1, min(k, pixels.shape[0]))

    try:
        _, labels, centers = cv2.kmeans(pixels, k_used, None, criteria, 3, flags)
    except Exception:
        # fallback: compute mean HSV
        centers = np.array([np.mean(pixels, axis=0)])
        labels = np.zeros((pixels.shape[0], 1), dtype=np.int32)

    # Count pixels per cluster
    labels = labels.flatten()
    counts = np.bincount(labels, minlength=centers.shape[0])
    dominant_idx = int(np.argmax(counts))
    dom_hsv = centers[dominant_idx]

    # Convert HSV center (float) back to uint8 and to BGR then RGB
    dom_hsv_u8 = np.uint8([[dom_hsv]])  # shape (1,1,3)
    dom_bgr_u8 = cv2.cvtColor(dom_hsv_u8, cv2.COLOR_HSV2BGR)[0][0]
    dom_rgb = (int(dom_bgr_u8[2]), int(dom_bgr_u8[1]), int(dom_bgr_u8[0]))

    name = _name_from_rgb(dom_rgb)
    return dom_rgb, name
