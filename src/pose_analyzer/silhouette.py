"""Pose-guided silhouette profile extraction."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .utils import LandmarkPoint, PoseLandmark, midpoint, safe_ratio


SCANLINE_BAND = 6


@dataclass
class SilhouetteScanline:
    """One horizontal width sample from the body silhouette."""

    name: str
    y: float
    width_px: float
    left_x: float
    right_x: float
    normalized_width: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "y": round(self.y, 3),
            "width_px": round(self.width_px, 3),
            "left_x": round(self.left_x, 3),
            "right_x": round(self.right_x, 3),
            "normalized_width": round(self.normalized_width, 4),
        }


@dataclass
class SilhouetteProfile:
    """Width profile sampled from the segmented visible body."""

    valid: bool
    widths: dict[str, float] = field(default_factory=dict)
    scanlines: list[SilhouetteScanline] = field(default_factory=list)
    contour: list[tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "widths": {key: round(value, 4) for key, value in self.widths.items()},
            "scanlines": [line.to_dict() for line in self.scanlines],
        }


def extract_silhouette_profile(
    body_mask: np.ndarray | None,
    landmarks: list[LandmarkPoint],
) -> SilhouetteProfile:
    """Extract shoulder, chest, waist, hip, and thigh widths from the mask."""
    if body_mask is None or body_mask.size == 0 or not np.any(body_mask > 0):
        return SilhouetteProfile(valid=False)

    mask = (body_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_points: list[tuple[int, int]] = []
    if contours:
        contour = max(contours, key=cv2.contourArea)
        contour_points = [(int(point[0][0]), int(point[0][1])) for point in contour]

    ls = landmarks[int(PoseLandmark.LEFT_SHOULDER)]
    rs = landmarks[int(PoseLandmark.RIGHT_SHOULDER)]
    lh = landmarks[int(PoseLandmark.LEFT_HIP)]
    rh = landmarks[int(PoseLandmark.RIGHT_HIP)]
    lk = landmarks[int(PoseLandmark.LEFT_KNEE)]
    rk = landmarks[int(PoseLandmark.RIGHT_KNEE)]

    shoulder_mid = midpoint(ls, rs)
    hip_mid = midpoint(lh, rh)
    knee_mid = midpoint(lk, rk)
    torso_length = abs(hip_mid.y - shoulder_mid.y)
    if torso_length <= 1e-6:
        return SilhouetteProfile(valid=False, contour=contour_points)

    scan_y = {
        "shoulder_width": shoulder_mid.y,
        "chest_width": shoulder_mid.y + torso_length * 0.25,
        "waist_width": shoulder_mid.y + torso_length * 0.55,
        "hip_width": hip_mid.y,
        "thigh_width": hip_mid.y + max(0.0, knee_mid.y - hip_mid.y) * 0.35,
    }

    widths: dict[str, float] = {}
    scanlines: list[SilhouetteScanline] = []
    center_x = (shoulder_mid.x + hip_mid.x) / 2.0
    max_window = max(abs(ls.x - rs.x), abs(lh.x - rh.x), 24.0) * 1.8

    for name, y in scan_y.items():
        line = _sample_width(mask, name, y, center_x, max_window, torso_length)
        if line is None:
            continue
        widths[name] = line.normalized_width
        scanlines.append(line)

    return SilhouetteProfile(
        valid=("shoulder_width" in widths and "hip_width" in widths),
        widths=widths,
        scanlines=scanlines,
        contour=contour_points,
    )


def _sample_width(
    mask: np.ndarray,
    name: str,
    y: float,
    center_x: float,
    max_window: float,
    torso_length: float,
) -> SilhouetteScanline | None:
    height, width = mask.shape[:2]
    y_i = int(round(y))
    if y_i < 0 or y_i >= height:
        return None

    y1 = max(0, y_i - SCANLINE_BAND)
    y2 = min(height, y_i + SCANLINE_BAND + 1)
    x1 = max(0, int(round(center_x - max_window)))
    x2 = min(width, int(round(center_x + max_window + 1)))
    if x2 <= x1:
        return None

    band = mask[y1:y2, x1:x2]
    active_columns = np.where(np.any(band > 0, axis=0))[0]
    if active_columns.size < 2:
        return None

    left_x = float(x1 + active_columns[0])
    right_x = float(x1 + active_columns[-1])
    width_px = right_x - left_x
    if width_px <= 0:
        return None

    return SilhouetteScanline(
        name=name,
        y=float(y_i),
        width_px=width_px,
        left_x=left_x,
        right_x=right_x,
        normalized_width=safe_ratio(width_px, torso_length),
    )
