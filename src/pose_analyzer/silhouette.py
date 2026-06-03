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

    # Outer x-span of the shoulder landmarks (image-space left edge, right edge).
    shoulder_left  = min(ls.x, rs.x)
    shoulder_right = max(ls.x, rs.x)

    def _body_bounds(t: float, pad_factor: float, clamp_to_shoulders: bool = False) -> tuple[float, float]:
        """Image-space left/right body boundary at torso fraction t (0=shoulder, 1=hip).

        pad_factor  – outward allowance as a fraction of torso_length.
        clamp_to_shoulders – when True, the outer edges are capped at the shoulder
            x-positions (plus a tiny tolerance).  Use for the chest scanline: the
            upper arm starts at the shoulder joint, so anything beyond that x is arm.
        """
        x_a = ls.x + t * (lh.x - ls.x)
        x_b = rs.x + t * (rh.x - rs.x)
        pad = torso_length * pad_factor
        left_b  = min(x_a, x_b) - pad
        right_b = max(x_a, x_b) + pad
        if clamp_to_shoulders:
            arm_tol = torso_length * 0.03   # 3% tolerance inward from shoulder joint
            left_b  = max(left_b,  shoulder_left  - arm_tol)
            right_b = min(right_b, shoulder_right + arm_tol)
        return left_b, right_b

    # Torso scanlines.
    # chest uses clamp_to_shoulders=True — the arm begins at the shoulder joint,
    # so the search window must not extend beyond that x-position.
    # hip uses a large pad because LEFT_HIP/RIGHT_HIP sit at the joint, well inside
    # the actual soft-tissue edge of the hips.
    torso_scans = [
        ("shoulder_width", shoulder_mid.y,                       0.0,  0.10, False),
        ("chest_width",    shoulder_mid.y + torso_length * 0.25, 0.25, 0.10, True),
        ("waist_width",    shoulder_mid.y + torso_length * 0.55, 0.55, 0.08, False),
        ("hip_width",      hip_mid.y,                            1.0,  0.26, False),
    ]

    widths: dict[str, float] = {}
    scanlines: list[SilhouetteScanline] = []

    for name, y, t, pad_factor, clamp in torso_scans:
        left_b, right_b = _body_bounds(t, pad_factor, clamp)
        line = _sample_width(mask, name, y, left_b, right_b, torso_length)
        if line is None:
            continue
        widths[name] = line.normalized_width
        scanlines.append(line)

    # Thigh — sample each leg half independently, report the average as one thigh.
    # Splitting at the body centre-line keeps the two thighs from being merged.
    # The landmark is the hip joint which sits inside the thigh outline, so a
    # larger pad is needed (same reasoning as for hip_width above).
    thigh_y = hip_mid.y + max(0.0, knee_mid.y - hip_mid.y) * 0.35
    body_cx = hip_mid.x  # split point between left and right thigh
    thigh_pad = torso_length * 0.20
    thigh_reach = abs(lh.x - rh.x) / 2.0 + thigh_pad

    r_thigh = _sample_width(mask, "thigh_r", thigh_y, body_cx, body_cx + thigh_reach, torso_length)
    l_thigh = _sample_width(mask, "thigh_l", thigh_y, body_cx - thigh_reach, body_cx, torso_length)

    valid_thighs = [tl for tl in (r_thigh, l_thigh) if tl is not None]
    if valid_thighs:
        avg_px = sum(tl.width_px for tl in valid_thighs) / len(valid_thighs)
        avg_norm = safe_ratio(avg_px, torso_length)
        rep = valid_thighs[0]
        widths["thigh_width"] = avg_norm
        scanlines.append(SilhouetteScanline(
            name="thigh_width",
            y=rep.y,
            width_px=avg_px,
            left_x=rep.left_x,
            right_x=rep.right_x,
            normalized_width=avg_norm,
        ))

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
    left_bound: float,
    right_bound: float,
    torso_length: float,
) -> SilhouetteScanline | None:
    """Sample the mask within [left_bound, right_bound] at height y."""
    img_h, img_w = mask.shape[:2]
    y_i = int(round(y))
    if y_i < 0 or y_i >= img_h:
        return None

    y1 = max(0, y_i - SCANLINE_BAND)
    y2 = min(img_h, y_i + SCANLINE_BAND + 1)
    x1 = max(0, int(round(left_bound)))
    x2 = min(img_w, int(round(right_bound)) + 1)
    if x2 <= x1:
        return None

    band = mask[y1:y2, x1:x2]
    active_cols = np.where(np.any(band > 0, axis=0))[0]
    if active_cols.size < 2:
        return None

    left_x = float(x1 + active_cols[0])
    right_x = float(x1 + active_cols[-1])
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
