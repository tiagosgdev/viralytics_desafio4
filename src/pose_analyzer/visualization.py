"""OpenCV visualization helpers for pose and silhouette analysis."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .segmentation import overlay_body_mask
from .silhouette import SilhouetteProfile
from .utils import LandmarkPoint, point_xy


POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28),
)

JOINT_COLOR = (57, 235, 193)
BONE_COLOR = (255, 189, 89)
MEASUREMENT_COLOR = (88, 166, 255)
SILHOUETTE_COLOR = (255, 122, 122)
TEXT_COLOR = (245, 245, 245)


def draw_pose_overlay(
    frame: np.ndarray,
    landmarks: list[LandmarkPoint],
    *,
    measurement_points: dict[str, LandmarkPoint] | None = None,
    visible_landmarks: Iterable[int] | None = None,
    body_mask: np.ndarray | None = None,
    silhouette_profile: SilhouetteProfile | None = None,
    body_shape: str | None = None,
    confidence: float | None = None,
) -> np.ndarray:
    """Draw skeleton, segmentation mask, silhouette scanlines, and labels."""
    output = overlay_body_mask(frame, body_mask)
    visible = set(visible_landmarks or range(len(landmarks)))

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx not in visible or end_idx not in visible:
            continue
        cv2.line(output, _as_int_point(landmarks[start_idx]), _as_int_point(landmarks[end_idx]), BONE_COLOR, 2, cv2.LINE_AA)

    for idx in visible:
        cv2.circle(output, _as_int_point(landmarks[idx]), 4, JOINT_COLOR, -1, cv2.LINE_AA)

    if measurement_points:
        _draw_segment(output, measurement_points.get("left_shoulder"), measurement_points.get("right_shoulder"), "Shoulders", MEASUREMENT_COLOR)
        _draw_segment(output, measurement_points.get("left_hip"), measurement_points.get("right_hip"), "Hips", MEASUREMENT_COLOR)

    if silhouette_profile is not None:
        _draw_silhouette_profile(output, silhouette_profile)

    if body_shape:
        label = body_shape.replace("_", " ").title()
        if confidence is not None:
            label += f" {confidence:.0%}"
        cv2.putText(output, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)

    return output


def _draw_silhouette_profile(frame: np.ndarray, profile: SilhouetteProfile) -> None:
    if profile.contour:
        contour = np.array(profile.contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(frame, [contour], -1, SILHOUETTE_COLOR, 1, cv2.LINE_AA)

    for scanline in profile.scanlines:
        p1 = (int(round(scanline.left_x)), int(round(scanline.y)))
        p2 = (int(round(scanline.right_x)), int(round(scanline.y)))
        cv2.line(frame, p1, p2, SILHOUETTE_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, scanline.name.replace("_width", ""), (p2[0] + 6, p2[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, SILHOUETTE_COLOR, 1, cv2.LINE_AA)


def _draw_segment(
    frame: np.ndarray,
    start: LandmarkPoint | None,
    end: LandmarkPoint | None,
    label: str,
    color: tuple[int, int, int],
) -> None:
    if start is None or end is None:
        return
    p1 = _as_int_point(start)
    p2 = _as_int_point(end)
    cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (int((p1[0] + p2[0]) / 2) + 6, int((p1[1] + p2[1]) / 2) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _as_int_point(point: LandmarkPoint) -> tuple[int, int]:
    x, y = point_xy(point)
    return int(round(x)), int(round(y))
