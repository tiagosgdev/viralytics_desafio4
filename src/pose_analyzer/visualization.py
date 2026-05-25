"""Pose skeleton and measurement rendering helpers."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .utils import LandmarkPoint, point_xy


POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (3, 7), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)

JOINT_COLOR = (57, 235, 193)
BONE_COLOR = (255, 189, 89)
MEASUREMENT_COLOR = (88, 166, 255)
TEXT_COLOR = (245, 245, 245)

def draw_pose_overlay(
    frame: np.ndarray,
    landmarks: list[LandmarkPoint],
    measurement_points: dict[str, LandmarkPoint] | None = None,
    visible_landmarks: Iterable[int] | None = None,
    body_mask: np.ndarray | None = None,
    body_shape: str | None = None,
    confidence: float | None = None,
) -> np.ndarray:
    """Draw the MediaPipe skeleton, key joints, and measurement labels."""
    output = frame.copy()
    if body_mask is not None and body_mask.size:
        color_layer = np.zeros_like(output)
        color_layer[:, :] = (40, 180, 200)
        mask_bool = body_mask > 0
        blended = cv2.addWeighted(output, 0.78, color_layer, 0.22, 0)
        output[mask_bool] = blended[mask_bool]

    visible = set(visible_landmarks or range(len(landmarks)))

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx not in visible or end_idx not in visible:
            continue
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        cv2.line(
            output,
            _as_int_point(start),
            _as_int_point(end),
            BONE_COLOR,
            2,
            cv2.LINE_AA,
        )

    for idx in visible:
        point = landmarks[idx]
        cv2.circle(output, _as_int_point(point), 4, JOINT_COLOR, -1, cv2.LINE_AA)

    if measurement_points:
        _draw_measurement_guides(output, measurement_points)

    if body_shape:
        label = body_shape.replace("_", " ").title()
        if confidence is not None:
            label += f" {confidence:.0%}"
        cv2.putText(
            output,
            label,
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

    return output


def _draw_measurement_guides(
    frame: np.ndarray,
    measurement_points: dict[str, LandmarkPoint],
) -> None:
    left_shoulder = measurement_points.get("left_shoulder")
    right_shoulder = measurement_points.get("right_shoulder")
    left_hip = measurement_points.get("left_hip")
    right_hip = measurement_points.get("right_hip")

    _draw_labeled_segment(frame, left_shoulder, right_shoulder, "Shoulders", MEASUREMENT_COLOR)
    _draw_labeled_segment(frame, left_hip, right_hip, "Hips", MEASUREMENT_COLOR)

    for name, point in (
        ("L Shoulder", left_shoulder),
        ("R Shoulder", right_shoulder),
        ("L Hip", left_hip),
        ("R Hip", right_hip),
    ):
        if point is None:
            continue
        cv2.putText(
            frame,
            name,
            (_as_int_point(point)[0] + 6, _as_int_point(point)[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )


def _draw_labeled_segment(
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
    mid_x = int((p1[0] + p2[0]) / 2)
    mid_y = int((p1[1] + p2[1]) / 2)
    cv2.putText(
        frame,
        label,
        (mid_x + 6, mid_y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def _as_int_point(point: LandmarkPoint) -> tuple[int, int]:
    x, y = point_xy(point)
    return int(round(x)), int(round(y))
