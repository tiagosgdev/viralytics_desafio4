"""Shared geometry helpers for pose analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

import math
import numpy as np


class PoseLandmark(IntEnum):
    """MediaPipe Pose landmark indices."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


LANDMARK_COUNT = 33


@dataclass(frozen=True)
class LandmarkPoint:
    """2D pose landmark projected into image coordinates."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0


def point_xy(point: LandmarkPoint | Sequence[float]) -> tuple[float, float]:
    if isinstance(point, LandmarkPoint):
        return point.x, point.y
    return float(point[0]), float(point[1])


def get_distance(p1: LandmarkPoint | Sequence[float], p2: LandmarkPoint | Sequence[float]) -> float:
    """Euclidean distance in 2D image space."""
    x1, y1 = point_xy(p1)
    x2, y2 = point_xy(p2)
    return math.hypot(x2 - x1, y2 - y1)


def midpoint(p1: LandmarkPoint, p2: LandmarkPoint) -> LandmarkPoint:
    """Return the midpoint between two landmarks."""
    return LandmarkPoint(
        x=(p1.x + p2.x) / 2.0,
        y=(p1.y + p2.y) / 2.0,
        z=(p1.z + p2.z) / 2.0,
        visibility=min(p1.visibility, p2.visibility),
        presence=min(p1.presence, p2.presence),
    )


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-8:
        return default
    return numerator / denominator


def landmark_is_reliable(
    landmark: LandmarkPoint | None,
    *,
    visibility_threshold: float,
    presence_threshold: float,
) -> bool:
    if landmark is None:
        return False
    return landmark.visibility >= visibility_threshold and landmark.presence >= presence_threshold


def average_confidence(landmarks: Iterable[LandmarkPoint]) -> float:
    values = [min(point.visibility, point.presence) for point in landmarks]
    return float(np.mean(values)) if values else 0.0
