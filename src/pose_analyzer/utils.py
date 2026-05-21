"""Shared geometry helpers and MediaPipe landmark constants."""

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
DEFAULT_VISIBILITY_THRESHOLD = 0.50
DEFAULT_PRESENCE_THRESHOLD = 0.50


@dataclass(frozen=True)
class LandmarkPoint:
    """2D landmark projected into image coordinates."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


def get_distance(p1: LandmarkPoint | Sequence[float], p2: LandmarkPoint | Sequence[float]) -> float:
    """Euclidean distance in 2D image space."""
    x1, y1 = point_xy(p1)
    x2, y2 = point_xy(p2)
    return math.hypot(x2 - x1, y2 - y1)


def point_xy(point: LandmarkPoint | Sequence[float]) -> tuple[float, float]:
    """Return the 2D coordinates for a point-like value."""
    if isinstance(point, LandmarkPoint):
        return point.x, point.y
    return float(point[0]), float(point[1])


def midpoint(
    p1: LandmarkPoint | Sequence[float],
    p2: LandmarkPoint | Sequence[float],
) -> LandmarkPoint:
    """Compute the midpoint between two landmarks."""
    x1, y1 = point_xy(p1)
    x2, y2 = point_xy(p2)
    if isinstance(p1, LandmarkPoint) and isinstance(p2, LandmarkPoint):
        return LandmarkPoint(
            x=(x1 + x2) / 2.0,
            y=(y1 + y2) / 2.0,
            z=(p1.z + p2.z) / 2.0,
            visibility=min(p1.visibility, p2.visibility),
            presence=min(p1.presence, p2.presence),
        )
    return LandmarkPoint(x=(x1 + x2) / 2.0, y=(y1 + y2) / 2.0)


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide with a zero guard."""
    if abs(denominator) < 1e-8:
        return default
    return numerator / denominator


def landmark_is_reliable(
    landmark: LandmarkPoint | None,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    presence_threshold: float = DEFAULT_PRESENCE_THRESHOLD,
) -> bool:
    """Check if a landmark is present and visible enough for measurement use."""
    if landmark is None:
        return False
    return (
        landmark.visibility >= visibility_threshold
        and landmark.presence >= presence_threshold
    )


def average_confidence(landmarks: Iterable[LandmarkPoint]) -> float:
    """Aggregate visibility/presence into a single confidence estimate."""
    values = [
        min(landmark.visibility, landmark.presence)
        for landmark in landmarks
    ]
    if not values:
        return 0.0
    return float(np.mean(values))
