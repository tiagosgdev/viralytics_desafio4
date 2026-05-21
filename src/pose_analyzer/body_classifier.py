"""Heuristic fashion body-shape classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .utils import safe_ratio


@dataclass(frozen=True)
class BodyShapeThresholds:
    """Configurable cutoffs for coarse fashion silhouette classification."""

    balanced_delta: float = 0.10
    pronounced_delta: float = 0.18
    hourglass_waist_max: float = 0.82
    rectangle_waist_max: float = 0.92
    oval_waist_min: float = 0.97


def classify_body_shape(
    measurements: Dict[str, float],
    thresholds: BodyShapeThresholds | None = None,
) -> Tuple[str, float]:
    """
    Classify fashion body shape from normalized proportions.

    Heuristic summary:
    - `hourglass`: shoulders and hips are balanced, waist clearly narrower.
    - `rectangle`: shoulders and hips are balanced, waist reduction is modest.
    - `inverted_triangle`: shoulders are materially wider than hips.
    - `triangle`: hips are materially wider than shoulders.
    - `oval`: waist is comparatively broad relative to hips/shoulders.
    """
    thresholds = thresholds or BodyShapeThresholds()

    shoulder = measurements.get("shoulder_width", 0.0)
    hip = measurements.get("hip_width", 0.0)
    waist = measurements.get("waist_width", 0.0)
    torso = measurements.get("torso_length", 0.0)

    # Classification is only meaningful when the core measurements exist.
    if min(shoulder, hip, waist, torso) <= 0.0:
        return "unknown", 0.0

    shoulder_hip_ratio = measurements.get(
        "shoulder_hip_ratio",
        safe_ratio(shoulder, hip, default=1.0),
    )
    waist_hip_ratio = measurements.get(
        "waist_hip_ratio",
        safe_ratio(waist, hip, default=1.0),
    )
    waist_shoulder_ratio = safe_ratio(waist, shoulder, default=1.0)
    balance_delta = abs(shoulder_hip_ratio - 1.0)

    if waist_hip_ratio >= thresholds.oval_waist_min and waist_shoulder_ratio >= 0.95:
        confidence = min(0.98, 0.72 + (waist_hip_ratio - thresholds.oval_waist_min))
        return "oval", round(confidence, 3)

    if balance_delta <= thresholds.balanced_delta:
        if waist_hip_ratio <= thresholds.hourglass_waist_max:
            confidence = 0.72 + (thresholds.hourglass_waist_max - waist_hip_ratio)
            return "hourglass", round(min(confidence, 0.99), 3)
        if waist_hip_ratio <= thresholds.rectangle_waist_max:
            confidence = 0.66 + (waist_hip_ratio - thresholds.hourglass_waist_max) * 0.4
            return "rectangle", round(min(confidence, 0.95), 3)
        return "oval", round(min(0.7 + (waist_hip_ratio - thresholds.rectangle_waist_max), 0.92), 3)

    if shoulder_hip_ratio >= 1.0 + thresholds.pronounced_delta:
        confidence = 0.70 + (shoulder_hip_ratio - (1.0 + thresholds.pronounced_delta))
        return "inverted_triangle", round(min(confidence, 0.98), 3)

    if shoulder_hip_ratio <= 1.0 - thresholds.pronounced_delta:
        confidence = 0.70 + ((1.0 - thresholds.pronounced_delta) - shoulder_hip_ratio)
        return "triangle", round(min(confidence, 0.98), 3)

    if waist_hip_ratio <= thresholds.hourglass_waist_max:
        return "hourglass", 0.67
    if waist_hip_ratio >= thresholds.oval_waist_min:
        return "oval", 0.67
    return "rectangle", 0.64
