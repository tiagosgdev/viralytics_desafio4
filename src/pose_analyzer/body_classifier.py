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


def classify_body_shape(
    measurements: Dict[str, float],
    thresholds: BodyShapeThresholds | None = None,
) -> Tuple[str, float]:
    """
    Classify fashion body shape from normalized proportions.

    Heuristic summary:
    - `rectangle`: shoulders and hips are balanced.
    - `hourglass`: shoulders and hips are balanced, waist clearly narrower.
    - `inverted_triangle`: shoulders are materially wider than hips.
    - `triangle`: hips are materially wider than shoulders.
    """
    thresholds = thresholds or BodyShapeThresholds()

    shoulder = measurements.get("shoulder_width", 0.0)
    hip = measurements.get("hip_width", 0.0)
    waist = measurements.get("waist_width", 0.0)

    # Classification is only meaningful when the core measurements exist.
    if min(shoulder, hip) <= 0.0:
        return "unknown", 0.0

    shoulder_hip_ratio = measurements.get(
        "shoulder_hip_ratio",
        safe_ratio(shoulder, hip, default=1.0),
    )
    balance_delta = abs(shoulder_hip_ratio - 1.0)

    if balance_delta <= thresholds.balanced_delta:
        waist_hip_ratio = measurements.get(
            "waist_hip_ratio",
            safe_ratio(waist, hip, default=1.0),
        )
        if waist > 0.0 and waist_hip_ratio <= thresholds.hourglass_waist_max:
            confidence = 0.72 + (thresholds.hourglass_waist_max - waist_hip_ratio)
            return "hourglass", round(min(confidence, 0.99), 3)
        return "rectangle", round(min(0.95, 0.72 + (thresholds.balanced_delta - balance_delta)), 3)

    if shoulder_hip_ratio >= 1.0 + thresholds.pronounced_delta:
        confidence = 0.70 + (shoulder_hip_ratio - (1.0 + thresholds.pronounced_delta))
        return "inverted_triangle", round(min(confidence, 0.98), 3)

    if shoulder_hip_ratio <= 1.0 - thresholds.pronounced_delta:
        confidence = 0.70 + ((1.0 - thresholds.pronounced_delta) - shoulder_hip_ratio)
        return "triangle", round(min(confidence, 0.98), 3)

    return "rectangle", 0.64
