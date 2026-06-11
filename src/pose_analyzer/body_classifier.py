"""Small rule-based body-shape classifier with gender-specific labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .utils import safe_ratio


@dataclass(frozen=True)
class BodyShapeThresholds:
    """Configurable body-shape thresholds."""

    balanced_delta: float = 0.10
    pronounced_delta: float = 0.18
    hourglass_waist_max: float = 0.82
    apple_oval_waist_min: float = 0.92


def classify_body_shape(
    measurements: Dict[str, float],
    thresholds: BodyShapeThresholds | None = None,
    gender: str = "",
) -> Tuple[str, float]:
    """
    Classify body shape from normalised width measurements.

    Gender-specific labels:
      female → hourglass, pear, triangle, rectangle, inverted_triangle, apple
      male   → trapezoid, rectangle, inverted_triangle, triangle, oval
      (empty/unknown) → hourglass, triangle, rectangle, inverted_triangle, apple
    """
    thresholds = thresholds or BodyShapeThresholds()
    shoulder = float(measurements.get("shoulder_width", 0.0))
    hip = float(measurements.get("hip_width", 0.0))
    waist = float(measurements.get("waist_width", 0.0))

    if min(shoulder, hip) <= 0.0:
        return "unknown", 0.0

    shoulder_hip_ratio = measurements.get(
        "shoulder_hip_ratio",
        safe_ratio(shoulder, hip, default=1.0),
    )
    balance_delta = abs(shoulder_hip_ratio - 1.0)
    waist_hip_ratio = safe_ratio(waist, hip, default=1.0) if waist > 0.0 else 1.0

    g = gender.lower().strip()
    is_female = g in ("female", "f", "woman")
    is_male = g in ("male", "m", "man")

    # ── Round / Apple / Oval: waist is nearly as wide as hips ─────────────
    if waist > 0.0 and waist_hip_ratio >= thresholds.apple_oval_waist_min:
        conf = round(min(0.95, 0.70 + (waist_hip_ratio - thresholds.apple_oval_waist_min) * 5), 3)
        label = "oval" if is_male else "apple"
        return label, conf

    # ── Balanced shoulder / hip ────────────────────────────────────────────
    if balance_delta <= thresholds.balanced_delta:
        if not is_male and waist > 0.0 and waist_hip_ratio <= thresholds.hourglass_waist_max:
            confidence = 0.72 + (thresholds.hourglass_waist_max - waist_hip_ratio)
            return "hourglass", round(min(confidence, 0.98), 3)
        return "rectangle", round(min(0.95, 0.72 + thresholds.balanced_delta - balance_delta), 3)

    # ── Shoulder-dominant ─────────────────────────────────────────────────
    if shoulder_hip_ratio > 1.0:
        if balance_delta >= thresholds.pronounced_delta:
            confidence = 0.70 + (balance_delta - thresholds.pronounced_delta)
            return "inverted_triangle", round(min(confidence, 0.98), 3)
        # Mild shoulder dominance (balanced_delta < delta < pronounced_delta)
        if is_male:
            confidence = 0.68 + (balance_delta - thresholds.balanced_delta)
            return "trapezoid", round(min(confidence, 0.98), 3)
        # Female mild shoulder dominance → inverted_triangle (softer confidence)
        confidence = 0.64 + (balance_delta - thresholds.balanced_delta)
        return "inverted_triangle", round(min(confidence, 0.98), 3)

    # ── Hip-dominant ──────────────────────────────────────────────────────
    if balance_delta >= thresholds.pronounced_delta:
        confidence = 0.70 + (balance_delta - thresholds.pronounced_delta)
        return "triangle", round(min(confidence, 0.98), 3)
    # Mild hip dominance
    if is_female:
        confidence = 0.68 + (balance_delta - thresholds.balanced_delta)
        return "pear", round(min(confidence, 0.98), 3)
    confidence = 0.64 + (balance_delta - thresholds.balanced_delta)
    return "triangle", round(min(confidence, 0.98), 3)
