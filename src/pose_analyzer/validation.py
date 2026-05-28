"""Pose validation checks for stable body measurements."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .utils import LandmarkPoint, PoseLandmark, get_distance


MAX_SHOULDER_TILT_RATIO = 0.18
MAX_HIP_TILT_RATIO = 0.16
MIN_AVERAGE_VISIBILITY = 0.55
MIN_BODY_COVERAGE = 0.015
FRAME_MARGIN_RATIO = 0.025
MIN_TORSO_LENGTH_PX = 24.0


@dataclass
class PoseValidationResult:
    valid: bool
    score: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "score": round(self.score, 3),
            "reasons": self.reasons,
        }


def validate_pose(
    landmarks: list[LandmarkPoint],
    image_shape: tuple[int, int, int],
    body_mask: np.ndarray | None,
    visibility_threshold: float,
) -> PoseValidationResult:
    """Return a simple pose validity score for measurement use."""
    height, width = image_shape[:2]
    reasons: list[str] = []
    penalties: list[float] = []

    ls = landmarks[int(PoseLandmark.LEFT_SHOULDER)]
    rs = landmarks[int(PoseLandmark.RIGHT_SHOULDER)]
    lh = landmarks[int(PoseLandmark.LEFT_HIP)]
    rh = landmarks[int(PoseLandmark.RIGHT_HIP)]

    shoulder_width = get_distance(ls, rs)
    hip_width = get_distance(lh, rh)
    torso_length = abs(((lh.y + rh.y) / 2.0) - ((ls.y + rs.y) / 2.0))

    if torso_length < MIN_TORSO_LENGTH_PX:
        reasons.append("torso too small")
        penalties.append(0.35)

    shoulder_tilt = abs(ls.y - rs.y) / max(shoulder_width, 1.0)
    if shoulder_tilt > MAX_SHOULDER_TILT_RATIO:
        reasons.append("shoulders tilted")
        penalties.append(min(0.40, shoulder_tilt + 0.05))

    hip_tilt = abs(lh.y - rh.y) / max(hip_width, 1.0)
    if hip_tilt > MAX_HIP_TILT_RATIO:
        reasons.append("hips tilted")
        penalties.append(min(0.40, hip_tilt + 0.05))

    average_visibility = float(np.mean([min(p.visibility, p.presence) for p in (ls, rs, lh, rh)]))
    if average_visibility < MIN_AVERAGE_VISIBILITY:
        reasons.append("low landmark visibility")
        penalties.append(MIN_AVERAGE_VISIBILITY - average_visibility)

    if body_mask is not None and body_mask.size:
        coverage = float(np.count_nonzero(body_mask)) / float(body_mask.shape[0] * body_mask.shape[1])
        if coverage < MIN_BODY_COVERAGE:
            reasons.append("low body mask coverage")
            penalties.append(MIN_BODY_COVERAGE - coverage)

    margin_x = width * FRAME_MARGIN_RATIO
    margin_y = height * FRAME_MARGIN_RATIO
    for point in (ls, rs, lh, rh):
        if point.x <= margin_x or point.x >= width - margin_x or point.y <= margin_y or point.y >= height - margin_y:
            reasons.append("body near frame edge")
            penalties.append(0.2)
            break

    shoulder_hip_ratio = shoulder_width / max(hip_width, 1.0)
    if shoulder_hip_ratio > 2.25 or shoulder_hip_ratio < 0.45:
        reasons.append("sideways or distorted body proportions")
        penalties.append(0.25)

    score = max(0.0, min(1.0, 1.0 - sum(penalties)))
    valid = score > 0.65 and average_visibility >= visibility_threshold
    return PoseValidationResult(valid=valid, score=round(score, 3), reasons=reasons)
