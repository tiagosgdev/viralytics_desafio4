"""Production-oriented MediaPipe pose analysis for body measurements."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os

import cv2
import numpy as np

from .body_classifier import BodyShapeThresholds, classify_body_shape
from .segmentation import BodySegmenter, estimate_mask_width
from .smoothing import MeasurementSmoother
from .utils import (
    LandmarkPoint,
    PoseLandmark,
    average_confidence,
    get_distance,
    landmark_is_reliable,
    midpoint,
    safe_ratio,
)
from .validation import validate_pose
from .visualization import draw_pose_overlay


DEFAULT_MODEL_RELATIVE_PATH = Path("models") / "weights" / "mediapipe" / "pose_landmarker_heavy.task"
DEFAULT_NUM_POSES = 4
MIN_REQUIRED_KEYPOINTS = 8


@dataclass
class PoseAnalysisResult:
    """Structured output for app integration."""

    measurements: dict[str, float]
    body_shape: str
    landmarks_detected: int
    confidence: float
    landmarks: list[dict[str, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pose_validation: dict[str, Any] = field(default_factory=dict)
    body_mask: np.ndarray | None = None
    annotated_image: np.ndarray | None = None

    def to_dict(self, include_landmarks: bool = True) -> dict[str, Any]:
        payload = {
            "measurements": self.measurements,
            "body_shape": self.body_shape,
            "landmarks_detected": self.landmarks_detected,
            "confidence": self.confidence,
            "pose_validation": self.pose_validation,
            "warnings": self.warnings,
        }
        if include_landmarks:
            payload["landmarks"] = self.landmarks
        return payload


class PoseAnalyzer:
    """MediaPipe Tasks API based pose + fashion body-shape analyzer."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        min_pose_detection_confidence: float = 0.55,
        min_pose_presence_confidence: float = 0.50,
        min_tracking_confidence: float = 0.50,
        visibility_threshold: float = 0.50,
        presence_threshold: float = 0.50,
        num_poses: int = DEFAULT_NUM_POSES,
        output_segmentation_masks: bool = False,
        enable_segmentation: bool = True,
        smoothing_window: int = 5,
        body_shape_thresholds: BodyShapeThresholds | None = None,
    ) -> None:
        self.model_path = Path(model_path or self._default_model_path())
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_pose_presence_confidence = min_pose_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.visibility_threshold = visibility_threshold
        self.presence_threshold = presence_threshold
        self.num_poses = num_poses
        self.output_segmentation_masks = output_segmentation_masks
        self.enable_segmentation = enable_segmentation
        self.body_shape_thresholds = body_shape_thresholds or BodyShapeThresholds()
        self._segmenter = BodySegmenter() if enable_segmentation else None
        self._smoother = MeasurementSmoother(window_size=smoothing_window)
        self._landmarker = None
        self._mp = None

    def is_available(self) -> bool:
        """Return True when both the model asset and MediaPipe import are available."""
        if not self.model_path.exists():
            return False
        try:
            self._import_mediapipe()
        except ImportError:
            return False
        return True

    def analyze(
        self,
        image_bgr: np.ndarray,
        *,
        draw_overlay: bool = True,
        include_landmarks: bool = True,
    ) -> PoseAnalysisResult:
        """Run pose detection, measurements, and body-shape classification."""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Input image is empty")

        result = self._detect(image_bgr)
        pose_index = self._select_primary_pose(result)
        warnings: list[str] = []

        if pose_index is None:
            return PoseAnalysisResult(
                measurements=self._empty_measurements(),
                body_shape="unknown",
                landmarks_detected=0,
                confidence=0.0,
                landmarks=[],
                pose_validation={"valid": False, "score": 0.0, "reasons": ["no reliable pose detected"]},
                warnings=["No reliable pose detected"],
                annotated_image=image_bgr.copy() if draw_overlay else None,
            )

        landmarks = self._normalized_landmarks_to_pixels(
            result.pose_landmarks[pose_index],
            image_bgr.shape,
        )
        visible_indices = self._visible_landmark_indices(landmarks)
        if len(visible_indices) < MIN_REQUIRED_KEYPOINTS:
            warnings.append("Pose detected, but too few landmarks were reliable for robust measurements")

        body_mask = self._segment_body(image_bgr, warnings)
        pose_validation = validate_pose(
            landmarks,
            image_bgr.shape,
            body_mask,
            visibility_threshold=self.visibility_threshold,
        )
        warnings.extend(pose_validation.reasons)

        measurements, measurement_points, measurement_warnings = self._compute_measurements(
            landmarks,
            body_mask=body_mask,
        )
        warnings.extend(measurement_warnings)
        pose_confidence = average_confidence([landmarks[idx] for idx in visible_indices])
        measurements_valid = self._measurements_are_valid(measurements, measurement_points)

        if measurements_valid and pose_validation.valid:
            measurements = self._smoother.update(measurements)
            body_shape, shape_confidence = classify_body_shape(measurements, self.body_shape_thresholds)
            overall_confidence = round(
                max(
                    0.0,
                    min(
                        1.0,
                        (pose_confidence * 0.35)
                        + (shape_confidence * 0.30)
                        + (pose_validation.score * 0.35),
                    ),
                ),
                3,
            )
        else:
            body_shape = "unknown"
            warnings.append("Body-shape classification skipped because pose quality or required landmarks were incomplete")
            overall_confidence = round(max(0.0, min(0.35, pose_confidence * pose_validation.score * 0.35)), 3)

        annotated = None
        if draw_overlay:
            annotated = draw_pose_overlay(
                image_bgr,
                landmarks,
                measurement_points=measurement_points,
                visible_landmarks=visible_indices,
                body_mask=body_mask,
                body_shape=body_shape,
                confidence=overall_confidence,
            )

        landmark_payload = []
        if include_landmarks:
            landmark_payload = [
                {
                    "index": idx,
                    "name": PoseLandmark(idx).name.lower(),
                    "x": round(point.x, 3),
                    "y": round(point.y, 3),
                    "z": round(point.z, 3),
                    "visibility": round(point.visibility, 3),
                    "presence": round(point.presence, 3),
                }
                for idx, point in enumerate(landmarks)
            ]

        return PoseAnalysisResult(
            measurements=measurements,
            body_shape=body_shape,
            landmarks_detected=len(visible_indices),
            confidence=overall_confidence,
            landmarks=landmark_payload,
            pose_validation=pose_validation.to_dict(),
            warnings=warnings,
            body_mask=body_mask,
            annotated_image=annotated,
        )

    def draw_skeleton(self, image_bgr: np.ndarray, analysis: PoseAnalysisResult) -> np.ndarray:
        """Return the annotated image from an existing analysis result if available."""
        if analysis.annotated_image is not None:
            return analysis.annotated_image
        return image_bgr.copy()

    def _detect(self, image_bgr: np.ndarray):
        mp = self._import_mediapipe()
        landmarker = self._get_landmarker()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return landmarker.detect(mp_image)

    def _get_landmarker(self):
        if self._landmarker is not None:
            return self._landmarker
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Pose Landmarker model not found at {self.model_path}. "
                "Set POSE_LANDMARKER_MODEL_PATH or add the .task asset to models/weights/mediapipe/."
            )

        mp = self._import_mediapipe()
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=self.output_segmentation_masks,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        return self._landmarker

    def _import_mediapipe(self):
        if self._mp is not None:
            return self._mp
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "mediapipe is required for PoseAnalyzer. Install the project requirements first."
            ) from exc
        self._mp = mp
        return mp

    def _normalized_landmarks_to_pixels(
        self,
        landmarks: list[Any],
        image_shape: tuple[int, int, int],
    ) -> list[LandmarkPoint]:
        height, width = image_shape[:2]
        output: list[LandmarkPoint] = []
        for landmark in landmarks:
            output.append(
                LandmarkPoint(
                    x=float(landmark.x) * width,
                    y=float(landmark.y) * height,
                    z=float(getattr(landmark, "z", 0.0)),
                    visibility=float(getattr(landmark, "visibility", 1.0)),
                    presence=float(getattr(landmark, "presence", 1.0)),
                )
            )
        return output

    def _visible_landmark_indices(self, landmarks: list[LandmarkPoint]) -> list[int]:
        return [
            idx for idx, landmark in enumerate(landmarks)
            if landmark_is_reliable(
                landmark,
                visibility_threshold=self.visibility_threshold,
                presence_threshold=self.presence_threshold,
            )
        ]

    def _select_primary_pose(self, result) -> int | None:
        poses = getattr(result, "pose_landmarks", None) or []
        if not poses:
            return None

        best_index = None
        best_score = -1.0
        for idx, pose_landmarks in enumerate(poses):
            reliable = [
                landmark for landmark in pose_landmarks
                if float(getattr(landmark, "visibility", 1.0)) >= self.visibility_threshold
                and float(getattr(landmark, "presence", 1.0)) >= self.presence_threshold
            ]
            if len(reliable) < MIN_REQUIRED_KEYPOINTS:
                continue
            xs = [float(landmark.x) for landmark in reliable]
            ys = [float(landmark.y) for landmark in reliable]
            area = max(xs) - min(xs)
            area *= max(ys) - min(ys)
            confidence = float(np.mean([
                min(float(getattr(landmark, "visibility", 1.0)), float(getattr(landmark, "presence", 1.0)))
                for landmark in reliable
            ]))
            score = area * confidence
            if score > best_score:
                best_index = idx
                best_score = score

        return best_index

    def _compute_measurements(
        self,
        landmarks: list[LandmarkPoint],
        body_mask: np.ndarray | None = None,
    ) -> tuple[dict[str, float], dict[str, LandmarkPoint], list[str]]:
        warnings: list[str] = []

        left_shoulder = self._require_landmark(landmarks, PoseLandmark.LEFT_SHOULDER, warnings)
        right_shoulder = self._require_landmark(landmarks, PoseLandmark.RIGHT_SHOULDER, warnings)
        left_hip = self._require_landmark(landmarks, PoseLandmark.LEFT_HIP, warnings)
        right_hip = self._require_landmark(landmarks, PoseLandmark.RIGHT_HIP, warnings)

        if not all((left_shoulder, right_shoulder, left_hip, right_hip)):
            return self._empty_measurements(), {}, warnings

        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid = midpoint(left_hip, right_hip)
        torso_length = get_distance(shoulder_mid, hip_mid)

        if torso_length <= 1e-6:
            warnings.append("Torso length collapsed to zero; returning empty measurements")
            return self._empty_measurements(), {}, warnings

        landmark_shoulder_width = get_distance(left_shoulder, right_shoulder)
        landmark_hip_width = get_distance(left_hip, right_hip)
        shoulder_width = estimate_mask_width(
            body_mask,
            shoulder_mid.y,
            shoulder_mid.x,
            landmark_shoulder_width,
        )
        hip_width = estimate_mask_width(
            body_mask,
            hip_mid.y,
            hip_mid.x,
            landmark_hip_width,
        )

        measurements = {
            "shoulder_width": round(safe_ratio(shoulder_width, torso_length), 4),
            "hip_width": round(safe_ratio(hip_width, torso_length), 4),
            "shoulder_hip_ratio": round(safe_ratio(shoulder_width, hip_width, default=1.0), 4),
        }

        measurement_points = {
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
            "left_hip": left_hip,
            "right_hip": right_hip,
            "shoulder_mid": shoulder_mid,
            "hip_mid": hip_mid,
        }
        return measurements, measurement_points, warnings

    def _segment_body(self, image_bgr: np.ndarray, warnings: list[str]) -> np.ndarray | None:
        if self._segmenter is None:
            return None
        try:
            return self._segmenter.segment(image_bgr)
        except Exception as exc:
            warnings.append(f"Body segmentation failed: {exc}")
            return None

    def _require_landmark(
        self,
        landmarks: list[LandmarkPoint],
        idx: PoseLandmark,
        warnings: list[str],
    ) -> LandmarkPoint | None:
        landmark = landmarks[int(idx)]
        if landmark_is_reliable(
            landmark,
            visibility_threshold=self.visibility_threshold,
            presence_threshold=self.presence_threshold,
        ):
            return landmark
        warnings.append(f"Required landmark {idx.name.lower()} was not reliable")
        return None

    def _best_available_ankle(
        self,
        landmarks: list[LandmarkPoint],
        primary_idx: PoseLandmark,
        fallback_idx: PoseLandmark,
        warnings: list[str],
    ) -> LandmarkPoint | None:
        primary = landmarks[int(primary_idx)]
        if landmark_is_reliable(
            primary,
            visibility_threshold=self.visibility_threshold,
            presence_threshold=self.presence_threshold,
        ):
            return primary
        fallback = landmarks[int(fallback_idx)]
        if landmark_is_reliable(
            fallback,
            visibility_threshold=self.visibility_threshold,
            presence_threshold=self.presence_threshold,
        ):
            warnings.append(f"Using {fallback_idx.name.lower()} as fallback for {primary_idx.name.lower()}")
            return fallback
        warnings.append(f"Required lower-body landmark {primary_idx.name.lower()} was not reliable")
        return None

    def _default_model_path(self) -> str:
        env_path = os.getenv("POSE_LANDMARKER_MODEL_PATH")
        if env_path:
            return env_path
        project_root = Path(__file__).resolve().parents[2]
        return str(project_root / DEFAULT_MODEL_RELATIVE_PATH)

    @staticmethod
    def _measurements_are_valid(
        measurements: dict[str, float],
        measurement_points: dict[str, LandmarkPoint],
    ) -> bool:
        if not measurement_points:
            return False
        required_measurements = (
            "shoulder_width",
            "hip_width",
            "shoulder_hip_ratio",
        )
        return all(float(measurements.get(key, 0.0)) > 0.0 for key in required_measurements)

    @staticmethod
    def _empty_measurements() -> dict[str, float]:
        return {
            "shoulder_width": 0.0,
            "hip_width": 0.0,
            "shoulder_hip_ratio": 0.0,
        }
