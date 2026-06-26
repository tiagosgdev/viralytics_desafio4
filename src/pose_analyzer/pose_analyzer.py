"""MediaPipe Pose Landmarker pipeline with segmentation and silhouette profile."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os

import cv2
import numpy as np

from .body_classifier import BodyShapeThresholds, classify_body_shape
from .segmentation import BodySegmenter, keep_largest_component
from .silhouette import SilhouetteProfile, extract_silhouette_profile
from .smoothing import MeasurementSmoother
from .utils import LandmarkPoint, PoseLandmark, average_confidence, get_distance, landmark_is_reliable, midpoint, safe_ratio
from .validation import validate_pose
from .visualization import draw_pose_overlay


DEFAULT_MODEL_RELATIVE_PATH = Path("models") / "weights" / "mediapipe" / "pose_landmarker_heavy.task"
DEFAULT_NUM_POSES = 4
MIN_REQUIRED_KEYPOINTS = 5


@dataclass
class PoseAnalysisResult:
    """Structured pose/body analysis output."""

    body_shape: str
    measurements: dict[str, float]
    confidence: float
    pose_validation: dict[str, Any]
    landmarks_detected: int
    silhouette: dict[str, Any] = field(default_factory=dict)
    landmarks: list[dict[str, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    body_mask: np.ndarray | None = None
    annotated_image: np.ndarray | None = None

    def to_dict(self, include_landmarks: bool = True) -> dict[str, Any]:
        payload = {
            "body_shape": self.body_shape,
            "measurements": self.measurements,
            "confidence": self.confidence,
            "pose_validation": self.pose_validation,
            "landmarks_detected": self.landmarks_detected,
            "silhouette": self.silhouette,
            "warnings": self.warnings,
        }
        if include_landmarks:
            payload["landmarks"] = self.landmarks
        return payload


class PoseAnalyzer:
    """Lightweight fashion body-shape analyzer."""

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
        self.body_shape_thresholds = body_shape_thresholds or BodyShapeThresholds()
        self._segmenter = BodySegmenter() if enable_segmentation else None
        self._smoother = MeasurementSmoother(window_size=smoothing_window)
        self._landmarker = None
        self._mp = None

    def is_available(self) -> bool:
        """Return True when MediaPipe and the pose model are available."""
        if not self.model_path.exists():
            return False
        try:
            self._import_mediapipe()
        except ImportError:
            return False
        return True

    def analyze(self, image_bgr: np.ndarray, *, draw_overlay: bool = True, include_landmarks: bool = True, user_height_cm: float | None = None, gender: str = "") -> PoseAnalysisResult:
        """Run pose detection, segmentation, validation, silhouette sampling, and classification."""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Input image is empty")

        result = self._detect(image_bgr)
        pose_index = self._select_primary_pose(result)
        if pose_index is None:
            return self._empty_result(image_bgr if draw_overlay else None, "No reliable pose detected")

        landmarks = self._normalized_landmarks_to_pixels(result.pose_landmarks[pose_index], image_bgr.shape)
        visible_indices = self._visible_landmark_indices(landmarks)
        warnings: list[str] = []
        if len(visible_indices) < MIN_REQUIRED_KEYPOINTS:
            warnings.append("Pose detected, but too few landmarks were reliable for robust measurements")

        body_mask = self._extract_pose_mask(result, pose_index)
        if body_mask is None:
            body_mask = self._segment_body(image_bgr, warnings)
        silhouette_profile = extract_silhouette_profile(body_mask, landmarks)
        pose_validation = validate_pose(landmarks, image_bgr.shape, body_mask, self.visibility_threshold)
        warnings.extend(pose_validation.reasons)

        measurements, measurement_points, measurement_warnings, torso_length_px = self._compute_measurements(landmarks, silhouette_profile)
        warnings.extend(measurement_warnings)
        pose_confidence = average_confidence([landmarks[idx] for idx in visible_indices])

        if self._measurements_are_valid(measurements, measurement_points) and pose_validation.valid:
            measurements = self._smoother.update(measurements)
            body_shape, shape_confidence = classify_body_shape(measurements, self.body_shape_thresholds, gender=gender)
            confidence = round(max(0.0, min(1.0, pose_confidence * 0.30 + pose_validation.score * 0.35 + shape_confidence * 0.35)), 3)
            if user_height_cm and user_height_cm > 0 and torso_length_px > 0:
                measurements.update(self._scale_to_cm(measurements, landmarks, torso_length_px, user_height_cm))
        else:
            body_shape = "unknown"
            confidence = round(max(0.0, min(0.35, pose_confidence * pose_validation.score * 0.35)), 3)
            warnings.append("Body-shape classification skipped because pose quality or silhouette measurements were incomplete")

        annotated = None
        if draw_overlay:
            annotated = draw_pose_overlay(
                image_bgr,
                landmarks,
                measurement_points=measurement_points,
                visible_landmarks=visible_indices,
                body_mask=body_mask,
                silhouette_profile=silhouette_profile,
                body_shape=body_shape,
                confidence=confidence,
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
            body_shape=body_shape,
            measurements=measurements,
            confidence=confidence,
            pose_validation=pose_validation.to_dict(),
            landmarks_detected=len(visible_indices),
            silhouette=silhouette_profile.to_dict(),
            landmarks=landmark_payload,
            warnings=warnings,
            body_mask=body_mask,
            annotated_image=annotated,
        )

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
        self._import_mediapipe()
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        return self._landmarker

    def _import_mediapipe(self):
        if self._mp is not None:
            return self._mp
        import mediapipe as mp

        self._mp = mp
        return mp

    def _normalized_landmarks_to_pixels(self, landmarks: list[Any], image_shape: tuple[int, int, int]) -> list[LandmarkPoint]:
        height, width = image_shape[:2]
        return [
            LandmarkPoint(
                x=float(point.x) * width,
                y=float(point.y) * height,
                z=float(getattr(point, "z", 0.0)),
                visibility=float(getattr(point, "visibility", 1.0)),
                presence=float(getattr(point, "presence", 1.0)),
            )
            for point in landmarks
        ]

    def _visible_landmark_indices(self, landmarks: list[LandmarkPoint]) -> list[int]:
        return [
            idx
            for idx, landmark in enumerate(landmarks)
            if landmark_is_reliable(landmark, visibility_threshold=self.visibility_threshold, presence_threshold=self.presence_threshold)
        ]

    def _select_primary_pose(self, result) -> int | None:
        poses = getattr(result, "pose_landmarks", None) or []
        if not poses:
            return None
        best_index = None
        best_score = -1.0
        for idx, pose_landmarks in enumerate(poses):
            reliable = [
                point for point in pose_landmarks
                if float(getattr(point, "visibility", 1.0)) >= self.visibility_threshold
                and float(getattr(point, "presence", 1.0)) >= self.presence_threshold
            ]
            if len(reliable) < MIN_REQUIRED_KEYPOINTS:
                continue
            xs = [float(point.x) for point in reliable]
            ys = [float(point.y) for point in reliable]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            confidence = float(np.mean([min(float(getattr(point, "visibility", 1.0)), float(getattr(point, "presence", 1.0))) for point in reliable]))
            score = area * confidence
            if score > best_score:
                best_index = idx
                best_score = score
        return best_index

    def _extract_pose_mask(self, result, pose_index: int) -> np.ndarray | None:
        """Extract the segmentation mask produced by the Pose Landmarker (Tasks API)."""
        masks = getattr(result, "segmentation_masks", None)
        if not masks or pose_index >= len(masks):
            return None
        try:
            mask_obj = masks[pose_index]
            raw = mask_obj.numpy_view() if hasattr(mask_obj, "numpy_view") else mask_obj
            if not isinstance(raw, np.ndarray):
                return None
            mask = (raw >= 0.5).astype(np.uint8) * 255
            kernel = np.ones((5, 5), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            return keep_largest_component(mask)
        except Exception:
            return None

    def _segment_body(self, image_bgr: np.ndarray, warnings: list[str]) -> np.ndarray | None:
        if self._segmenter is None:
            return None
        try:
            return self._segmenter.segment(image_bgr)
        except Exception as exc:
            warnings.append(f"Body segmentation failed: {exc}")
            return None

    def _compute_measurements(self, landmarks: list[LandmarkPoint], silhouette_profile: SilhouetteProfile) -> tuple[dict[str, float], dict[str, LandmarkPoint], list[str], float]:
        warnings: list[str] = []
        ls = self._required_landmark(landmarks, PoseLandmark.LEFT_SHOULDER, warnings)
        rs = self._required_landmark(landmarks, PoseLandmark.RIGHT_SHOULDER, warnings)
        lh = self._required_landmark(landmarks, PoseLandmark.LEFT_HIP, warnings)
        rh = self._required_landmark(landmarks, PoseLandmark.RIGHT_HIP, warnings)
        if not all((ls, rs, lh, rh)):
            return self._empty_measurements(), {}, warnings, 0.0

        shoulder_mid = midpoint(ls, rs)
        hip_mid = midpoint(lh, rh)
        torso_length = get_distance(shoulder_mid, hip_mid)
        if torso_length <= 1e-6:
            warnings.append("Torso length collapsed to zero; returning empty measurements")
            return self._empty_measurements(), {}, warnings, 0.0

        landmark_shoulder = safe_ratio(get_distance(ls, rs), torso_length)
        landmark_hip = safe_ratio(get_distance(lh, rh), torso_length)
        widths = silhouette_profile.widths if silhouette_profile.valid else {}
        shoulder = float(widths.get("shoulder_width", landmark_shoulder))
        hip = float(widths.get("hip_width", landmark_hip))
        waist = float(widths.get("waist_width", 0.0))

        measurements = {
            "shoulder_width": round(shoulder, 4),
            "hip_width": round(hip, 4),
            "shoulder_hip_ratio": round(safe_ratio(shoulder, hip, default=1.0), 4),
        }
        if waist > 0.0:
            measurements["waist_width"] = round(waist, 4)
            measurements["waist_hip_ratio"] = round(safe_ratio(waist, hip, default=1.0), 4)

        return measurements, {"left_shoulder": ls, "right_shoulder": rs, "left_hip": lh, "right_hip": rh}, warnings, torso_length

    def _scale_to_cm(self, measurements: dict[str, float], landmarks: list[LandmarkPoint], torso_length_px: float, user_height_cm: float) -> dict[str, float]:
        """Convert normalised ratio measurements to cm using the user's declared height."""
        nose = landmarks[int(PoseLandmark.NOSE)]
        l_heel = landmarks[int(PoseLandmark.LEFT_HEEL)]
        r_heel = landmarks[int(PoseLandmark.RIGHT_HEEL)]
        scale: float | None = None
        heel_ok = (
            landmark_is_reliable(l_heel, visibility_threshold=self.visibility_threshold, presence_threshold=self.presence_threshold) or
            landmark_is_reliable(r_heel, visibility_threshold=self.visibility_threshold, presence_threshold=self.presence_threshold)
        )
        nose_ok = landmark_is_reliable(nose, visibility_threshold=self.visibility_threshold, presence_threshold=self.presence_threshold)
        if nose_ok and heel_ok:
            heels = [p for p in (l_heel, r_heel) if landmark_is_reliable(p, visibility_threshold=self.visibility_threshold, presence_threshold=self.presence_threshold)]
            feet_y = sum(p.y for p in heels) / len(heels)
            nose_to_heel_px = feet_y - nose.y
            if nose_to_heel_px > 10:
                scale = (user_height_cm * 0.88) / nose_to_heel_px
        if scale is None:
            scale = (user_height_cm * 0.32) / torso_length_px
        torso_cm = torso_length_px * scale
        result: dict[str, float] = {"torso_length_cm": round(torso_cm, 1)}
        for key in ("shoulder_width", "hip_width", "waist_width"):
            if key in measurements:
                result[key.replace("_width", "_width_cm")] = round(measurements[key] * torso_cm, 1)
        return result

    def _required_landmark(self, landmarks: list[LandmarkPoint], idx: PoseLandmark, warnings: list[str]) -> LandmarkPoint | None:
        point = landmarks[int(idx)]
        if landmark_is_reliable(point, visibility_threshold=self.visibility_threshold, presence_threshold=self.presence_threshold):
            return point
        warnings.append(f"Required landmark {idx.name.lower()} was not reliable")
        return None

    def _default_model_path(self) -> str:
        env_path = os.getenv("POSE_LANDMARKER_MODEL_PATH")
        if env_path:
            return env_path
        project_root = Path(__file__).resolve().parents[2]
        return str(project_root / DEFAULT_MODEL_RELATIVE_PATH)

    @staticmethod
    def _measurements_are_valid(measurements: dict[str, float], measurement_points: dict[str, LandmarkPoint]) -> bool:
        if not measurement_points:
            return False
        return all(float(measurements.get(key, 0.0)) > 0.0 for key in ("shoulder_width", "hip_width", "shoulder_hip_ratio"))

    @staticmethod
    def _empty_measurements() -> dict[str, float]:
        return {"shoulder_width": 0.0, "hip_width": 0.0, "shoulder_hip_ratio": 0.0}

    @staticmethod
    def _empty_result(image_bgr: np.ndarray | None, warning: str) -> PoseAnalysisResult:
        return PoseAnalysisResult(
            body_shape="unknown",
            measurements=PoseAnalyzer._empty_measurements(),
            confidence=0.0,
            pose_validation={"valid": False, "score": 0.0, "reasons": [warning]},
            landmarks_detected=0,
            silhouette={"valid": False, "widths": {}, "scanlines": []},
            warnings=[warning],
            annotated_image=image_bgr.copy() if image_bgr is not None else None,
        )
