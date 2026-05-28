from src.pose_analyzer.body_classifier import BodyShapeThresholds, classify_body_shape
from src.pose_analyzer.pose_analyzer import PoseAnalyzer
from src.pose_analyzer.silhouette import extract_silhouette_profile
from src.pose_analyzer.smoothing import MeasurementSmoother
from src.pose_analyzer.utils import LANDMARK_COUNT, LandmarkPoint, PoseLandmark
from src.pose_analyzer.validation import validate_pose
import numpy as np


def _blank_landmarks() -> list[LandmarkPoint]:
    return [LandmarkPoint(0.0, 0.0, visibility=1.0, presence=1.0) for _ in range(LANDMARK_COUNT)]


def test_measurements_are_normalized_and_ratios_are_consistent():
    analyzer = PoseAnalyzer(model_path="missing.task")
    landmarks = _blank_landmarks()

    landmarks[PoseLandmark.LEFT_SHOULDER] = LandmarkPoint(80, 100, visibility=0.99, presence=0.99)
    landmarks[PoseLandmark.RIGHT_SHOULDER] = LandmarkPoint(220, 100, visibility=0.99, presence=0.99)
    landmarks[PoseLandmark.LEFT_HIP] = LandmarkPoint(100, 220, visibility=0.99, presence=0.99)
    landmarks[PoseLandmark.RIGHT_HIP] = LandmarkPoint(210, 220, visibility=0.99, presence=0.99)
    landmarks[PoseLandmark.LEFT_ANKLE] = LandmarkPoint(110, 420, visibility=0.99, presence=0.99)
    landmarks[PoseLandmark.RIGHT_ANKLE] = LandmarkPoint(200, 420, visibility=0.99, presence=0.99)

    mask = np.zeros((480, 320), dtype=np.uint8)
    mask[95:160, 70:230] = 255
    mask[160:230, 95:205] = 255
    profile = extract_silhouette_profile(mask, landmarks)

    measurements, _, warnings = analyzer._compute_measurements(landmarks, profile)

    assert warnings == []
    assert profile.valid is True
    assert measurements["shoulder_width"] > measurements["hip_width"]
    assert abs(round(measurements["shoulder_width"] / measurements["hip_width"], 4) - measurements["shoulder_hip_ratio"]) < 0.001


def test_classifier_covers_expected_shapes():
    thresholds = BodyShapeThresholds()

    assert classify_body_shape(
        {"shoulder_width": 1.05, "hip_width": 1.02, "waist_width": 0.72, "torso_length": 1.0, "shoulder_hip_ratio": 1.03, "waist_hip_ratio": 0.71},
        thresholds,
    )[0] == "hourglass"
    assert classify_body_shape(
        {"shoulder_width": 1.08, "hip_width": 1.03, "waist_width": 0.93, "torso_length": 1.0, "shoulder_hip_ratio": 1.05, "waist_hip_ratio": 0.90},
        thresholds,
    )[0] == "rectangle"
    assert classify_body_shape(
        {"shoulder_width": 1.30, "hip_width": 1.00, "waist_width": 0.90, "torso_length": 1.0, "shoulder_hip_ratio": 1.30, "waist_hip_ratio": 0.90},
        thresholds,
    )[0] == "inverted_triangle"
    assert classify_body_shape(
        {"shoulder_width": 0.92, "hip_width": 1.18, "waist_width": 0.92, "torso_length": 1.0, "shoulder_hip_ratio": 0.78, "waist_hip_ratio": 0.78},
        thresholds,
    )[0] == "triangle"
    assert classify_body_shape(
        {"shoulder_width": 1.02, "hip_width": 1.00, "waist_width": 1.02, "torso_length": 1.0, "shoulder_hip_ratio": 1.02, "waist_hip_ratio": 1.02},
        thresholds,
    )[0] == "rectangle"


def test_classifier_rejects_empty_measurements():
    thresholds = BodyShapeThresholds()

    body_shape, confidence = classify_body_shape(
        {
            "shoulder_width": 0.0,
            "hip_width": 0.0,
            "shoulder_hip_ratio": 0.0,
        },
        thresholds,
    )

    assert body_shape == "unknown"
    assert confidence == 0.0


def test_invalid_landmarks_skip_shape_classification():
    analyzer = PoseAnalyzer(model_path="missing.task")
    landmarks = _blank_landmarks()

    landmarks[PoseLandmark.LEFT_SHOULDER] = LandmarkPoint(80, 100, visibility=0.99, presence=0.99)
    landmarks[PoseLandmark.RIGHT_SHOULDER] = LandmarkPoint(220, 100, visibility=0.99, presence=0.99)
    landmarks[PoseLandmark.LEFT_HIP] = LandmarkPoint(100, 220, visibility=0.10, presence=0.10)
    landmarks[PoseLandmark.RIGHT_HIP] = LandmarkPoint(210, 220, visibility=0.99, presence=0.99)
    landmarks[PoseLandmark.LEFT_ANKLE] = LandmarkPoint(110, 420, visibility=0.99, presence=0.99)
    landmarks[PoseLandmark.RIGHT_ANKLE] = LandmarkPoint(200, 420, visibility=0.99, presence=0.99)

    profile = extract_silhouette_profile(np.zeros((480, 320), dtype=np.uint8), landmarks)
    measurements, measurement_points, warnings = analyzer._compute_measurements(landmarks, profile)

    assert measurement_points == {}
    assert any("left_hip" in warning for warning in warnings)
    assert analyzer._measurements_are_valid(measurements, measurement_points) is False


def test_silhouette_profile_samples_widths():
    landmarks = _blank_landmarks()
    landmarks[PoseLandmark.LEFT_SHOULDER] = LandmarkPoint(50, 20, visibility=0.99, presence=0.99)
    landmarks[PoseLandmark.RIGHT_SHOULDER] = LandmarkPoint(110, 20, visibility=0.99, presence=0.99)
    landmarks[PoseLandmark.LEFT_HIP] = LandmarkPoint(60, 80, visibility=0.99, presence=0.99)
    landmarks[PoseLandmark.RIGHT_HIP] = LandmarkPoint(100, 80, visibility=0.99, presence=0.99)
    landmarks[PoseLandmark.LEFT_KNEE] = LandmarkPoint(65, 140, visibility=0.99, presence=0.99)
    landmarks[PoseLandmark.RIGHT_KNEE] = LandmarkPoint(95, 140, visibility=0.99, presence=0.99)
    mask = np.zeros((100, 160), dtype=np.uint8)
    mask[15:95, 40:121] = 255

    profile = extract_silhouette_profile(mask, landmarks)

    assert profile.valid is True
    assert profile.widths["shoulder_width"] > 1.0
    assert profile.widths["hip_width"] > 1.0


def test_measurement_smoother_rejects_spikes():
    smoother = MeasurementSmoother(window_size=3, spike_threshold=0.20)

    smoother.update({"shoulder_width": 1.0, "hip_width": 0.9, "shoulder_hip_ratio": 1.11})
    smoother.update({"shoulder_width": 1.02, "hip_width": 0.91, "shoulder_hip_ratio": 1.12})
    smoothed = smoother.update({"shoulder_width": 2.0, "hip_width": 0.2, "shoulder_hip_ratio": 10.0})

    assert smoothed["shoulder_width"] < 1.1
    assert smoothed["hip_width"] > 0.85
    assert smoothed["shoulder_hip_ratio"] < 1.2


def test_pose_validation_rejects_tilted_shoulders():
    landmarks = _blank_landmarks()
    landmarks[PoseLandmark.LEFT_SHOULDER] = LandmarkPoint(60, 40, visibility=0.99, presence=0.99)
    landmarks[PoseLandmark.RIGHT_SHOULDER] = LandmarkPoint(140, 85, visibility=0.99, presence=0.99)
    landmarks[PoseLandmark.LEFT_HIP] = LandmarkPoint(70, 150, visibility=0.99, presence=0.99)
    landmarks[PoseLandmark.RIGHT_HIP] = LandmarkPoint(130, 150, visibility=0.99, presence=0.99)
    mask = np.zeros((220, 200), dtype=np.uint8)
    mask[35:190, 50:150] = 255

    validation = validate_pose(landmarks, (220, 200, 3), mask, visibility_threshold=0.5)

    assert validation.valid is False
    assert validation.score < 0.65
    assert "shoulders tilted" in validation.reasons
