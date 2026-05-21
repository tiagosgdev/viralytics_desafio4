from src.pose_analyzer.body_classifier import BodyShapeThresholds, classify_body_shape
from src.pose_analyzer.pose_analyzer import PoseAnalyzer
from src.pose_analyzer.utils import LANDMARK_COUNT, LandmarkPoint, PoseLandmark


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

    measurements, _, warnings = analyzer._compute_measurements(landmarks)

    assert warnings == []
    assert measurements["torso_length"] == 1.0
    assert measurements["shoulder_width"] > measurements["hip_width"]
    assert 0.0 < measurements["waist_width"] < measurements["hip_width"]
    assert measurements["leg_length"] > 1.0
    assert round(measurements["shoulder_width"] / measurements["hip_width"], 4) == measurements["shoulder_hip_ratio"]


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
    )[0] == "oval"


def test_classifier_rejects_empty_measurements():
    thresholds = BodyShapeThresholds()

    body_shape, confidence = classify_body_shape(
        {
            "shoulder_width": 0.0,
            "hip_width": 0.0,
            "waist_width": 0.0,
            "torso_length": 0.0,
            "leg_length": 0.0,
            "shoulder_hip_ratio": 0.0,
            "waist_hip_ratio": 0.0,
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

    measurements, measurement_points, warnings = analyzer._compute_measurements(landmarks)

    assert measurement_points == {}
    assert any("left_hip" in warning for warning in warnings)
    assert analyzer._measurements_are_valid(measurements, measurement_points) is False
