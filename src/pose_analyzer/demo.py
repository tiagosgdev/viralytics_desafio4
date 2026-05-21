"""Demo runner for pose-based body analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .pose_analyzer import PoseAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MediaPipe body-shape analysis on an image.")
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument("--model", type=Path, default=None, help="Path to the Pose Landmarker .task model")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save the annotated image")
    parser.add_argument("--no-window", action="store_true", help="Skip OpenCV display window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    analyzer = PoseAnalyzer(model_path=args.model)
    result = analyzer.analyze(image, draw_overlay=True, include_landmarks=False)
    print(json.dumps(result.to_dict(include_landmarks=False), indent=2))

    if result.annotated_image is not None and args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), result.annotated_image)

    if not args.no_window and result.annotated_image is not None:
        cv2.imshow("Pose Analysis", result.annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
