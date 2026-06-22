#!/usr/bin/env python3
"""
Affine 2D pixel-to-metric converter for the Cruzr tablet map.

The robot's ROS/AMCL stack operates in absolute metric space (metres).
The tablet UI stores markers in pixel/grid space relative to the map image.
This module bridges the two using a calibrated affine transform.

Transform:   metric = M · [pixel_x, pixel_y, 1]ᵀ
Calibration: 3 control points surveyed manually (see calibration.json).

Standalone usage (prints converted coordinates):
    python robotics/coord_converter.py 120 150
    python robotics/coord_converter.py --recalibrate   # recompute from calibration.json
"""

import json
import sys
from pathlib import Path

_CAL_FILE = Path(__file__).parent / "calibration.json"

# Pre-computed from 3 control points (2026-06-20).
# Recompute with --recalibrate if control points change.
_COEFF_X = (5.1244, -0.0895, -638.7161)   # (a, b, c) for metric_x = a*px + b*py + c
_COEFF_Y = (0.0930, -5.1845, 1618.2753)   # (d, e, f) for metric_y = d*px + e*py + f


def pixel_to_metric(px: float, py: float) -> tuple[float, float]:
    """
    Convert tablet map pixel coordinates to ROS AMCL metric coordinates.

    Args:
        px: pixel X from the tablet map image
        py: pixel Y from the tablet map image

    Returns:
        (metric_x, metric_y) in the ROS coordinate frame (metres)
    """
    ax, bx, cx = _COEFF_X
    ay, by, cy = _COEFF_Y
    return float(ax * px + bx * py + cx), float(ay * px + by * py + cy)


def _recalibrate() -> None:
    """Recompute coefficients from calibration.json and print them."""
    import numpy as np

    if not _CAL_FILE.exists():
        print(f"calibration.json not found at {_CAL_FILE}")
        sys.exit(1)

    with open(_CAL_FILE) as f:
        cal = json.load(f)

    pts = cal["control_points"]
    if len(pts) < 3:
        print("Need at least 3 control points.")
        sys.exit(1)

    tablet = np.array([[p["pixel"][0], p["pixel"][1], 1.0] for p in pts])
    mx     = np.array([p["metric"][0] for p in pts])
    my     = np.array([p["metric"][1] for p in pts])

    cx = np.linalg.solve(tablet, mx)
    cy = np.linalg.solve(tablet, my)

    print("=== RECOMPUTED COEFFICIENTS ===")
    print(f"metric_x = ({cx[0]:.4f} * px) + ({cx[1]:.4f} * py) + {cx[2]:.4f}")
    print(f"metric_y = ({cy[0]:.4f} * px) + ({cy[1]:.4f} * py) + {cy[2]:.4f}")
    print()
    print("Residuals (should be ~0 for all control points):")
    for i, p in enumerate(pts):
        ex, ey = pixel_to_metric(p["pixel"][0], p["pixel"][1])
        print(f"  Point {i+1}: got ({ex:.3f}, {ey:.3f})  expected {p['metric']}  "
              f"Δ=({ex - p['metric'][0]:.4f}, {ey - p['metric'][1]:.4f})")


if __name__ == "__main__":
    if "--recalibrate" in sys.argv:
        _recalibrate()
        sys.exit(0)

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) != 2:
        print("Usage: python coord_converter.py <pixel_x> <pixel_y>")
        print("       python coord_converter.py --recalibrate")
        sys.exit(1)

    px, py = float(args[0]), float(args[1])
    mx, my = pixel_to_metric(px, py)
    print(f"pixel ({px}, {py})  →  metric ({mx:.4f}, {my:.4f})")
