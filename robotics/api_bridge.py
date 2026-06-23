#!/usr/bin/env python3
"""
One-shot navigation trigger — sends the robot to a saved location.

Usage:
    python robotics/api_bridge.py                    # lists available locations
    python robotics/api_bridge.py test_spot          # navigate to 'test_spot'
    python robotics/api_bridge.py "t-shirt stand"    # navigate to 't-shirt stand'

The robot connects to this script via MQTT on test.mosquitto.org.
Coordinates are loaded from coordinates.json (populated by survey_tool.py).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cruzr_bridge import CruzrBridge

COORDS_FILE = Path(__file__).parent / "coordinates.json"


def load_coords() -> dict:
    if not COORDS_FILE.exists():
        return {}
    with open(COORDS_FILE) as f:
        return json.load(f)


def main():
    coords = load_coords()

    if not coords:
        print("No saved locations. Run survey_tool.py to survey the space first.")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Saved locations:")
        for name, loc in coords.items():
            note = f"  ({loc['note']})" if "note" in loc else ""
            print(f"  {name:<28} x={loc['x']:.3f}  y={loc['y']:.3f}  theta={loc.get('theta', 0):.4f}{note}")
        print(f"\nUsage: python robotics/api_bridge.py <location_name>")
        sys.exit(0)

    name = " ".join(sys.argv[1:])
    if name not in coords:
        print(f"Location '{name}' not found.")
        print(f"Available: {', '.join(coords)}")
        sys.exit(1)

    loc = coords[name]
    x, y, theta = loc["x"], loc["y"], loc.get("theta", 0.0)

    print(f"Connecting to test.mosquitto.org...")
    bridge = CruzrBridge()
    if not bridge.connect():
        print("Connection failed.")
        sys.exit(1)

    print(f"Sending robot to '{name}': x={x}, y={y}, theta={theta}")
    bridge.navigate_to_coords(x, y, theta)
    bridge.disconnect()
    print("Command sent.")


if __name__ == "__main__":
    main()
