#!/usr/bin/env python3
"""
Standalone test harness for the CruzrBridge.

Run this to send commands to the robot and watch status events in real time.
No AI agents, no FastAPI server — just the MQTT broker and the robot.

Usage:
    python robotics/test_harness.py
    python robotics/test_harness.py --host test.mosquitto.org
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cruzr_bridge import CruzrBridge  # noqa: E402

COORDS_FILE = Path(__file__).parent / "coordinates.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

MENU = """
┌──────────────────────────────────────────────────┐
│           CRUZR Bridge Test Harness              │
├──────────────────────────────────────────────────┤
│  1. speak                   (TTS only)           │
│  2. navigate_to             (named map marker)   │
│  3. guide_to                (speak + move)       │
│  4. stop_navigation                              │
│  5. locate_self             (re-localize)        │
│  6. request_status                               │
│  7. navigate_to_coords      (raw x, y, theta)    │
│  8. navigate_to_pixel       (tablet px, py)      │
│  9. go_to_saved_location    (from coordinates)   │
│  q. Quit                                         │
└──────────────────────────────────────────────────┘"""


def load_coords() -> dict:
    if COORDS_FILE.exists():
        with open(COORDS_FILE) as f:
            return json.load(f)
    return {}


def on_robot_status(event: dict):
    ev = event.get("event", "unknown").upper()
    print(f"\n  ← ROBOT [{ev}]  {event}")


def main():
    parser = argparse.ArgumentParser(description="CruzrBridge test harness")
    parser.add_argument("--host", default="test.mosquitto.org", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()

    bridge = CruzrBridge(broker_host=args.host, port=args.port)
    bridge.on_status(on_robot_status)

    print(f"\nConnecting to {args.host}:{args.port} …", end="", flush=True)
    if not bridge.connect():
        print(" FAILED.\nIs the MQTT broker reachable?")
        sys.exit(1)
    print(" OK\n")

    try:
        while True:
            print(MENU)
            choice = input("\nCommand > ").strip().lower()

            if choice == "q":
                break

            elif choice == "1":
                text = input("  Text: ").strip()
                if text:
                    bridge.speak(text)

            elif choice == "2":
                target = input("  Marker name: ").strip()
                if target:
                    bridge.navigate_to(target)

            elif choice == "3":
                target = input("  Marker name: ").strip()
                intro  = input("  Intro text:  ").strip()
                if target and intro:
                    bridge.guide_to(target, intro)

            elif choice == "4":
                bridge.stop_navigation()

            elif choice == "5":
                bridge.locate_self()

            elif choice == "6":
                bridge.request_status()

            elif choice == "7":
                try:
                    x     = float(input("  x (metres): ").strip())
                    y     = float(input("  y (metres): ").strip())
                    theta = float(input("  theta (radians, default 0): ").strip() or "0")
                    bridge.navigate_to_coords(x, y, theta)
                except ValueError:
                    print("  Invalid number.")

            elif choice == "8":
                try:
                    px    = float(input("  pixel x: ").strip())
                    py    = float(input("  pixel y: ").strip())
                    theta = float(input("  theta (radians, default 0): ").strip() or "0")
                    mx, my = CruzrBridge._pixel_to_metric(px, py)
                    print(f"  → converted to metric ({mx:.3f}, {my:.3f})")
                    bridge.navigate_to_pixel(px, py, theta)
                except ValueError:
                    print("  Invalid number.")

            elif choice == "9":
                coords = load_coords()
                if not coords:
                    print("  No saved locations. Run survey_tool.py first.")
                else:
                    print(f"  Available: {', '.join(coords)}")
                    name = input("  Location name: ").strip()
                    if name in coords:
                        loc = coords[name]
                        print(f"  Sending to '{name}' ({loc['x']:.3f}, {loc['y']:.3f})...")
                        bridge.navigate_to_coords(loc["x"], loc["y"], loc.get("theta", 0.0))
                    else:
                        print(f"  '{name}' not found.")

            else:
                print("  Unknown command.")
                continue

            time.sleep(1.0)

    except KeyboardInterrupt:
        pass
    finally:
        bridge.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
