#!/usr/bin/env python3
"""
Teach-and-Repeat survey tool for the Cruzr robot.

Workflow:
  1. Manually drive the robot to the target physical location.
  2. Run this script and choose option 1 (Capture).
  3. The script sends get_status over MQTT; the robot replies with its exact
     AMCL metric coordinates, which are saved to coordinates.json.
  4. Repeat for every location you want to store.
  5. Use option 4 to verify: the robot should drive back to any saved spot.

Usage:
    python robotics/survey_tool.py
    python robotics/survey_tool.py --host test.mosquitto.org
"""

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cruzr_bridge import CruzrBridge  # noqa: E402

COORDS_FILE = Path(__file__).parent / "coordinates.json"

logging.basicConfig(level=logging.WARNING)  # suppress bridge noise during interactive use

MENU = """
┌─────────────────────────────────────────┐
│        CRUZR Survey Tool                │
├─────────────────────────────────────────┤
│  1. Capture current location            │
│  2. List saved locations                │
│  3. Delete a location                   │
│  4. Navigate to saved location (verify) │
│  q. Quit                                │
└─────────────────────────────────────────┘"""


def load_coords() -> dict:
    if COORDS_FILE.exists():
        with open(COORDS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_coords(coords: dict) -> None:
    with open(COORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(coords, f, indent=2)


def capture_location(bridge: CruzrBridge, status_holder: dict, status_event: threading.Event) -> dict | None:
    """Ask the robot for its current position and return the coordinates."""
    status_event.clear()
    bridge.request_status()

    print("  Waiting for robot position...", end="", flush=True)
    if not status_event.wait(timeout=10.0):
        print(" TIMEOUT — is the robot online and subscribed?")
        return None

    data = status_holder.copy()
    if "x" not in data:
        print(f" Response received but no location data: {data}")
        return None

    print(f" x={data['x']:.3f}  y={data['y']:.3f}  theta={data.get('theta', 0.0):.4f}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Cruzr survey tool")
    parser.add_argument("--host", default="test.mosquitto.org", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=8883)
    parser.add_argument("--no-tls", action="store_true", help="Disable TLS (for local brokers on port 1883)")
    args = parser.parse_args()

    bridge = CruzrBridge(broker_host=args.host, port=args.port, use_tls=not args.no_tls)

    # Single persistent status listener — avoids accumulating callbacks on repeated captures.
    status_holder: dict = {}
    status_event = threading.Event()

    def on_status(evt: dict) -> None:
        if evt.get("event") == "status_report":
            status_holder.clear()
            status_holder.update(evt)
            status_event.set()

    bridge.on_status(on_status)

    print(f"\nConnecting to {args.host}:{args.port}...", end="", flush=True)
    if not bridge.connect():
        print(" FAILED.\nCheck that the MQTT broker is reachable.")
        sys.exit(1)
    print(" OK\n")

    try:
        while True:
            print(MENU)
            choice = input("\nCommand > ").strip().lower()

            if choice == "q":
                break

            elif choice == "1":
                print("\n  Drive the robot to the target location first.")
                input("  Press ENTER when the robot is in position...")
                data = capture_location(bridge, status_holder, status_event)
                if data is None:
                    continue

                name = input("  Location name (e.g. 't-shirt stand'): ").strip()
                if not name:
                    print("  Aborted — no name given.")
                    continue

                coords = load_coords()
                coords[name] = {
                    "x":     round(data["x"], 4),
                    "y":     round(data["y"], 4),
                    "theta": round(data.get("theta", 0.0), 6),
                }
                save_coords(coords)
                print(f"  Saved '{name}' → {coords[name]}")

            elif choice == "2":
                coords = load_coords()
                if not coords:
                    print("  No locations saved yet.")
                else:
                    print(f"\n  {'NAME':<28} {'X':>12} {'Y':>12} {'THETA':>10}")
                    print("  " + "─" * 64)
                    for name, loc in coords.items():
                        note = f"  ({loc['note']})" if "note" in loc else ""
                        print(f"  {name:<28} {loc['x']:>12.3f} {loc['y']:>12.3f} {loc.get('theta', 0):>10.4f}{note}")

            elif choice == "3":
                coords = load_coords()
                if not coords:
                    print("  No locations to delete.")
                    continue
                name = input("  Location name to delete: ").strip()
                if name in coords:
                    del coords[name]
                    save_coords(coords)
                    print(f"  Deleted '{name}'.")
                else:
                    print(f"  '{name}' not found. Available: {', '.join(coords)}")

            elif choice == "4":
                coords = load_coords()
                if not coords:
                    print("  No locations saved yet.")
                    continue
                print(f"  Available: {', '.join(coords)}")
                name = input("  Location name: ").strip()
                if name not in coords:
                    print(f"  '{name}' not found.")
                    continue
                loc = coords[name]
                print(f"  Sending robot to '{name}' ({loc['x']:.3f}, {loc['y']:.3f})...")
                bridge.navigate_to_coords(loc["x"], loc["y"], loc.get("theta", 0.0))
                time.sleep(1.0)

            else:
                print("  Unknown command.")

    except KeyboardInterrupt:
        pass
    finally:
        bridge.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
