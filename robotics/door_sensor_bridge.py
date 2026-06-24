#!/usr/bin/env python3
"""Bridge Serial events from the Arduino door sensors to REST and Cruzr MQTT."""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import requests

try:
    import serial
except ImportError:  # pragma: no cover - friendly runtime error
    serial = None

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover - MQTT can be disabled
    mqtt = None


DEFAULT_WELCOME = "Bem-vindo. Espero que tenha uma excelente visita."
DEFAULT_GOODBYE = "Obrigado pela visita. Ate breve."
DEFAULT_ALERT = "Alerta. Movimento suspeito detectado."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read Arduino JSON door events and forward them to the API and robot."
    )
    parser.add_argument("--serial-port", default=os.getenv("DOOR_SERIAL_PORT", "/dev/ttyACM0"))
    parser.add_argument("--baud", type=int, default=int(os.getenv("DOOR_SERIAL_BAUD", "9600")))
    parser.add_argument("--api-url", default=os.getenv("DOOR_API_URL", "http://localhost:8000/api/door-event"))
    parser.add_argument("--token", default=os.getenv("DOOR_API_TOKEN", ""))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("DOOR_API_TIMEOUT", "3")))
    parser.add_argument("--mqtt-host", default=os.getenv("CRUZR_MQTT_HOST", "test.mosquitto.org"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("CRUZR_MQTT_PORT", "1883")))
    parser.add_argument("--mqtt-topic", default=os.getenv("CRUZR_MQTT_TOPIC", "cruzr/commands"))
    parser.add_argument("--no-mqtt", action="store_true", help="Only send HTTP posts; do not command the robot.")
    parser.add_argument("--welcome-text", default=os.getenv("DOOR_WELCOME_TEXT", DEFAULT_WELCOME))
    parser.add_argument("--goodbye-text", default=os.getenv("DOOR_GOODBYE_TEXT", DEFAULT_GOODBYE))
    parser.add_argument("--alert-text", default=os.getenv("DOOR_ALERT_TEXT", DEFAULT_ALERT))
    return parser.parse_args()


def build_headers(token: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def read_event(raw_line: bytes) -> Optional[Dict[str, Any]]:
    line = raw_line.decode("utf-8", errors="replace").strip()
    if not line:
        return None

    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        print(f"Ignored non-JSON serial line: {line}", file=sys.stderr)
        return None

    if "debug" in payload or "status" in payload or "tipo" not in payload:
        return None

    if payload["tipo"] not in {"entrada", "saida"}:
        print(f"Ignored event with invalid tipo: {payload}", file=sys.stderr)
        return None

    payload["robot_fala"] = DEFAULT_WELCOME if payload["tipo"] == "entrada" else DEFAULT_GOODBYE
    if payload.get("alerta") is True:
        payload["robot_alerta_som"] = True
    return payload


def post_event(api_url: str, headers: Dict[str, str], timeout: float, event: Dict[str, Any]) -> None:
    response = requests.post(api_url, json=event, headers=headers, timeout=timeout)
    response.raise_for_status()


class RobotPublisher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.enabled = not args.no_mqtt
        self.topic = args.mqtt_topic
        self.welcome_text = args.welcome_text
        self.goodbye_text = args.goodbye_text
        self.alert_text = args.alert_text
        self.client = None

        if not self.enabled:
            return
        if mqtt is None:
            raise RuntimeError("paho-mqtt is not installed. Install it or run with --no-mqtt.")

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "door_sensor_bridge")
        self.client.connect(args.mqtt_host, args.mqtt_port)
        self.client.loop_start()

    def close(self) -> None:
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()

    def publish_json(self, payload: Dict[str, Any]) -> None:
        if self.client is None:
            return
        self.client.publish(self.topic, json.dumps(payload))

    def handle_event(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        if event.get("alerta") is True:
            self.publish_json({"action": "sound", "name": "alert"})
            self.publish_json({"action": "speak", "text": self.alert_text})
            return

        if event["tipo"] == "entrada":
            self.publish_json({"action": "speak", "text": self.welcome_text})
        elif event["tipo"] == "saida":
            self.publish_json({"action": "speak", "text": self.goodbye_text})


def main() -> int:
    args = parse_args()
    if serial is None:
        print("pyserial is not installed. Install with: pip install pyserial", file=sys.stderr)
        return 2

    headers = build_headers(args.token)
    robot = RobotPublisher(args)

    try:
        with serial.Serial(args.serial_port, args.baud, timeout=1) as ser:
            print(f"Listening on {args.serial_port} at {args.baud} baud.")
            time.sleep(2)

            while True:
                event = read_event(ser.readline())
                if event is None:
                    continue

                event["robot_fala"] = args.welcome_text if event["tipo"] == "entrada" else args.goodbye_text

                try:
                    post_event(args.api_url, headers, args.timeout, event)
                    print(f"Posted event: {event}")
                except requests.RequestException as exc:
                    print(f"Failed to post event: {exc}", file=sys.stderr)

                try:
                    robot.handle_event(event)
                except Exception as exc:
                    print(f"Failed to command robot: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
