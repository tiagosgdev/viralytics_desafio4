"""
CruzrBridge — bidirectional MQTT link between the PC server / AI agents and the CRUZR robot.

MQTT topics
───────────
  cruzr/commands  PC → Robot   All commands: greet, guide_user, move_to_stand,
                               move_to_coords, speak, gesture, farewell, locate_self,
                               get_status, stop_navigation.
                               Payload: JSON object with at minimum {"action": "<name>"}.

  cruzr/status    Robot → PC   Navigation events, errors, position reports.
                               Payload: JSON object with at minimum {"event": "<name>"}.
                               Known events: greeting_started, greeting_at_entrance,
                               person_detected, navigation_arrived, navigation_failed,
                               session_ended, lidar_timeout, status_report, error.

  cruzr/persona   PC → Robot   Active persona configuration (QoS 1).
  cruzr/scan_result PC → Robot  Scan results to display on the tablet (QoS 1).

Broker
──────
  Default: test.mosquitto.org:8883 (TLS, public, no authentication).
  TLS is used for wire encryption but certificate verification is disabled because
  test.mosquitto.org uses a custom CA that predates modern Python/Android requirements.
  This is acceptable for a university demo. For production, use a private broker
  (e.g. self-hosted Mosquitto, HiveMQ Cloud, AWS IoT Core) with proper cert verification
  and username/password or client-certificate authentication.

Usage (standalone):
    bridge = CruzrBridge()
    bridge.on_status(lambda evt: print(evt))
    bridge.connect()
    bridge.speak("Bem-vindo!")
    bridge.navigate_to("T-shirt Stand")
    bridge.disconnect()

Usage (context manager):
    with CruzrBridge() as bridge:
        bridge.guide_to("T-shirt Stand", "Siga-me!")

AI agent integration:
    Instantiate once, share the reference, call command methods as needed.
    Register on_status() callbacks to react to navigation results.
"""

import json
import logging
import math
import random
import ssl
import threading
import time
from typing import Callable

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class CruzrBridge:
    COMMANDS_TOPIC = "cruzr/commands"
    STATUS_TOPIC   = "cruzr/status"
    _COORD_LIMIT   = 10_000.0  # metres — beyond this a coordinate is almost certainly a bug

    def __init__(
        self,
        broker_host: str = "test.mosquitto.org",
        port: int = 8883,
        client_id: str | None = None,
        keepalive: int = 60,
        use_tls: bool = True,
    ):
        self.broker_host = broker_host
        self.port = port
        self.keepalive = keepalive
        self.use_tls = use_tls

        # Randomize suffix to avoid broker ejecting a previous stale connection
        # with the same ID (a common issue on shared brokers like test.mosquitto.org).
        if client_id is None:
            client_id = f"ViralyticsBridge_{random.randint(1000, 9999)}"
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id)
        self._connected = False
        self._connect_event = threading.Event()
        # Paho's network thread calls _on_message; main thread calls on_status().
        # Both touch _status_callbacks, so a lock is required (T5 slide 29).
        self._callbacks_lock = threading.Lock()
        self._status_callbacks: list[Callable[[dict], None]] = []

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message
        # Back off 1–30 s between reconnect attempts; paho handles this automatically.
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)

    # ------------------------------------------------------------------ #
    #  Connection                                                           #
    # ------------------------------------------------------------------ #

    def connect(self, timeout: float = 10.0) -> bool:
        """Connect and start the background network loop. Returns True on success."""
        try:
            if self.use_tls:
                # test.mosquitto.org uses a custom CA whose cert predates the keyUsage
                # extension that Python 3.10+ now requires, so standard verification fails.
                # Since this is a public test broker (no auth, anyone can subscribe), we
                # still use TLS for wire encryption but skip cert verification.
                # Switch to CERT_REQUIRED + a private broker before going to production.
                ssl_ctx = ssl.create_default_context()
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = ssl.CERT_NONE
                self._client.tls_set_context(ssl_ctx)
            # LWT: broker auto-publishes this if our TCP connection drops unexpectedly.
            # Subscribers (e.g. a monitoring dashboard) learn the PC is offline without polling.
            lwt_payload = json.dumps({"event": "pc_offline"})
            self._client.will_set(self.STATUS_TOPIC, lwt_payload, qos=1, retain=True)
            self._client.connect(self.broker_host, self.port, self.keepalive)
            self._client.loop_start()
            if not self._connect_event.wait(timeout):
                logger.error("[CruzrBridge] connect() timed out after %.1fs", timeout)
                return False
            return self._connected
        except Exception as exc:
            logger.error("[CruzrBridge] connect() error: %s", exc)
            return False

    def disconnect(self):
        self._client.loop_stop()
        self._client.disconnect()
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ------------------------------------------------------------------ #
    #  Commands  (PC → Robot, topic: cruzr/commands)                       #
    # ------------------------------------------------------------------ #

    def speak(self, text: str) -> bool:
        """Have the robot say something via its TTS engine."""
        return self._publish({"action": "speak", "text": text})

    def navigate_to(self, target: str) -> bool:
        """Send the robot to a named map marker."""
        return self._publish({"action": "move_to_stand", "target": target})

    def guide_to(self, target: str, intro_text: str) -> bool:
        """
        Speak an intro then navigate — robot handles sequencing so the PC
        doesn't need to sleep between the two steps.
        """
        return self._publish({
            "action": "guide_user",
            "target": target,
            "intro_text": intro_text,
        })

    def navigate_to_coords(self, x: float, y: float, theta: float = 0.0) -> bool:
        """Send the robot to an exact ROS AMCL metric position (metres)."""
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(theta)):
            logger.error(
                "[CruzrBridge] navigate_to_coords: non-finite coordinate rejected "
                "(x=%s y=%s theta=%s)", x, y, theta,
            )
            return False
        if abs(x) > self._COORD_LIMIT or abs(y) > self._COORD_LIMIT:
            logger.error(
                "[CruzrBridge] navigate_to_coords: coordinate exceeds safety limit "
                "(x=%.1f y=%.1f limit=%.0f)", x, y, self._COORD_LIMIT,
            )
            return False
        theta = max(-math.pi, min(math.pi, theta))  # clamp; don't reject — callers may overshoot slightly
        return self._publish({"action": "move_to_coords", "x": x, "y": y, "theta": theta})

    def navigate_to_pixel(self, px: float, py: float, theta: float = 0.0) -> bool:
        """Convert tablet pixel coordinates to metric via affine calibration and navigate."""
        x, y = self._pixel_to_metric(px, py)
        logger.info("[CruzrBridge] pixel (%.1f, %.1f) → metric (%.3f, %.3f)", px, py, x, y)
        return self.navigate_to_coords(x, y, theta)

    def greet(self, x: float, y: float, theta: float = 0.0,
              text: str = "Welcome! I'm Cruzr, your personal fashion assistant.",
              gesture: str = "wave") -> bool:
        """
        Compound greeting sequence: navigate to entrance position, then on arrival
        speak welcome text and perform a gesture.  The Android handles the sequencing
        so the PC doesn't need to wait or chain commands manually.
        """
        if not (math.isfinite(x) and math.isfinite(y)):
            logger.error("[CruzrBridge] greet: non-finite coordinates (x=%s y=%s)", x, y)
            return False
        return self._publish({"action": "greet", "x": x, "y": y, "theta": theta,
                               "text": text, "gesture": gesture})

    def gesture(self, name: str) -> bool:
        """Play a named body gesture (wave, bow, nod, …) without navigation."""
        return self._publish({"action": "gesture", "name": name})

    def farewell(self, text: str = "Até logo! Esperamos vê-lo em breve!", gesture: str = "goodbye") -> bool:
        """
        Say goodbye to a departing customer.  The Android only acts on this if the
        robot is idle at the entrance (not navigating and not serving another customer).
        """
        return self._publish({"action": "farewell", "text": text, "gesture": gesture})

    def stop_navigation(self) -> bool:
        """Request the robot to abort its current navigation."""
        return self._publish({"action": "stop_navigation"})

    def locate_self(self) -> bool:
        """Ask the robot to run its self-localization routine."""
        return self._publish({"action": "locate_self"})

    def request_status(self) -> bool:
        """Ask the robot to publish its current state to cruzr/status."""
        return self._publish({"action": "get_status"})

    # ------------------------------------------------------------------ #
    #  Status events  (Robot → PC, topic: cruzr/status)                   #
    # ------------------------------------------------------------------ #

    def on_status(self, callback: Callable[[dict], None]) -> "CruzrBridge":
        """
        Register a listener for robot status events.

        The callback receives a dict with at least an ``"event"`` key.
        Known events:
          navigation_started, navigation_progress, navigation_arrived,
          navigation_failed, marker_not_found, speech_started,
          localization_started, localization_success, localization_failed,
          status_report, error

        Multiple callbacks can be registered; all are called in order.
        Returns self to allow chaining.
        """
        with self._callbacks_lock:
            self._status_callbacks.append(callback)
        return self

    # ------------------------------------------------------------------ #
    #  Internal paho callbacks                                             #
    # ------------------------------------------------------------------ #

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self._connected = True
            client.subscribe(self.STATUS_TOPIC, qos=1)
            logger.info(
                "[CruzrBridge] Connected to %s:%d | listening on %s",
                self.broker_host, self.port, self.STATUS_TOPIC,
            )
        else:
            logger.error("[CruzrBridge] Connection refused: reason_code=%s", reason_code)
        self._connect_event.set()

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        self._connected = False
        if reason_code != 0:
            logger.warning(
                "[CruzrBridge] Unexpected disconnect (rc=%s). Paho will auto-reconnect.",
                reason_code,
            )

    def _on_message(self, client, userdata, message):
        try:
            payload = json.loads(message.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("[CruzrBridge] Non-JSON status message: %r", message.payload)
            return

        logger.info("[CruzrBridge] ← robot: %s", payload)
        with self._callbacks_lock:
            callbacks = list(self._status_callbacks)
        for cb in callbacks:
            try:
                cb(payload)
            except Exception as exc:
                logger.error("[CruzrBridge] Status callback raised: %s", exc)

    @staticmethod
    def _pixel_to_metric(px: float, py: float) -> tuple[float, float]:
        """
        Affine pixel-to-metric calibration (3 control points, 2026-06-20).
        See calibration.json for control points; coord_converter.py to recalibrate.
        """
        metric_x = 5.1244 * px - 0.0895 * py - 638.7161
        metric_y = 0.0930 * px - 5.1845 * py + 1618.2753
        return float(metric_x), float(metric_y)

    def _publish(self, payload: dict) -> bool:
        if not self._connected:
            logger.error("[CruzrBridge] Cannot publish — not connected.")
            return False
        try:
            result = self._client.publish(
                self.COMMANDS_TOPIC, json.dumps(payload), qos=1
            )
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.error("[CruzrBridge] publish() rc=%d", result.rc)
                return False
            logger.info("[CruzrBridge] → %s", payload)
            return True
        except Exception as exc:
            logger.error("[CruzrBridge] publish() raised: %s", exc)
            return False
