import json
import logging

log = logging.getLogger(__name__)


def publish_scan_result(response: dict, persona: str, broker_host: str = "localhost", broker_port: int = 1883) -> None:
    """Publish scan result to the local MQTT broker so the tablet can display it."""
    try:
        import paho.mqtt.publish as publish

        payload = json.dumps({
            "event": "scan_result",
            "session_id": response.get("session_id"),
            "persona": persona,
            "detections": response.get("detections", []),
            "recommendations": response.get("recommendations", []),
            "annotated_frame": response.get("annotated_frame"),
        })
        publish.single(
            topic="cruzr/scan_result",
            payload=payload,
            hostname=broker_host,
            port=broker_port,
            qos=1,
            keepalive=5,
        )
        log.debug("Published scan result to cruzr/scan_result (%d bytes)", len(payload))
    except Exception as e:
        log.warning("Could not publish scan result to MQTT: %s", e)
