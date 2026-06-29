import json
import logging
import random
import ssl

import paho.mqtt.client as mqtt

log = logging.getLogger(__name__)

_BROKER_HOST = "test.mosquitto.org"
_BROKER_PORT = 8883


def publish_scan_result(response: dict, persona: str) -> None:
    """Publish scan result to cruzr/scan_result so the tablet can display it."""
    try:
        payload = json.dumps({
            "event": "scan_result",
            "session_id": response.get("session_id"),
            "persona": persona,
            "detections": response.get("detections", []),
            "recommendations": response.get("recommendations", []),
            "annotated_frame": response.get("annotated_frame"),
            "body_annotated_frame": response.get("body_annotated_frame"),
            "body_analysis": response.get("body_analysis"),
        })

        client_id = f"ViralyticsScan_{random.randint(1000, 9999)}"
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id)

        # test.mosquitto.org uses a custom CA — skip cert verification (same as CruzrBridge)
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        client.tls_set_context(ssl_ctx)

        client.connect(_BROKER_HOST, _BROKER_PORT, keepalive=10)
        result = client.publish("cruzr/scan_result", payload, qos=1)
        client.loop_start()
        result.wait_for_publish(timeout=5.0)
        client.disconnect()
        client.loop_stop()

        log.info("Published scan result to %s:%d cruzr/scan_result (%d bytes)",
                 _BROKER_HOST, _BROKER_PORT, len(payload))
    except Exception as e:
        log.warning("Could not publish scan result to MQTT: %s", e)
