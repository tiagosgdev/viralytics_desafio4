# Plan: make `arduino_door_sensor.ino` network-portable & speak-only

> Status: **proposed — pending colleague confirmation before implementation.**

## Decisions baked into this plan (confirm with colleagues)

- **ESP32 talks only to the PC API over HTTP.** It does *not* touch MQTT. The API
  relays speech to the robot over the public broker (`test.mosquitto.org`), so the
  **robot can be on any network** — this is the key fix for "not sure about the
  robot's network."
- **Robot just speaks**, one line per event: entering / leaving / running(alert).
  No navigation greet.
- **Use the existing `POST /api/robot/speak?text=...`** endpoint
  (`src/api/main.py:1386`). **No server or Android changes required.**

## Why the current firmware is broken (recap)

1. It publishes MQTT to `192.168.1.80:1883`, but the robot subscribes to
   `test.mosquitto.org:8883` (TLS) — so every MQTT message is dropped.
   (Robot: `android_app/.../MainActivity.kt:774`; PC bridge: `robotics/cruzr_bridge.py:45`.)
2. `API_BASE` (hotspot subnet) and `MQTT_HOST` (different LAN) are inconsistent.
3. It double-acts on entry (API greet *and* a direct speak).
4. `{"action":"sound"}` has no handler in the Android app (`MainActivity.kt:810`).

## Firmware changes

1. **WiFi without re-flashing → WiFiManager (`tzapu/WiFiManager`).**
   - Auto-connects to the last saved network; if absent, opens a `DoorSensorSetup`
     AP → join with phone → pick network. Saved to flash, survives power cycles.
     Works on powerbank.
   - To force the portal while the old network is still in range: hold BOOT (GPIO0)
     at power-up.
   - New library dependency to install in Arduino IDE.
2. **API URL also configurable in the same portal**, persisted via `Preferences`
   (NVS). Fixes the hotspot-IP-changes-every-session problem without re-flashing.
   (No new lib — `Preferences` is built into the ESP32 core.)
3. **Remove all MQTT** (`PubSubClient`, broker config, `ensureMqtt`,
   `publishRobotCommands`).
4. **Replace it with one HTTP call**: `sendSpeak(text)` →
   `POST {apiBase}/api/robot/speak?text=<url-encoded>`. Add a small `urlEncode()`
   for spaces/accents in the Portuguese text.
5. **`reportEvent`** picks the line — alert → `ALERT_TEXT`, else entry →
   `WELCOME_TEXT`, exit → `GOODBYE_TEXT` — logs the JSON to Serial (unchanged),
   then calls `sendSpeak`.
6. **Sensor logic (state machine, calibration, detection) stays exactly as-is.**

## Files touched

- Only `robotics/arduino_door_sensor/arduino_door_sensor.ino`. Nothing server-side.

## Open questions for colleagues

- **Who owns the message text?** This plan keeps the 3 strings in the firmware. If
  their design has the *server* deciding messages from an event type, instead POST
  an event (e.g. `{type, alerta}`) to one of their endpoints and let it pick the
  text — a one-function swap, isolated in `sendSpeak`/`reportEvent`.
- **Is `/api/robot/speak` the intended entry point**, or do they already have a
  dedicated door/sensor event route they want used? (The existing
  `/api/robot/sensor/door` does a *navigate-to-entrance greet*, not speak-only — so
  it doesn't fit "just speak.")
- **Does the PC API have reliable internet** at the venue (needed to reach
  `test.mosquitto.org`)?
