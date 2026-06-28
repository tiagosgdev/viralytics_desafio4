# Entrance Sensor — Architecture

How the store-entrance presence sensor detects a customer and reaches the backend.

> Source of truth: firmware `robotics/arduino_door_sensor/arduino_door_sensor.ino`, backend endpoint
> `src/api/main.py:1383`, and `robotics/arduino_door_sensor/PLAN.md` (implemented & verified end-to-end 2026-06-26).
> The reserved public URL is shown as `<reserved-ngrok-domain>`; the real value is hard-coded as `API_HOST` in the
> firmware.

---

## 1. Overview

A self-contained **ESP32-S3** node at the doorway detects a person crossing, decides the **direction** (entering vs
leaving), and notifies the backend with a single HTTPS request. The sensor carries **no application logic**: it only
reports the crossing. The **server** decides what happens (on *entering* the robot navigates to the entrance and greets;
*leaving* is ignored). The node is battery-powered (a USB power bank) and needs no wired connection to any PC.

```
  [Customer crosses doorway]
            │  ultrasonic A then B (or B then A)
            ▼
  ┌───────────────────┐   HTTPS POST (Wi-Fi → internet)
  │  ESP32-S3 sensor   │ ───────────────────────────────►  ngrok tunnel  ──►  FastAPI backend
  │  2× HC-SR04        │   /api/robot/sensor/door             (fixed URL)       (decides robot action)
  └───────────────────┘   ?direction=entering|leaving                          │ MQTT cruzr/commands
                                                                                ▼
                                                                           CRUZR robot (navigate + greet)
```

## 2. Hardware & detection

- **MCU:** ESP32-S3, powered from a USB power bank (fully wireless install).
- **Sensors:** two **HC-SR04 ultrasonic** range finders, `A` (TRIG 6 / ECHO 7) and `B` (TRIG 4 / ECHO 5), mounted a
  fixed `SENSOR_DISTANCE_CM = 15 cm` apart along the direction of travel (`arduino_door_sensor.ino:40-55`).
- **Calibration:** at boot each sensor measures the empty-doorway floor distance (20-sample average,
  `readStableDistanceCm`). A person is detected when the live distance drops more than `PERSON_THRESHOLD_CM` below that
  floor baseline — i.e. something tall enough is under the sensor (`personDetected`, `:241`).
- **Direction (state machine, `loop()` `:276`):** which beam breaks **first** gives the direction —
  `A→B = "entrada"` (entering), `B→A = "saida"` (leaving). The node also derives the person's **height** and **crossing
  speed** (`SENSOR_DISTANCE_CM / Δt`) and flags a fast crossing (`> 150 cm/s`); these are logged for diagnostics but no
  longer suppress an event. A `WAIT_CLEAR` state debounces until both beams read clear before re-arming.

## 3. Wi-Fi connectivity  ⭐

This is the core of the design: the node must work in **any venue** on **whatever Wi-Fi is available**, with **no
re-flashing** and **no PC tether**.

### 3.1 Provisioning — captive portal, no hard-coded credentials
Wi-Fi credentials are **never compiled in**. The firmware uses the **WiFiManager** library (`tzapu/WiFiManager`):

- On boot, `wm.autoConnect("DoorSensorSetup")` tries the **last saved network** from flash
  (`setupWifi`, `:108-121`).
- If nothing is saved — or the **GPIO0/BOOT** pin is held LOW at power-up (`forcePortal`) — the node raises its own
  **captive-portal access point** `DoorSensorSetup`. An installer joins it with a phone and picks the venue Wi-Fi from a
  web form.
- The chosen SSID/password are **persisted to flash** and reused on every power cycle, so changing networks is a phone
  task, never a re-flash. The portal self-closes after `PORTAL_TIMEOUT_S = 180 s` so the node never sits as an open AP
  in the field.

### 3.2 Keeping the link reliable
- **Modem sleep is disabled** (`WiFi.setSleep(false)`, `:127`). By default the ESP32 powers its radio down between
  beacons; that intermittently breaks the **multi-round-trip TLS handshake** and is the classic *"the POST only works
  sometimes"* bug. Holding the radio on fixes it.
- **Auto-reconnect** — every loop, `ensureWifi()` (`:140`) checks the link and, if dropped (e.g. a power-bank blip),
  calls `WiFi.reconnect()` (which reuses the flash credentials) and waits up to 10 s.

### 3.3 Why a fixed public URL (ngrok) instead of a LAN IP
The backend (FastAPI) runs on a laptop that is itself a **Wi-Fi client**: its IP is DHCP-assigned and changes, and a
single Wi-Fi radio can't be both a client *and* a fixed-IP hotspot. Hard-coded LAN IPs, laptop-as-AP, and mDNS were all
rejected (unstable / machine-dependent / extra hardware).

Instead, the laptop exposes the API through an **ngrok tunnel** with a **reserved static domain**
(`<reserved-ngrok-domain>`, free-tier, permanent). This gives:

- A **fixed URL independent of laptop, IP and network.** The sensor and the laptop each only need *internet* — they need
  **not** be on the same Wi-Fi.
- Whichever machine runs `ngrok http --url=https://<reserved-ngrok-domain> 8000` (with the account authtoken) is
  reachable at the same address, so the demo is portable.

### 3.4 The request (HTTPS over the tunnel)
`postEvent()` (`:157-201`) sends:

```
POST https://<reserved-ngrok-domain>/api/robot/sensor/door?direction=entering|leaving
```

- **TLS:** the ngrok tunnel terminates TLS, so the firmware uses `WiFiClientSecure` with `setInsecure()` (skip cert
  validation) and a 15 s handshake timeout. A **fresh `WiFiClientSecure` is created per request** (more reliable than
  reusing one TLS context on the ESP32).
- **ngrok header:** `ngrok-skip-browser-warning: true` bypasses the free-tier interstitial page.
- **Auth (optional):** a Bearer token header is sent if `API_TOKEN` is set (empty = disabled).
- **Body:** none — `direction` travels in the query string.
- **Idempotent retry:** up to 3 attempts, but **only** on a connection-level failure (`code ≤ 0`, no HTTP response).
  Any real HTTP status (200/404/503) means the event was delivered, so it stops immediately — a crossing is never
  duplicated.

## 4. Backend side

`POST /api/robot/sensor/door` (`src/api/main.py:1383`, `robot_door_sensor(direction=...)`):

- Reads `direction` from the query string; validates it is `entering`/`leaving` (else 400).
- `entering` → the **server** drives the robot (navigate to the surveyed entrance + greet). `leaving` is acknowledged
  but **silently ignored**.
- The robot command is published by the server over **MQTT** (`cruzr/commands`) — note the **sensor never speaks MQTT**;
  that hop is entirely server→robot.

## 5. Data flow (end to end)

```
ESP32-S3 (2× HC-SR04)
   │  detect crossing + direction (A→B entering / B→A leaving)
   │  HTTPS POST  ?direction=…   (Wi-Fi → internet, TLS via WiFiClientSecure)
   ▼
ngrok tunnel  (fixed reserved domain, TLS termination)
   ▼
FastAPI  /api/robot/sensor/door
   │  entering → navigate-to-entrance + greet ;  leaving → ignored
   │  publish MQTT → cruzr/commands
   ▼
CRUZR robot  (executes greet/navigation)
```

## 6. Operational notes

- **Force the setup portal:** hold GPIO0/BOOT LOW at power-up → join AP `DoorSensorSetup` → pick Wi-Fi.
- **Run the backend for a demo:** `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` (not `127.0.0.1`) +
  `ngrok http --url=https://<reserved-ngrok-domain> 8000`; sanity-check the domain + `/docs`.
- **Arduino deps:** `tzapu/WiFiManager`, `HCSR04`; `WiFiClientSecure`/`HTTPClient`/`WiFi` ship with the ESP32 core. If
  the TLS build overflows flash: Tools → Partition Scheme → *Minimal SPIFFS (large APP)*.
- A `503` from the endpoint means the **robot/MQTT bridge** isn't connected yet — that's the robot side, not the sensor.

## 7. Legacy path (not used)

`robotics/door_sensor_bridge.py` is an **earlier, superseded** design where the Arduino streamed events over **USB
Serial** to a tethered PC that forwarded them to REST + MQTT. It is not referenced by the current system (the node now
posts directly over Wi-Fi/ngrok, removing the PC tether). It explains older one-line descriptions that read
"door sensor → PC server → MQTT".

## 8. Security note (for distribution)

- `setInsecure()` skips TLS certificate validation — acceptable here because the endpoint is a demo behind ngrok, but it
  means the channel is encrypted-but-unauthenticated. The optional `API_TOKEN` Bearer header is the intended hardening.
- The reserved ngrok domain is a public URL hard-coded in the firmware; treat it as semi-secret and **redact it in any
  published report** (use `<reserved-ngrok-domain>`).
