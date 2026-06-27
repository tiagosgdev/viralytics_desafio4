# Entrance Door Sensor — Architecture & Integration

## 1. Purpose

An autonomous **entrance sensor** detects a customer crossing the store doorway and its
**direction** (entering vs leaving). On entry it makes the CRUZR robot drive to the entrance
and greet the customer; on exit it bids farewell. The sensor itself holds no robot logic — it
only reports the event; the **server** decides the robot's behaviour.

## 2. Components & data flow

```
  ┌──────────────────┐   HTTPS POST (ngrok)    ┌─────────────────┐   MQTT (TLS)    ┌──────────────┐
  │  ESP32-S3 sensor │ ──────────────────────▶ │   API server    │ ──────────────▶ │ CRUZR robot  │
  │  2× HC-SR04      │  /api/robot/sensor/door │  (FastAPI)      │  cruzr/commands │ (Android app)│
  │  WiFiManager     │  ?direction=entering    │  CruzrBridge    │  greet / speak  │  navigate +  │
  │  power bank      │           │ leaving     │  + coordinates  │  / gesture      │  TTS + gesture│
  └──────────────────┘                         └─────────────────┘                 └──────────────┘
        detects crossing + direction          navigate-to-entrance decision        physical greeting
```

- **Sensor (ESP32-S3, battery-powered).** Two ultrasonic rangefinders (HC-SR04) placed in
  line across the doorway. A small **state machine** decides direction by **which sensor
  breaks first**: A-then-B = `entering`, B-then-A = `leaving`. It also estimates crossing
  speed/height (logged only). The board does **HTTP POST only** — it never touches MQTT or
  the robot.
- **API server (FastAPI).** Endpoint `POST /api/robot/sensor/door?direction=entering|leaving`.
  On `entering` it looks up the surveyed `Entrance` coordinate and tells the robot (via the
  `CruzrBridge`) to navigate there and greet; on `leaving` it sends a farewell. Direction is a
  query param; the body is ignored.
- **CRUZR robot.** The bridge publishes a `greet` command over **MQTT** (`cruzr/commands`); the
  robot's Android app drives the actual navigation + speech + gesture sequence.

## 3. Networking — why ngrok

**Constraint:** the API laptop is a Wi-Fi *client* with a DHCP IP that changes, and a single
Wi-Fi radio can't be both client and hotspot — so there is no stable LAN address to hard-code.
Hard-coded IP, laptop-as-AP, and mDNS were all rejected (unstable / extra hardware /
machine-dependent).

**Solution: an ngrok tunnel** gives a **fixed public URL** independent of laptop, IP and
network — the sensor and the server only each need internet, not the same Wi-Fi. A reserved
free static domain is hard-coded in the firmware as the API host. The firmware uses TLS
(`WiFiClientSecure`) and skips ngrok's free-tier browser interstitial via a header.

**Wi-Fi onboarding without re-flashing:** a **WiFiManager** captive portal (AP
`DoorSensorSetup`, forced by holding GPIO0/BOOT at power-up) lets staff pick the venue Wi-Fi
on-site; credentials persist in flash. Only Wi-Fi is configured in the field — the API URL is
fixed in code.

## 4. Directional logic (sensor state machine)

| Event order | Meaning | Sent as | Server action |
|---|---|---|---|
| Sensor **A** breaks, then **B** | walking **in** | `direction=entering` | navigate to entrance → greet (speak + gesture) |
| Sensor **B** breaks, then **A** | walking **out** | `direction=leaving` | farewell (speak + gesture) |

Both crossings are always reported. A "fast crossing" (run) flag is computed from the A→B
timing but is informational only — it no longer suppresses an event.

## 5. Robustness & status

- **Decoupled by design:** the sensor's only job is one HTTPS POST; all robot behaviour and
  spoken text live server-side, so the firmware never needs changing to adjust greetings or
  locations.
- **Graceful degradation:** if the robot bridge is offline the endpoint returns `503` (entry)
  or silently ignores a `leaving` event — the sensor keeps working regardless. The robot also
  tracks a **busy** state and ignores a greet while already serving a customer.
- **Verified end-to-end (2026-06-26):** a physical door crossing reaches the API over the
  fixed public URL on any network; the remaining dependency for a live greeting is the
  robot/MQTT side (the CRUZR app subscribed to `cruzr/commands` and a surveyed `Entrance`
  coordinate).
