# Door sensor → API via ngrok (implemented & verified)

> Status: **implemented and verified end-to-end (2026-06-26).** A door event reaches
> the API over a fixed public URL regardless of network. The remaining 503 is the
> robot/MQTT side, not the sensor.

## What the sensor does

- ESP32-S3 (2× HC-SR04, runs off a power bank) → **HTTP POST only**:
  `POST https://<ngrok-domain>/api/robot/sensor/door?direction=entering|leaving`
  (`src/api/main.py:1339`; `direction` is a query param, body ignored).
- It does **not** touch MQTT or the robot directly. The **server** decides the robot's
  behaviour and spoken text (entering → navigate + greet; leaving → server ignores it).
- **enter and leave are always sent.** The "run" speed flag (`alerta`, a fast crossing)
  is still computed and logged to Serial, but no longer suppresses a crossing.

## Networking — why ngrok

The hard constraint: the laptop is a **Wi-Fi client** (no Ethernet), so its IP is
DHCP-assigned and changes; and a single Wi-Fi radio can't be client + access point, so
the laptop can't be a fixed-IP hotspot. Options like a hardcoded LAN IP, laptop-as-AP,
or mDNS were all rejected (unstable, needs extra hardware, or machine-dependent).

**ngrok** gives a fixed public URL that is independent of laptop, IP and network — the
sensor and laptop only each need internet, and need not be on the same Wi-Fi. Whichever
laptop runs the API + tunnel (with the account's authtoken) is reachable at the same URL.

- **Reserved free static domain** (free tier includes one; permanent until deleted):
  `unaltering-unabjectly-micha.ngrok-free.dev`, hardcoded as `API_HOST` in the sketch.
- Firmware uses **`WiFiClientSecure` + `setInsecure()`** (the tunnel is TLS) and sends
  `ngrok-skip-browser-warning: true` (skips the free-tier interstitial).
- **WiFiManager** (`tzapu/WiFiManager`) captive portal handles **Wi-Fi selection only**
  (`DoorSensorSetup` AP; hold GPIO0/BOOT at power-up to force it). The API URL is fixed
  in code, so nothing else needs configuring.

## Run it (on the laptop, each demo)

```
uvicorn src.api.main:app --host 0.0.0.0 --port 8000          # NOT 127.0.0.1
ngrok http --url=https://unaltering-unabjectly-micha.ngrok-free.dev 8000
```
(Older ngrok: `--domain=` instead of `--url=`.) Sanity check: open the domain + `/docs`.

## Dependencies (Arduino IDE)

- `tzapu/WiFiManager` (Library Manager). `WiFiClientSecure`/`HTTPClient`/`WiFi` ship with
  the ESP32 core. `PubSubClient`/MQTT are no longer used.
- If the TLS build overflows flash: **Tools → Partition Scheme → Minimal SPIFFS (large APP)**.

## Remaining (robot side, not the sensor)

- The endpoint returns **503** until `robot_bridge` is connected
  (`CruzrBridge` → `test.mosquitto.org:8883` TLS, 8s window at API startup,
  `src/api/main.py:410`). Restart the API and look for `🤖  Cruzr robot bridge connected`.
- For an actual greet: the CRUZR Android app must be subscribed to `cruzr/commands` on
  the same broker, and an `entrance` location must be surveyed in `coordinates.json`.
