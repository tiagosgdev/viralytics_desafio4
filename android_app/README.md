# Viralytics Mobile — Android App

Native Android client for the Viralytics retail assistant system, designed to run on a **UBTECH CRUZR robot tablet** and a companion **phone camera**.

---

## Architecture

The system uses **two devices running the same APK**, detected at runtime:

| Device | Mode | Role |
|--------|------|------|
| CRUZR robot tablet | `TABLET` | Displays recommendations, controls robot navigation/speech/gestures |
| Regular phone | `PHONE_CAMERA` | Captures customer outfit photos, uploads to server |

**Scan flow:**
```
Phone captures photo
  → POST /api/mobile/scan (PC server)
  → Server publishes result to MQTT cruzr/scan_result
  → Tablet receives via MQTT, renders detections + recommendation cards
```

**Navigation flow:**
```
Door sensor → PC server → MQTT cruzr/commands (greet)
  → Robot navigates to entrance → LIDAR detects customer
  → Speak + gesture → customer session starts
  → Customer scans outfit → AI recommends item
  → Robot navigates to product stand → session ends → return to entrance
```

---

## Requirements

1. PC server running on the local network:
   ```powershell
   .\scripts\app\start_full_app.ps1 -BindHost 0.0.0.0
   ```
2. MQTT broker running on PC port 1883 (Mosquitto).
3. All devices on the same Wi-Fi network.
4. Windows Firewall allowing inbound TCP on ports `8000` and `1883`.

---

## Build & Install

Open `android_app/` in Android Studio (Iguana or newer).

Requirements:
- Android SDK 34
- JDK 17
- Gradle sync will fetch all dependencies

The CRUZR SDK (`libs/cruzr-sdk-2_8_0.jar`) is `compileOnly` — the robot provides the real implementation at runtime. Do not change it to `implementation`.

Build and install on both devices:
```
Build → Run (or Shift+F10)
```

---

## First-run setup

In the app on **each device**, tap the settings icon (⚙️) and set:

- **Server URL**: `http://192.168.x.x:8000` (your PC's LAN IP)
- **Device mode**: Phone or Tablet

The app auto-detects tablet mode via the CRUZR SDK at startup. If auto-detection is wrong, override it in settings.

Find your PC's LAN IP:
```powershell
ipconfig
```

---

## Personas

Two AI stylist personas selectable at first launch or via "Switch Stylist":

| Persona | Vision | Text | Matching |
|---------|--------|------|---------|
| **Cruella** | YOLO | LLM | Strict |
| **Edna** | FashionNet | Custom NLP | Flexible |

Persona is synced across devices via MQTT.

---

## MQTT topics

| Topic | Direction | Purpose |
|-------|-----------|---------|
| `cruzr/commands` | Server → Tablet | Navigation, speech, gesture commands |
| `cruzr/scan_result` | Server → Tablet | Scan results from phone uploads |
| `cruzr/status` | Tablet → Server | Navigation events, robot state |
| `cruzr/persona` | Bidirectional | Persona sync |

---

## Supported MQTT commands (tablet)

```json
{"action": "move_to_stand", "target": "marker-name"}
{"action": "move_to_coords", "x": 1.0, "y": 2.0, "theta": 0.0}
{"action": "speak", "text": "Hello!"}
{"action": "guide_user", "target": "marker-name", "intro_text": "Follow me!"}
{"action": "greet", "x": 0.0, "y": 0.0, "theta": 0.0, "text": "Welcome!", "gesture": "wave"}
{"action": "gesture", "name": "wave"}
{"action": "farewell", "text": "Goodbye!", "gesture": "goodbye"}
{"action": "locate_self"}
{"action": "get_status"}
```

---

## Customer session lifecycle

1. `greet` command → robot navigates to entrance, waits with LIDAR (25s timeout)
2. Customer detected → speaks welcome + gesture → 3-minute session timer starts
3. Customer scans outfit on phone → recommendations appear on tablet
4. Customer taps "Take me there!" → robot navigates to product stand
5. Arrival → session ends → robot returns to entrance automatically

---

## Robot navigation

Navigation uses the map configured in the UBTECH cloud dashboard. Markers placed on the map become navigation targets by name.

Known issue: `navigate()` may fail with error code `-11` (`find_plan_failed`) if the map's polyline routes don't connect the robot's current position to the target marker. Ensure routes are drawn and synced from the dashboard before deploying.

Navigation config (in `MainActivity.kt`):
```kotlin
private val TRACK_MODE = false   // true = follow polyline routes; false = point-to-point
private val NAV_MAX_SPEED = 0.5f
```

---

## Logcat filters

| Tag | Content |
|-----|---------|
| `CruzrNav` | Navigation events, map markers, polylines, errors |
| `CruzrApp` | MQTT commands, gestures, LIDAR, TTS |

---

## Project structure

```
android_app/
├── app/src/main/java/com/viralytics/mobile/
│   ├── MainActivity.kt        # UI + robot SDK callbacks
│   ├── MainViewModel.kt       # State + UiEvent emission
│   ├── CameraActivity.kt      # CameraX fullscreen with 5s countdown
│   ├── ScanRepository.kt      # POST /api/mobile/scan
│   ├── ChatRepository.kt      # POST /api/chat
│   ├── SessionRepository.kt   # POST /api/session/start
│   └── AgentRepository.kt     # POST /api/recommend
├── app/src/main/res/
│   ├── layout/activity_main.xml
│   ├── layout/activity_camera.xml
│   ├── values/colors.xml      # Brand + Cruella theme colors
│   ├── values/dimens.xml      # Base dimensions (< sw480dp)
│   ├── values-sw480dp/dimens.xml
│   └── values-sw600dp/dimens.xml
└── libs/
    └── cruzr-sdk-2_8_0.jar   # compileOnly — provided by robot at runtime
```
