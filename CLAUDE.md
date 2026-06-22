# CLAUDE.md — Project Context & Handoff

> This file gives an AI coding assistant the full context of this project and the
> in-progress CRUZR robot navigation investigation. It was written to hand off work
> from a separate chat session into the Claude Code VS Code extension.

---

## 1. Project Overview

**App package:** `com.viralytics.mobile`
**Platform:** Android app that runs *on a UBTECH CRUZR robot's built-in Android tablet*.
**Language:** Kotlin (single-Activity app, `MainActivity.kt`).
**Purpose:** A retail assistant. The robot scans products via camera, gets AI
recommendations from a PC server, speaks to the customer, and physically navigates
to product locations ("stands") in the store.

### Moving parts
- **Android app (this project):** UI, camera scan upload, chat refinement, MQTT
  listener, and CRUZR hardware control (navigation + speech).
- **PC server (Python, separate):** Receives camera scans at `/api/mobile/scan`,
  handles chat refinement at `/api/chat`, and publishes commands to the robot over MQTT.
- **MQTT broker:** Runs on the PC (port 1883). The app subscribes to topic
  `cruzr/commands`. The PC publishes JSON commands like:
  - `{"action": "move_to_stand", "target": "<marker name>"}`
  - `{"action": "speak", "text": "<text>"}`
- **CRUZR SDK v2.8.0:** Provided by the robot at runtime as a `compileOnly` JAR
  (`cruzr-sdk-2_8_0.jar`). Do NOT bundle it with `implementation` — it must be
  `compileOnly` or you get class conflicts. The robot supplies the real implementation.

### Server URL / connection
- Stored in SharedPreferences (`viralytics_mobile` / `server_url`), default
  `http://192.168.1.80:8000`. The MQTT broker IP is derived from this same URL.

---

## 2. CRUZR SDK v2.8.0 — Key Facts (verified by decompiling the JAR)

Architecture: **Manager → Proxy → Master Bridge (IPC)**. The app talks to managers,
which serialize commands as Parcelables over the "Hermes" IPC bridge to the robot's
system services. Almost everything returns a `Promise` or `ProgressivePromise`
(async; callbacks may arrive on a background thread).

Managers are obtained via `Robot.globalContext().getSystemService("name")` after
`Robot.initialize(applicationContext)`.

### Navigation API (confirmed from bytecode)
`NavigationManager` methods:
- `getCurrentNavMap(): Promise<NavMap>` — **returns a Promise** (async, use `.done{}`).
- `setCurrentNavMap(String id): Promise`
- `locateSelf(): ProgressivePromise<Location, LocatingException, LocatingProgress>`
- `isSelfLocated(): Boolean`
- `navigate(Location): ProgressivePromise`
- `navigate(NavigationOption): ProgressivePromise`  ← the real entry point; the
  `navigate(Location)` overload just wraps the location in a default NavigationOption.

### Class hierarchy (important)
- `Marker extends Location` — so a `Marker` **can be passed directly** anywhere a
  `Location` is expected, including `NavigationOption.Builder(...)`. No need to rebuild it.
- `Location` fields: `position: Point`, `z: Float`, `rotation: Float`.
- `Point` fields: `x: Float`, `y: Float`.
- `NavMap` fields: `id`, `name`, `scale`, `markerList: List<Marker>`,
  `polylineList: List<Polyline>`, `navFile`, `navFileUrl`, etc.
  - `getMarkerList()`, `getMarker(id)`, `getPolylineList()`, `getPolyline(id)`.
- `Polyline` fields: `id`, `name`, `description`, `locationList: List<Location>`.
  **This is what a drawn route/track becomes in the map data.**
- `NavigationOption` fields: `destination: Location`, `maxSpeed: Float`,
  `retryCount: Int`, `retryInterval: Int`, `trackMode: Boolean`.
  - Builder: `NavigationOption.Builder(Location).setMaxSpeed(f).setRetryCount(i)
    .setRetryInterval(i).setTrackMode(bool).build()`

### Error codes
- `NavigationConstants` does NOT define the runtime nav error codes.
- `NavigationException extends ExceptionWithCode` (has `getCode()`).
- **The `-11 / find_plan_failed` error is NOT from the SDK.** It is relayed from the
  robot's underlying path planner. This means it is a **map/route data problem**, not
  something fixable in app code.

---

## 3. THE CORE PROBLEM (current blocker)

**Goal:** Send the robot from its current position to a named marker (e.g. "Teste 3",
"Teste 4") by code, triggered via MQTT `move_to_stand`.

**Symptom:** `nav.navigate(...)` fails with `find_plan_failed` (code `-11`).

### What is confirmed working
- MQTT receive + parse: WORKS.
- Speech / TTS (native + `com.svox.pico` fallback): WORKS.
- Marker lookup by title (`markerList.find { it.title.equals(target, ignoreCase=true) }`):
  WORKS — the marker is found, with valid coordinates (e.g. `Point{x=85.33, y=98.67}`).
- `Location` builds fine from the marker position.
- The `navigate()` call itself executes and returns a clean `-11` failure (no crash).
- **Manual navigation via the robot's native tablet UI WORKS** — the robot drives to
  the same points without problems.

### What is NOT working
- Programmatic `navigate()` fails to plan a path (`-11`), even though:
  - A map is active on the tablet, robot is self-located.
  - The robot's position is near a route node.
  - Manual navigation to the same point works.

### Leading hypothesis (NOT yet confirmed)
The native tablet UI navigates using the **track/polyline network** (route-constrained
planning). The app's plain `navigate(Location)` call defaults to `trackMode = false`
(free-space planning), which the planner can't satisfy → `find_plan_failed`.

**Proposed fix to test first:** call `navigate(NavigationOption)` with
`setTrackMode(true)`. This is implemented in the current `MainActivity.kt` (v1.5),
gated behind a `TRACK_MODE` constant so it can be flipped to `false` to A/B test.

> IMPORTANT CAVEAT: This is an inference, not proven. The path planner lives on the
> robot, not in the JAR, so the exact behavior of `trackMode` can't be verified from
> code. It's a one-line change worth testing. If `trackMode=true` still returns `-11`,
> the problem is in the map data itself (see below).

### If trackMode does NOT fix it — the map-data angle
- Routes are authored on the **UBTECH cloud dashboard**, NOT freely from the SDK.
  The SDK can read/replace whole `NavMap` objects (`addNavMap`/`modifyNavMap`) but
  there is no documented builder to construct markers/routes programmatically.
- Dashboard workflow: physical SLAM mapping first → place markers → draw routes →
  press **Sync** (manual, ~5s–1m) → robot updates. Map editing requires a prior
  physical mapping session; markers/routes cannot be placed purely virtually.
- The map editor has: **Posição** (markers), **Pista virtual** (virtual track/route),
  **Parede virtual** (walls), **Zona de segurança/declive** (zones).
- **Open question / suspected root cause:** the user could only create ONE "Pista
  virtual" line, and it does not visibly connect the position points. A single
  polyline that doesn't pass through (or very near) both the robot's localized
  position AND the target marker won't give the planner a usable path. The route
  graph likely needs to actually connect start → target.
- The "Guia / Projeto da rota" section is a **tour-guide route system** (ordered
  Ponto inicial → Ponto de guia → Terminal). This is SEPARATE from free navigation
  and is probably NOT what `nav.navigate()` consumes.

### Next diagnostic steps
1. Test `trackMode=true` (already wired up). Capture Logcat filtered by `CruzrNav`.
2. If still `-11`: inspect the map's `polylineList` at runtime — log each polyline's
   name and its `locationList` points — to see whether any route actually spans the
   robot's position to the target marker.
3. Compare: does manual navigation use track mode implicitly? (It works, so its mode
   is the one to replicate.)
4. On the dashboard: verify routes genuinely connect the relevant markers, re-sync,
   and confirm the sync landed (there's a sync log / "Registro de Sincronização").

---

## 4. Code Notes / Gotchas (from the debugging history)

- **`getCurrentNavMap()` is async (Promise).** Use `.done { navMap -> ... }`. Earlier
  confusion about it being synchronous was wrong — it compiles with `.done{}` because
  it genuinely returns a Promise.
- **Do NOT add a `startActivity()` "focus-stealing" hack around navigation.** An
  earlier v1.4 attempt relaunched the activity mid-navigation to "steal locomotion
  focus." Navigation has nothing to do with window focus; this just risked tearing
  down the IPC binder connection (seen as `MST|Master...Connection: Remote binder
  maybe crashed` in logs, which killed the process). It has been removed in v1.5.
- **Speech module warnings are expected.** Logs show `Module Synthesizer NOT found,
  use SynthesizerProxy` and `Native voice failed: Synthesizer uninitialized`. The
  app's dual-engine `speakText()` falls back to Android `com.svox.pico` TTS by design.
- **`rosa.jar` warning is benign:** `Asset path '/system/framework/rosa.jar' does not
  exist or contains no resources` appears at startup but doesn't block functionality.
  (ROSA must still be declared via `<uses-library>` in the manifest for SDK init.)
- **Threading:** Promise callbacks may arrive off the main thread; all UI updates are
  wrapped in `runOnUiThread { ... }`.
- **Version tracker:** `APP_VERSION` constant is bumped each build and shown in the
  status bar so you can confirm the robot actually installed the new APK.

---

## 5. Current `findMarkerAndNavigate` logic (v1.5)

```kotlin
// Pseudocode summary of the current approach:
nav.currentNavMap.done { navMap ->
    val marker = navMap.markerList?.find { it.title.equals(target, ignoreCase = true) }
        ?: return@done  // "marker not found"

    // Marker IS a Location, so pass it straight in.
    val option = NavigationOption.Builder(marker)
        .setTrackMode(TRACK_MODE)        // <-- key hypothesis; default true in v1.5
        .setMaxSpeed(0.5f)
        .setRetryCount(2)
        .setRetryInterval(2000)
        .build()

    nav.navigate(option)
        .done { /* arrived */ }
        .progress { /* log progress */ }
        .fail { e -> /* log e.message + e.code */ }
}
```

---

## 6. Tasks the user may want help with next

- Confirm/deny the `trackMode` hypothesis once tested on hardware.
- Add runtime logging of `navMap.polylineList` to diagnose the route graph.
- General review of the Android app and the Python PC server integration.
- Harden MQTT (reconnect handling, QoS, the connection-lost callback).
- Possibly: investigate whether map sync can be triggered programmatically
  (currently a manual dashboard button press).

---

## 7. Security / hygiene reminders

- Do not commit MQTT credentials, the UBTECH dashboard login, or any API keys.
- Add a `.claudeignore` excluding: `build/`, `.gradle/`, `.idea/`, `*.apk`,
  `cruzr-sdk-2_8_0.jar` (compileOnly, large, not needed for context), and any
  files containing secrets.
