# Codebase Architecture

## System Overview

This repository implements a fashion outfit detection and recommendation system, combining:

1. A computer-vision pipeline for clothing detection (YOLOv8 fine-tuning and a custom detector)
2. A pose-estimation pipeline for body shape analysis (MediaPipe)
3. A **multi-agent sealed-bid recommendation system** (SPADE over XMPP)
4. A semantic search subsystem for natural-language refinement (LNIAGIA)
5. A web application layer (FastAPI + browser frontend)
6. Training, evaluation, and dataset preparation tooling

---

## Architectural Layers

### 1. Runtime Application Layer

The deployable prototype that serves the end-user experience.

| Module | Responsibility |
|--------|---------------|
| `src/api/main.py` | FastAPI app: HTTP/WebSocket endpoints, model loading, inference, agent startup |
| `src/api/schemas.py` | Pydantic request/response models |
| `src/api/auth.py` | JWT authentication |
| `src/api/search_service.py` | Search session lifecycle management |
| `src/detection/detector.py` | `BaseDetector` + `FashionDetector` (YOLOv8 wrapper); `CATEGORY_NAMES` map (class ID 0–12) |
| `src/detection/fashionnet_detector.py` | FashionNet/edna inference wrapper |
| `src/detection/camera.py` | Multi-frame WebSocket camera session with confidence accumulation |
| `frontend/index.html` | Single-file browser UI (HTML + CSS + JS) |
| `frontend/static/css/style.css` | UI styles including agent status indicator |

**Design decisions:**
- The detector is loaded once at startup and shared across requests.
- `CATEGORY_NAMES` maps class IDs 0–12 to fashion labels. Any detection with an unmapped class ID gets a `class_N` fallback name and is filtered out before reaching the recommendation pipeline.
- Camera sessions average confidence across frames to reduce flicker.
- On scan completion the frontend immediately shows DB recommendations, then fires a background `POST /api/recommend` call. When the agent round completes (~20–30 s), the agent recommendations replace the initial results.

### 2. Multi-Agent Recommendation System

A SPADE 3.x multi-agent system running over XMPP (Prosody broker in Docker).

#### Agents

| Agent | JID | Role |
|-------|-----|------|
| `OrchestratorAgent` | `orchestrator@localhost` | Coordinates rounds, aggregates results |
| `FeatureWeightAgent` | `weights@localhost` | Computes feature importances + DB filters |
| `BodyRecommenderAgent` | `body@localhost` | Scores by body-shape compatibility |
| `ClothingRecommenderAgent` | `clothing@localhost` | Scores by garment-type / intent match |
| `ColourRecommenderAgent` | `colour@localhost` | Scores by colour harmony |
| `StockRecommenderAgent` | `stock@localhost` | Scores by inventory health (push_score) |

All extend `BaseRecommenderAgent` (thin wrapper over `spade.Agent` that disables TLS cert verification for the dev Prosody cert).

#### Communication Protocol

Messages use **FIPA-ACL performatives** (`REQUEST`, `INFORM`, `CFP`, `PROPOSE`) carried over XMPP. The negotiation pattern is a **sealed-bid Contract Net Protocol** — agents receive candidate items simultaneously and respond independently with no cross-talk.

```
orchestrator → weights     REQUEST   context: {detected_type, detected_color, detected_body_type, …}
weights      → orchestrator INFORM   {query, filters: {include, exclude}, weights: {color, type, bodyType}}
orchestrator → DB           (sync)   StockAgent.get_candidates() → 40 candidates
orchestrator → body         CFP      {candidates[40], weights_result, context}   ┐ sealed bid
orchestrator → clothing     CFP      (same, simultaneously)                       │ no cross-talk
orchestrator → colour       CFP      (same)                                       │
orchestrator → stock        CFP      (same)                                      ┘
body         → orchestrator PROPOSE  {agent_id, scores: {"itemId:size": 0.0–1.0}}
clothing     → orchestrator PROPOSE  (same)
colour       → orchestrator PROPOSE  (same)
stock        → orchestrator PROPOSE  (same)
→ weighted Borda count → top-10 → asyncio.Future resolved
```

Every message carries a `conv_id` (UUID hex) so concurrent rounds are never mixed up.

#### Aggregation (`aggregator.py`)

Weighted Borda count:
1. Each agent ranks 40 candidates by its score (sealed — agents don't see each other's rankings)
2. Borda points: rank-1 item gets N pts, rank-N gets 1 pt
3. Each agent's vector is scaled by its weight
4. Vectors summed → composite score → top-10

Weight distribution:
- `stock` receives a fixed 20% (inventory health signal)
- Remaining 80% split among `body`/`clothing`/`colour` proportionally to `FeatureWeightAgent` importances

**Fault tolerance:** if any scorer agent fails to respond before `COLLECT_TIMEOUT_S` (60 s), its weight is pooled and redistributed proportionally among the agents that did respond. The round completes with a reduced agent set.

#### Shared History (`history.py`)

A process-level singleton (`RoundHistory`) shared by all agents. Records every round through its lifecycle: `queued → running → complete / failed / stale`. Used for two purposes:

1. **Queue staleness check (orchestrator):** rounds that have been waiting in the queue longer than `QUEUE_TTL_S` (60 s) are dropped with `fut.set_result([])`. Prevents stale scans from being processed after the user has moved on.

2. **Agent context on comeback (scorer agents):** each scorer agent's `setup()` calls `history.agent_context_summary(agent_id)` to log what happened while it was offline — how many rounds ran, which it missed, and whether its absence triggered weight redistribution.

#### `RecommendationSystem` (`run.py`)

Public lifecycle wrapper used by FastAPI:
- `start()` — starts all 6 agents, registers with XMPP broker
- `recommend(…)` — creates `asyncio.Future`, calls `trigger_round()`, awaits result with 90 s timeout
- `stop()` — graceful shutdown
- `is_ready` — guards `/api/recommend`; returns 503 if False

Rounds are serialised through `OrchestratorBehaviour`'s `asyncio.Queue` — concurrent calls queue safely.

### 3. Custom Detector (FashionNet / edna)

A single-shot anchor-free detector built from scratch in PyTorch.

| Module | Responsibility |
|--------|---------------|
| `src/custom_model/model.py` | Architecture: ConvBnReLU, ResBlock, CSPBlock, Backbone, FPN Neck, DetectionHead |
| `src/custom_model/loss.py` | CIoU box loss, focal BCE objectness, BCE class loss, multi-scale target assignment |
| `src/custom_model/dataset.py` | YOLO-format dataset adapter with Albumentations augmentation |
| `src/custom_model/postprocess.py` | Grid decoding, NMS, confidence filtering |

Architecture: Input (3×640×640) → Backbone (4 downsampling stages, P3/P4/P5 at strides 8/16/32) → Neck (bidirectional FPN) → Head (objectness + class + bbox per scale).

| Scale | Parameters | Channel widths |
|-------|-----------|---------------|
| s | ~11.7M | 64-128-256-512 |
| m | ~25M | 96-192-384-768 |
| l | ~43M | 128-256-512-1024 |

### 4. Pose Analysis

MediaPipe Pose Landmarker is used to estimate body shape from the camera frame.

| Module | Responsibility |
|--------|---------------|
| `src/detection/pose_analyzer.py` | Extracts shoulder/hip widths, computes ratio, classifies body shape |

Body shapes classified: hourglass, pear, triangle, rectangle, inverted_triangle, apple, trapezoid, oval. Used as `detected_body_type` input to the agent round.

### 5. Stock Agent (`stock_agent/`)

SQLite-backed inventory database with scoring logic.

| Module | Responsibility |
|--------|---------------|
| `stock_agent/stock_agent.py` | `get_candidates(filters, n)` — retrieves N candidate items matching DB filters |
| `stock_agent/stock_stats.py` | `get_push_scores(pairs)` — precomputed push_score (stock age, sales velocity, stock count) |

`push_score` is a composite signal indicating how urgently the store wants to move a given item. Normalised to [0, 1] across the 40 candidates for each round.

### 6. LNIAGIA Search Subsystem

Semantic natural-language clothing search used by Cruella.

| Module | Responsibility |
|--------|---------------|
| `LNIAGIA/search_app.py` | Entry point, `search_detected_items()` called by FastAPI |
| `LNIAGIA/feature_weighting.py` | `analyze_intent()` — converts context to DB filters + feature importances |
| `LNIAGIA/llm_query_parser.py` | LLM-based query parsing (Ollama + qwen2.5) |
| `LNIAGIA/DB/SQLLite/DBManager.py` | Structured item storage (clothing.db) |
| `LNIAGIA/DB/vector/VectorDBManager.py` | Qdrant vector search with BGE embeddings |
| `LNIAGIA/DB/vector/description_generator.py` | Item struct → natural-language text for embedding |

Hybrid retrieval: SQL metadata filters + semantic vector similarity. The LLM (via `analyze_intent`) translates visual context into structured filters; these are applied as Qdrant metadata constraints.

### 7. Data Pipeline

| Module | Responsibility |
|--------|---------------|
| `scripts/data_prep/sample_balanced.py` | Stratified sampling from DeepFashion2 CSV metadata |
| `scripts/data_prep/analyze_raw_dataset.py` | EDA: class balance, box sizes, occlusion stats |

### 8. Training and Evaluation

| Module | Responsibility |
|--------|---------------|
| `scripts/training/train_custom.py` | FashionNet training loop with full experiment configuration |
| `scripts/evaluation/evaluate_custom.py` | FashionNet/edna evaluation with custom metrics |
| `src/utils/metrics.py` | IoU, AP, confusion matrix — implemented from first principles |

---

### 9. Android Robot App (`android_app/`)

A Kotlin single-Activity app (`com.viralytics.mobile`) installed on the **UBTech Cruzr robot**. The same APK runs on two device types, detected at runtime.

#### Dual mode

```kotlin
enum class AppMode { TABLET, PHONE_CAMERA }
```

Detection: `Robot.globalContext() != null` → `TABLET` (robot screen); otherwise `PHONE_CAMERA` (regular phone used as camera because the robot tablet's built-in camera is unusable). Both devices can have a SharedPreferences override (`device_mode_override`).

| Mode | Role |
|------|------|
| `PHONE_CAMERA` | Shows camera button only. Uploads photo to `/api/mobile/scan`, then shows a toast. Never renders recommendations. |
| `TABLET` | Hides camera button. Listens on MQTT `cruzr/scan_result` for scan results. Renders detection cards, recommendations, chat, and body shape. Controls robot navigation, speech, and gestures via Cruzr SDK. |

#### Architecture

Repository + ViewModel pattern. All network I/O is `suspend fun … = withContext(Dispatchers.IO) { runCatching { … } }` in Repository classes. `MainActivity` only calls ViewModel methods and observes `LiveData<UiEvent>`.

| File | Responsibility |
|------|---------------|
| `MainActivity.kt` | UI binding, SDK callbacks (MQTT, navigation, LIDAR, session), robot hardware control |
| `MainViewModel.kt` | Session state, `UiEvent` emission via `MutableLiveData`, coroutine launch |
| `ScanRepository.kt` | Multipart POST `/api/mobile/scan` (scales image ≤1280 px) |
| `AgentRepository.kt` | POST `/api/recommend` — agent round results |
| `ChatRepository.kt` | POST `/api/chat` with conversation history |
| `SessionRepository.kt` | POST `/api/session/start` |
| `CameraActivity.kt` | Fullscreen CameraX with 5-second auto-countdown |

#### Scan result pipeline (phone → tablet)

```
Phone:  CameraActivity (5s countdown)
         → viewModel.uploadScan()
         → POST /api/mobile/scan

Server: builds DetectionResponse + agent recs
         → BackgroundTasks: publish_scan_result()
         → MQTT publish cruzr/scan_result (QoS 1)

Tablet: messageArrived("cruzr/scan_result")
         → handleScanResult(json)
         → viewModel.injectScanResult()       ← single postValue (no coalescing)
         → UiEvent.ScanComplete observer
         → renderDetections() + renderRecommendations() + updateAnnotatedImage()
         → fetchAgentRecommendations() (HTTP, tablet has server URL)
```

#### Scan image display

Two frames arrive per scan: `annotatedFrame` (clothing bounding boxes + colour labels) and `bodyFrame` (skeleton pose). Stored as `currentAnnotatedFrame` / `currentBodyFrame` on `MainActivity`.

- **Main view:** `updateAnnotatedImage()` prefers `annotatedFrame`; falls back to `bodyFrame` only if clothing frame is absent.
- **Fullscreen toggle:** tapping `resultImage` opens a dialog. When both frames exist, a non-dismissing Neutral button switches between "Skeleton view" and "Clothing view" without closing the dialog (button listener wired via `dialog.getButton(BUTTON_NEUTRAL).setOnClickListener {}` after `.show()`).
- **Body shape label:** `bodyShapeLabel` is a `TextView` overlaid at `top|end` of the `FrameLayout` wrapping `resultImage`. Background: `mobile_body_shape_bg.xml` — dark semi-transparent pill (`#BB000000`, 20 dp corners).

#### Agent recommendation enrichment

`rec_system.recommend()` returns items with only `item_id` and scoring fields — no `image_url`, `name`, or `description`. `_enrich_agent_results_with_db()` in `src/api/main.py` queries `SELECT * FROM items WHERE id IN (...)` on `clothing.db` and merges image, name, description, and a human-readable score summary before publishing over MQTT or returning from `/api/recommend`. The Android `fromAgentJson()` companion method reads `image_url` from the enriched payload.

#### MQTT topics

| Topic | Direction | Purpose |
|-------|-----------|---------|
| `cruzr/scan_result` | Server → Tablet | Scan detections + DB recommendations + annotated frame |
| `cruzr/commands` | Server → Tablet | Navigation, speech, gesture commands |
| `cruzr/persona` | Server → Both | Persona sync |
| `cruzr/status` | Tablet → Server | Navigation events, session lifecycle |

Broker: `tcp://<server_ip>:1883`. Stable client IDs (`viralytics_tablet` / `viralytics_phone`) + `isCleanSession=false` + `isAutomaticReconnect=true` — broker persists subscriptions across reconnects.

#### Cruzr SDK v2.8.0

`compileOnly` JAR (`libs/cruzr-sdk-2_8_0.jar`) — robot supplies the real implementation at runtime. **Must never change to `implementation`** (class conflicts). Key managers: `NavigationManager` (marker-based and coordinate navigation), `SpeechManager` (TTS), `MotionManager` (gestures), `CruzrSensorManager` (LIDAR for person detection).

Navigation known issue: `navigate()` returns code `-11 / find_plan_failed` for marker-based nav. Hypothesis: map polylines don't connect robot position to target. `TRACK_MODE` constant (currently `false`) can be flipped to test polyline-constrained planning.

---

## Technology Choices

| Technology | Rationale |
|------------|-----------|
| PyTorch | Standard framework for custom model research |
| Ultralytics YOLOv8 | Strong baseline; minimal code; easy comparison |
| FastAPI | Async networking, schema-driven APIs, WebSocket support |
| SPADE 3.x | Python FIPA-compliant multi-agent framework over XMPP |
| Prosody (Docker) | Lightweight XMPP broker; custom Dockerfile bypasses `setpriv` on WSL2 |
| slixmpp | SPADE's underlying XMPP client; STARTTLS with `verify_security=False` for self-signed cert |
| asyncio.Queue | Serialises concurrent recommend() calls through a single CyclicBehaviour |
| MediaPipe | Reliable on-device pose estimation; heavy task model for accurate landmark detection |
| Qdrant | Local vector DB with metadata filtering, no external service required |
| sentence-transformers (BGE) | Strong general-purpose retrieval embeddings |
| Ollama + qwen2.5 | Local LLM for query parsing, no API key required |
| Albumentations | Correct bounding box transformation during augmentation |
| OpenCV + NumPy | Standard CV pipeline tooling |

---

## Main Application Flow

```
Browser (WebSocket /ws/camera)
  → Camera captures frames → multi-frame confidence accumulation
  → Detector (YOLOv8 / FashionNet) → clothing class detections
  → PoseAnalyzer (MediaPipe) → body shape classification
  → DB search (LNIAGIA/search_app) → immediate recommendations sent to browser
  → triggerAgentRecommendations() called in browser JS (non-blocking)
      → POST /api/recommend
          → RecommendationSystem.recommend()
              → OrchestratorAgent.trigger_round()         (queued)
              → history.record_enqueued() + staleness check
              → REQUEST → FeatureWeightAgent → INFORM weights+filters
              → StockAgent.get_candidates() → 40 items
              → CFP → body, clothing, colour, stock (sealed bid)
              → PROPOSE × 4 → weighted Borda count → top-10
              → history.record_complete()
          → JSON response to browser
      → renderRecs() replaces DB recs with agent recs
```

**Fault path:** if a scorer agent is down, `_collect_proposals` times out for that agent. The orchestrator detects the missing agent, logs a warning in the XMPP trace, redistributes its weight to the remaining agents, and the round completes normally. On restart the missing agent reads `history.agent_context_summary()` to understand what it missed.
```
