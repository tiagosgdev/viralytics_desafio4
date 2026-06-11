# Viralytics — FashionSense
> Master's Project | Computer Vision + Multi-Agent Fashion Recommendation

FashionSense is a fashion outfit detection and recommendation system. It detects the clothing a person is wearing via live camera, analyses their body shape, and runs a **multi-agent sealed-bid recommendation round** to return personalised store recommendations. Two runtime personas are available:

- **Cruella** — YOLOv8-based outfit detection + LLM-powered semantic search
- **Edna** — Custom FashionNet (edna) outfit detection + local text parsing

---

## Requirements

- Python 3.10+
- **Docker Desktop** (required for the XMPP broker used by the multi-agent system)
- `ffmpeg` installed and on PATH (voice transcription)
- Ollama running locally (Cruella's LLM backend)
- CUDA GPU recommended for training; CPU and Apple MPS also supported

Install Python dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Model Weights

The app auto-discovers fine-tuned YOLOv8 weights under `models/weights/`. Without weights it falls back to the base `yolov8n.pt` (COCO — inaccurate for fashion). Place trained weights at:

```
models/weights/
└── yolov8n_fashion/        # or yolov8s_fashion / yolov8m_fashion / yolov8l_fashion
    └── weights/
        └── best.pt
```

Pre-trained edna model weights are available on SharePoint:

**[Download model weights](https://myisepipp-my.sharepoint.com/:u:/g/personal/1140331_isep_ipp_pt/IQCCukBlXdvVRoOwaETcMQwwAbNnmYm78sjY0hEas7aCeS4?e=1mLUmX)**

Extract and place under `models/weights/` following the structure above.

---

## Running the App

A single PowerShell script starts everything — Ollama check, vector DB check, XMPP Docker container, and the FastAPI server:

```powershell
.\scripts\app\start_full_app.ps1
```

Open `http://127.0.0.1:8000` in a browser.

**LAN / mobile testing:**

```powershell
.\scripts\app\start_full_app.ps1 -BindHost 0.0.0.0
```

**Key flags:**

| Flag | Description |
|------|-------------|
| `-BindHost <ip>` | Bind host — use `0.0.0.0` for LAN access (default: `127.0.0.1`) |
| `-BindPort <n>` | Bind port (default: `8000`) |
| `-Reload` | Enable uvicorn auto-reload (development only) |
| `-SkipOllama` | Skip Ollama check (Edna-only mode) |
| `-SkipVectorCheck` | Skip vector DB check |
| `-SkipXmpp` | Skip XMPP Docker container (disables multi-agent recommendations) |
| `-AutoPullModel` | Automatically pull required Ollama model if missing |

**Android app:**

1. Open `android_app/` in Android Studio
2. Set the server IP to your machine's LAN IP
3. `Build > Build Bundle(s) / APK(s) > Build APK(s)`

---

## Multi-Agent Recommendation System

FashionSense uses a **SPADE multi-agent system** to produce recommendations. Six XMPP agents coordinate via a sealed-bid Contract Net Protocol:

```
OrchestratorAgent  — coordinates rounds, aggregates results
FeatureWeightAgent — computes feature importances and DB filters from context
BodyAgent          — scores items by body-shape compatibility
ClothingAgent      — scores items by garment-type / intent match
ColourAgent        — scores items by colour harmony
StockAgent         — scores items by inventory health (push_score)
```

**Round protocol per scan:**
1. Browser scan detects clothing type + body shape via camera
2. FastAPI calls `POST /api/recommend` asynchronously (non-blocking — DB recs appear immediately)
3. Orchestrator requests feature weights, retrieves 40 candidates from the stock DB
4. CFP broadcast to all four scorer agents simultaneously (sealed bid — no cross-talk)
5. Each agent responds with a `PROPOSE` containing item scores
6. Weighted Borda count aggregation → top-10 recommendations
7. Agent recommendations replace the initial DB results in the UI

**Fault tolerance:** if any scorer agent fails to respond before the timeout, its weight budget is redistributed proportionally among the remaining agents and the round completes normally.

**Shared history:** all agents share a round history log. When an agent comes back online after a failure it reads a summary of what ran while it was absent.

The XMPP broker (Prosody) runs in Docker and is started automatically by `start_full_app.ps1`. To manage it manually:

```bash
docker compose up -d xmpp        # start broker
docker compose logs -f xmpp      # view logs
docker compose down xmpp         # stop broker
```

---

## Datasets

### DeepFashion2 (required for training)

Dataset source: [Kaggle — DeepFashion2 Original with Dataframes](https://www.kaggle.com/datasets/thusharanair/deepfashion2-original-with-dataframes?resource=download)

Download and extract to `data/raw/` with this structure:

```
data/raw/
├── train/
│   ├── image/
│   └── annos/
├── validation/
│   ├── image/
│   └── annos/
└── DeepFashion2/
    └── img_info_dataframes/
        ├── train.csv
        └── validation.csv
```

### Build the balanced dataset

```bash
python scripts/data_prep/sample_balanced.py \
    --train_csv data/raw/DeepFashion2/img_info_dataframes/train.csv \
    --val_csv   data/raw/DeepFashion2/img_info_dataframes/validation.csv \
    --img_dirs  data/raw/train/image data/raw/validation/image \
    --output_dir data/balanced_dataset \
    --n_per_class 7641 \
    --seed 42
```

This produces an 84,051-image balanced dataset across 11 classes (70/15/15 split).

### Background images (required for edna_1.4m+)

```bash
python scripts/data/download_bg_images.py
cp bg_images/*.jpg data/balanced_dataset/images/train/
for f in bg_images/*.jpg; do
    touch data/balanced_dataset/labels/train/$(basename $f .jpg).txt
done
```

---

## Training

```bash
# GPU (recommended)
python scripts/training/train_custom.py \
    --data data/balanced_dataset \
    --model_scale m \
    --epochs 100 \
    --batch 32 \
    --lr 0.001 \
    --lambda_box 5.0 \
    --lambda_obj 1.0 \
    --lambda_cls 0.5 \
    --augment medium \
    --multi_cell \
    --optimizer adamw \
    --weight_decay 0.01 \
    --device cuda \
    --output models/weights/yolov8n_fashion

# CPU (slow — testing only)
python scripts/training/train_custom.py --epochs 10 --batch 4 --device cpu
```

---

## Evaluation

```bash
python scripts/evaluation/evaluate_custom.py \
    --weights models/weights/yolov8n_fashion/weights/best.pt \
    --data data/balanced_dataset \
    --conf 0.25
```

---

## Project Structure

```
viralytics_desafio4/
├── android_app/                    # Native Android client
├── data/
│   └── raw/                        # DeepFashion2 raw data (not committed)
├── docs/organized/                 # Research documentation
├── frontend/
│   ├── index.html                  # Single-file browser UI
│   └── static/css/style.css
├── LNIAGIA/                        # Semantic search subsystem (Cruella)
│   ├── search_app.py
│   ├── llm_query_parser.py
│   ├── feature_weighting.py
│   └── DB/
│       ├── SQLLite/                # Structured item storage (clothing.db)
│       └── vector/                 # Qdrant vector search
├── models/
│   └── weights/                    # Trained model weights (not committed)
│       ├── yolov8n_fashion/weights/best.pt
│       └── mediapipe/              # Pose landmarker model
├── multi_agent/                    # SPADE multi-agent recommendation system
│   ├── config.py                   # JIDs, timeouts, budget constants
│   ├── messages.py                 # ACL message builders (CFP/PROPOSE/INFORM/REQUEST)
│   ├── aggregator.py               # Weighted Borda count + weight redistribution
│   ├── history.py                  # Shared round history (fault tolerance + context)
│   ├── run.py                      # RecommendationSystem lifecycle wrapper
│   └── agents/
│       ├── base.py
│       ├── orchestrator.py
│       ├── weight_agent.py
│       ├── body_agent.py
│       ├── clothing_agent.py
│       ├── colour_agent.py
│       └── stock_agent.py
├── prosody_config/                 # Prosody XMPP broker Docker setup
│   ├── Dockerfile
│   └── prosody.cfg.lua
├── scripts/
│   ├── app/
│   │   ├── start_full_app.ps1      # Main launcher (PowerShell)
│   │   └── start_full_app.py       # Main launcher (Python — called by PS1)
│   ├── data_prep/
│   ├── evaluation/
│   └── training/
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI app: endpoints, startup, agent integration
│   │   ├── schemas.py              # Pydantic models
│   │   ├── search_service.py       # Session management
│   │   └── auth.py                 # JWT auth
│   ├── custom_model/               # FashionNet architecture
│   └── detection/
│       ├── detector.py             # YOLOv8 wrapper + CATEGORY_NAMES (0–12)
│       ├── fashionnet_detector.py  # FashionNet inference wrapper
│       └── camera.py               # Multi-frame WebSocket camera session
├── stock_agent/                    # Stock DB + inventory scoring
│   ├── stock_agent.py
│   └── stock_stats.py
├── docker-compose.yml
└── requirements.txt
```

---

## API Routes

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Serve frontend |
| GET | `/health` | Health check |
| POST | `/api/detect/image` | Detect clothing in uploaded image |
| POST | `/api/mobile/scan` | Mobile scan endpoint |
| POST | `/api/recommend` | Run multi-agent recommendation round |
| POST | `/api/session/start` | Start a new search session |
| GET | `/api/session/{session_id}` | Get session state |
| POST | `/api/chat` | Chat-based recommendation refinement |
| POST | `/api/chat/warmup` | Warmup LLM |
| POST | `/api/transcribe` | Transcribe voice input |
| GET | `/api/conf` | Get detection confidence threshold |
| POST | `/api/conf/{value}` | Set detection confidence threshold |
| WS | `/ws/camera` | Live camera WebSocket stream |

`/api/recommend` requires the XMPP broker to be running. Returns `503` if the multi-agent system is unavailable (app continues to serve DB recommendations in that case).

---

## Notes

- Docker Desktop must be running for the XMPP broker. The startup script manages it automatically.
- Model weights are not committed. Without fine-tuned weights the app falls back to base YOLOv8n (COCO), which has incorrect class mappings for fashion — get or train proper weights.
- After installing dependencies, run `python -m spacy download en_core_web_sm` for the semantic search to work.
- Cruella requires Ollama with a compatible model (e.g. `qwen2.5:7b-instruct-q3_K_M`).
- Voice transcription requires `ffmpeg` on PATH.
- See `docs/organized/` for full research documentation.

# Cruzr Robotics Bridge

This folder contains the Edge Computing bridge that connects our multi-agent AI to the Cruzr robot. 
It uses standard HTTP requests to receive commands and MQTT to trigger the robot's hardware.

## Prerequisites (System Requirements)
Before running these scripts, you must have the following installed on your machine:
1. **WSL (Windows Subsystem for Linux):** Running Ubuntu.
2. **Python 3:** Installed inside your WSL environment.
3. **MQTT Explorer (Optional):** For debugging the raw ZBOS JSON payloads.

## Python Dependencies
Install the required Python libraries by running this from the root directory:
`pip install -r requirements.txt`

## How to Run
Open two separate WSL terminals and run:
1. API Bridge: `python src/robotics/api_bridge.py`
2. Camera Streamer: `python src/robotics/camera_streamer.py`