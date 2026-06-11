# FashionSense
> Master's Project | Computer Vision + Semantic Fashion Search

FashionSense is a fashion outfit detection and recommendation system with two runtime personas:

- **Cruella** — trained YOLO-based outfit detection + LLM-powered semantic search
- **Edna** — custom FashionNet (edna) outfit detection + local text parsing

The user selects a persona on the landing screen, scans an outfit via live camera, receives store recommendations, and can refine results through chat or voice.

---

## Model Weights

Pre-trained edna model weights are too large for the repository. Download from SharePoint:

**[Download model weights](https://myisepipp-my.sharepoint.com/:u:/g/personal/1140331_isep_ipp_pt/IQCCukBlXdvVRoOwaETcMQwwAbNnmYm78sjY0hEas7aCeS4?e=1mLUmX)**

Extract and place under `models/weights/` so the structure matches:

```
models/weights/
└── edna_1.5m/
    └── best.pt
```

Then run the app with `--edna-weights edna_1.5m` (see Running the App below).

---

## Requirements

- Python 3.10+
- CUDA GPU recommended for training (CPU and Apple MPS supported)
- `ffmpeg` installed for voice transcription
- Ollama running locally for Cruella's LLM text backend

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Datasets

### DeepFashion2 (required for training)

Dataset source: [Kaggle — DeepFashion2 Original with Dataframes](https://www.kaggle.com/datasets/thusharanair/deepfashion2-original-with-dataframes?resource=download)

A Kaggle account is required. Download and extract to `data/raw/` with this structure:

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

Once raw data is in place, run:

```bash
python scripts/data_prep/sample_balanced.py \
    --train_csv data/raw/DeepFashion2/img_info_dataframes/train.csv \
    --val_csv   data/raw/DeepFashion2/img_info_dataframes/validation.csv \
    --img_dirs  data/raw/train/image data/raw/validation/image \
    --output_dir data/balanced_dataset \
    --n_per_class 7641 \
    --seed 42
```

This produces an 84,051-image balanced dataset across 11 classes (70/15/15 split) at `data/balanced_dataset/`.

### Background images (required for edna_1.4m+)

Download 2,000 COCO val2017 background images (no people or clothing) used as negative training examples:

```bash
python scripts/data/download_bg_images.py
```

Then add them to the training split:

```bash
cp bg_images/*.jpg data/balanced_dataset/images/train/

for f in bg_images/*.jpg; do
    touch data/balanced_dataset/labels/train/$(basename $f .jpg).txt
done
```

---

## Running the App

The launcher checks all dependencies (Ollama, vector DB, imports) before starting uvicorn.

**Standard launch (edna 1.5m weights):**

```bash
python3 scripts/app/start_full_app.py --edna-weights edna_1.5m --auto-pull-model
```

**With auto-reload (development):**

```bash
python3 scripts/app/start_full_app.py --edna-weights edna_1.5m --auto-pull-model --reload
```

**LAN / mobile testing:**

```bash
python3 scripts/app/start_full_app.py --edna-weights edna_1.5m --auto-pull-model --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in a browser (or the machine's LAN IP for mobile).

**Key flags:**

| Flag | Description |
|------|-------------|
| `--edna-weights <folder>` | Weights folder under `models/weights/` for Edna (e.g. `edna_1.5m`) |
| `--auto-pull-model` | Automatically pull the required Ollama model (`qwen2.5:3b-instruct`) if missing |
| `--reload` | Enable uvicorn auto-reload (development only) |
| `--host <ip>` | Bind host — use `0.0.0.0` for LAN access (default: `127.0.0.1`) |
| `--port <n>` | Bind port (default: `8000`) |
| `--skip-ollama` | Skip Ollama checks (use if running Edna-only without Cruella) |
| `--skip-vector-check` | Skip vector DB check (use if LNIAGIA collection not yet built) |

**Android app:**

1. Open `android_app/` in Android Studio
2. Set the server IP in the app to your machine's LAN IP
3. `Build > Build Bundle(s) / APK(s) > Build APK(s)`

---

## Training

Train the custom FashionNet (edna) model:

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
    --gr 0.0 \
    --augment medium \
    --multi_cell \
    --optimizer adamw \
    --weight_decay 0.01 \
    --device cuda \
    --output models/weights/fashionnet

# CPU (slow — use for testing only)
python scripts/training/train_custom.py --epochs 10 --batch 4 --device cpu

# Apple Silicon
python scripts/training/train_custom.py --epochs 50 --batch 16 --device mps
```

Checkpoints are saved to `models/weights/fashionnet/`. The best validation checkpoint is saved as `best.pt`.

---

## Evaluation

Evaluate a trained model on the balanced dataset:

```bash
python scripts/evaluation/evaluate_custom.py \
    --weights models/weights/fashionnet/best.pt \
    --data data/balanced_dataset \
    --conf 0.25
```

Generate training plots:

```bash
python scripts/evaluation/visualize_results.py \
    --metrics_json models/weights/fashionnet/metrics.json \
    --output docs/organized/04_edna/plots/
```

---

## Dataset Analysis

Generate EDA figures for the raw dataset:

```bash
python scripts/data_prep/analyze_raw_dataset.py \
    --csv data/raw/DeepFashion2/img_info_dataframes/train.csv \
    --output docs/organized/01_dataset/raw_dataset/
```

---

## Project Structure

```
FashionSense/
├── android_app/                    # Native Android client
├── data/
│   ├── mock_store_catalogue_template.json
│   └── raw/                        # DeepFashion2 raw data (not committed)
├── docs/
│   ├── README.md
│   └── organized/                  # Research documentation
│       ├── 01_dataset/
│       ├── 02_yolo_experiments/
│       ├── 03_fashionnet_experiments/
│       ├── 04_edna/
│       ├── 05_evaluation/
│       └── 06_codebase/
├── frontend/
│   ├── index.html
│   └── static/css/style.css
├── LNIAGIA/                        # Semantic search subsystem (Cruella)
├── models/
│   └── weights/                    # Trained model weights (not committed)
│       └── fashionnet/
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_model_comparison.ipynb
├── scripts/
│   ├── app/
│   ├── data/
│   ├── data_prep/
│   ├── evaluation/
│   └── training/
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── search_service.py
│   │   ├── personas.py
│   │   └── custom_text_parser.py
│   ├── custom_model/               # FashionNet architecture
│   ├── detection/
│   │   ├── detector.py
│   │   ├── fashionnet_detector.py
│   │   └── camera.py
│   └── recommendations/
│       ├── engine.py
│       └── catalogue.py
├── requirements.txt
└── docker-compose.yml
```

---

## API Routes

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Serve frontend |
| GET | `/health` | Health check |
| POST | `/api/detect/image` | Detect clothing in uploaded image |
| POST | `/api/mobile/scan` | Mobile scan endpoint |
| POST | `/api/session/start` | Start a new session |
| GET | `/api/session/{session_id}` | Get session state |
| POST | `/api/chat` | Chat refinement |
| POST | `/api/chat/warmup` | Warmup LLM |
| POST | `/api/transcribe` | Transcribe voice input |
| GET | `/api/conf` | Get detection confidence threshold |
| POST | `/api/conf/{value}` | Set detection confidence threshold |
| WS | `/ws/camera` | Live camera WebSocket stream |

All scan/chat/session requests carry a `persona` field (`cruella` or `edna`).

---

## Notes

- Model weights are not committed. Place trained weights at `models/weights/fashionnet/best.pt` for Edna to load automatically.
- Cruella requires Ollama running locally with a compatible model pulled.
- Voice transcription requires `ffmpeg` on PATH and `faster-whisper` installed.
- The store catalogue at `data/mock_store_catalogue_template.json` can be replaced with real store data.
- See `docs/organized/` for full research documentation including dataset analysis, experiment results, and architecture details.

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