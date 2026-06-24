---
name: spade-env-setup
description: Env/run/test setup and gotchas for the SPADE multi-agent system (desafio4)
metadata: 
  node_type: memory
  type: reference
  originSessionId: f5c32068-8876-46a2-a96b-d678ae671701
---

Running/testing the SPADE system (see [[spade-experiments-state]]):

- **Python:** use **system** `python3` (3.13.4 at `/Library/Frameworks/Python.framework/...`).
  `spade` **4.1.4** is installed there (project pins `>=3.2.3`; 4.x works — imports fine).
  pandas/torch/spacy/ollama present.
- **Do NOT run `pip install -r requirements.txt`** — it pins `numpy>=1.26.4,<2` which has no
  wheel on Python 3.13 and fails to build from source (libc++ `type_traits` error). System
  python already has numpy 2.3.3 and everything needed. Install single packages individually
  if ever required. (2026-06-24: confirmed the numpy<2 pin is **stale** — `mediapipe 0.10.35`
  installs fine against numpy 2.3.3; the requirements pin should be relaxed to numpy>=2 / unpinned,
  but that's a team-affecting config change, not yet done.)
- **Pose analyzer / body-shape (fixed 2026-06-24):** startup logged *"Pose analyzer unavailable;
  body silhouette analysis will be skipped"* because `PoseAnalyzer.is_available()`
  (`src/pose_analyzer/pose_analyzer.py:87`) needs BOTH the `mediapipe` pkg AND the model file.
  Fix: `pip3 install 'mediapipe>=0.10.14'` (pulls mediapipe+sounddevice+opencv-contrib-python,
  numpy untouched) **and** download `pose_landmarker_heavy.task` to
  `models/weights/mediapipe/` (gitignored; `curl` from
  `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task`,
  or use the team SharePoint weights bundle from README.md:41). Then is_available()=True.
- **Still-missing model weights (NOT fixed):** startup also shows *"No fine-tuned weights found,
  using base yolov8n.pt"* and *"No FashionNet checkpoint found"* — these (the fine-tuned garment
  detector + FashionNet) live in the team's **SharePoint weights bundle** (README.md:41), not in
  git (`models/weights/*` is gitignored). Their absence is why live detection degrades (wrong
  colour naming, pants undetected). Download the bundle into `models/weights/` to restore. See
  [[recommender-architecture]] for how detection feeds the agents.
- **XMPP broker:** `docker compose up -d xmpp` (Prosody, port 5222). Needed for any live round
  (`python -m multi_agent.run` or the smoke). Docker 29.x available.
- **Tests (no broker needed):**
  `python3 -m pytest tests/test_agent_weights.py tests/test_strategies.py tests/test_retrieval.py tests/test_agent_memory.py tests/test_rl_policy.py -q`
  → 65 pass at last checkpoint. **`tests/test_pose_analyzer.py` has 3 PRE-EXISTING unrelated
  failures — ignore them.**
- **Live smoke pattern that works** (verified): start `RecommendationSystem`, call
  `recommend(detected_color=..., detected_type=..., detected_body_type=..., user_gender=...,
  detected_body_type_conf=...)` on the **empty-`user_answer` fast path** (no Ollama needed),
  inspect `results[0]["agent_weights"]`. Confirmed body_conf=0.3 drops body weight 0.21→0.08,
  weights sum to 1.0, RL stays 0.15. Restore `multi_agent/history.db` after (`git checkout --`)
  since live rounds mutate that tracked file; per-agent `multi_agent/memory/*.db` are gitignored.
- `macOS` has no `timeout` command; run scripts directly (the smoke self-terminates).
