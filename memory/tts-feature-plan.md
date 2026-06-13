---
name: tts-feature-plan
description: Text-to-speech (spoken personas) feature — implemented, plus the constraint that drove the engine choice
metadata:
  type: project
---

Text-to-speech so the robot speaks its assistant replies out loud, with a distinct feminine voice per persona (Cruella = theatrical/dramatic, Edna = blunt/clipped). **Implemented 2026-06-11.**

Design (confirmed by user): backend generates audio, both clients play it; local/self-hosted; preset voices styled per persona; English only.

**Engine: Kokoro-82M on CPU** — chosen because the machine's GPU is an **RTX 3050 Laptop, 4GB VRAM, already ~maxed by Ollama (qwen2.5:7b) + YOLO/Whisper (~460MB free)**. Rules out Maya1 (16GB), Chatterbox/XTTS on GPU. Kokoro runs on CPU at 1-3x realtime, Apache-2.0. Mirrors the existing Whisper STT at `/api/transcribe`.

What was built:
- `src/api/personas.py` — PersonaConfig gained `voice_id` + `voice_speed`. cruella=`af_heart`@0.9, edna=`bf_emma`@1.1.
- `src/api/tts_service.py` (new) — lazy KPipeline per accent ('a'/'b'), `synthesize(text, persona)->WAV bytes`, `preload()`, `is_ready()`. Kokoro/soundfile imported lazily so the module never breaks app import.
- `src/api/main.py` — `POST /api/tts` (returns audio/wav) + `_load_tts()` startup task. `TtsRequest` in schemas.py.
- `frontend/index.html` — `speakText()` plays /api/tts; 🔊 mute toggle in header (localStorage `fashionSenseTts`); hooked into the 3 persona-reply sites.
- Android `MainActivity.kt` — `speak()`/`playAudio()` via MediaPlayer, called after chat reply; released in onDestroy.

**Setup still required to actually hear it** (verified NOT yet installed on this machine 2026-06-11): `pip install kokoro soundfile` and `winget install eSpeak-NG.eSpeak-NG`. First synth downloads ~300MB Kokoro weights.

**Why CPU/small model:** hard 4GB VRAM limit already consumed by Ollama. **How to apply:** prefer CPU-friendly small models for any new local AI feature here; don't assume GPU headroom. To change a persona's voice, edit `voice_id`/`voice_speed` in personas.py (American `af_*`, British `bf_*`).
