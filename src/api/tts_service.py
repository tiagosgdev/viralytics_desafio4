"""
src/api/tts_service.py
──────────────────────
Text-to-speech for the spoken personas.

Uses Kokoro-82M — a small neural TTS that runs comfortably on CPU, so it does
not compete for VRAM with the GPU models (Ollama / YOLO / Whisper). Each persona
maps to a Kokoro voice id + delivery speed (see ``src/api/personas.py``):

    cruella → af_heart @ 0.9   (warm American female, theatrical)
    edna    → bf_emma  @ 1.1   (crisp British female, clipped)

The Kokoro import is deferred into the functions below so that importing this
module never fails — even before ``pip install kokoro`` has been run. Heavy work
(model load + synthesis) is blocking and is meant to be called from a threadpool.

Available Kokoro voices to swap in (first letter = accent, second = gender):
    American female: af_heart, af_bella, af_nicole, af_sarah, af_sky
    British  female: bf_emma, bf_isabella, bf_alice, bf_lily
"""

from __future__ import annotations

import io
import threading

import numpy as np

from src.api.personas import PERSONA_CONFIGS, normalize_persona

# Kokoro synthesises at 24 kHz.
SAMPLE_RATE = 24000
_REPO_ID = "hexgrad/Kokoro-82M"

# One KPipeline per accent/lang code ('a' = American, 'b' = British). Built
# lazily and cached; protected by a lock because synthesis may be called
# concurrently from the FastAPI threadpool.
_pipelines: dict[str, object] = {}
_pipelines_lock = threading.Lock()

_ready = False
_error: str | None = None


def is_ready() -> bool:
    """True once at least one pipeline has been built and a warm-up synth ran."""
    return _ready


def last_error() -> str | None:
    return _error


def _get_pipeline(lang_code: str):
    """Return (building if needed) the Kokoro pipeline for an accent code."""
    cached = _pipelines.get(lang_code)
    if cached is not None:
        return cached
    with _pipelines_lock:
        cached = _pipelines.get(lang_code)
        if cached is not None:
            return cached
        try:
            from kokoro import KPipeline
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "kokoro is not installed. Run: pip install kokoro soundfile"
            ) from exc
        pipeline = KPipeline(lang_code=lang_code, repo_id=_REPO_ID)
        _pipelines[lang_code] = pipeline
        return pipeline


def _to_wav_bytes(samples: np.ndarray) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def synthesize(text: str, persona: str) -> bytes:
    """Render ``text`` in the persona's voice and return WAV (24 kHz) bytes."""
    cfg = PERSONA_CONFIGS[normalize_persona(persona)]
    voice = cfg.voice_id
    speed = cfg.voice_speed
    lang_code = voice[0]  # 'a' (American) or 'b' (British)

    pipeline = _get_pipeline(lang_code)

    chunks: list[np.ndarray] = []
    for result in pipeline(text, voice=voice, speed=speed):
        # Newer Kokoro yields Result objects with an `.audio` attribute; older
        # versions yield a (graphemes, phonemes, audio) tuple.
        audio = getattr(result, "audio", None)
        if audio is None and isinstance(result, (tuple, list)):
            audio = result[-1]
        if audio is None:
            continue
        if hasattr(audio, "detach"):  # torch tensor
            audio = audio.detach().cpu().numpy()
        chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))

    if not chunks:
        return b""
    return _to_wav_bytes(np.concatenate(chunks))


def preload() -> None:
    """Eagerly build the pipelines and run one tiny synth.

    Called once at startup (in a threadpool). This triggers the first-run model
    download and warms the graph so the first user request isn't slow. Raises on
    failure (e.g. kokoro/espeak-ng missing) — the caller logs it and leaves the
    endpoint returning 503.
    """
    global _ready, _error
    try:
        lang_codes = {cfg.voice_id[0] for cfg in PERSONA_CONFIGS.values()}
        for lang_code in sorted(lang_codes):
            _get_pipeline(lang_code)
        # Warm-up synth for EVERY persona — forces the model weights to download
        # and caches each voice + its G2P backend, so the first real request of
        # each persona is fast. (The first-ever use of a new accent/voice can
        # otherwise take ~15-20s while its pronunciation data is prepared.)
        for persona_key in PERSONA_CONFIGS:
            synthesize("Ready.", persona_key)
        _ready = True
        _error = None
    except Exception as exc:
        _error = str(exc)
        raise
