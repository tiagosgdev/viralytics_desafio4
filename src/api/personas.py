from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PersonaConfig:
    key: str
    label: str
    vision_backend: str
    text_backend: str
    # Text-to-speech voice (Kokoro voice id + delivery speed). The first letter
    # of the voice id is the accent/lang code: 'a' = American, 'b' = British.
    voice_id: str
    voice_speed: float


PERSONA_CONFIGS = {
    "cruella": PersonaConfig(
        key="cruella",
        label="Cruella",
        vision_backend="yolo",
        text_backend="llm",
        # Warm, expressive American female, slowed for theatrical delivery.
        voice_id="af_heart",
        voice_speed=0.9,
    ),
    "edna": PersonaConfig(
        key="edna",
        label="Edna",
        vision_backend="fashionnet",
        text_backend="custom",
        # Crisp British female, sped up for clipped, no-nonsense cadence.
        voice_id="bf_emma",
        voice_speed=1.1,
    ),
}


def normalize_persona(value: Optional[str]) -> str:
    if not value:
        return "cruella"
    key = str(value).strip().lower()
    return key if key in PERSONA_CONFIGS else "cruella"
