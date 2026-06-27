"""
multi_agent/experiments/shopper.py
───────────────────────────────────
The LLM-played customer.

Replaces the human shopper + the browser frontend in an experiment episode. It
reuses Ollama like ``LNIAGIA/query_parsing/feature_weighting.py`` does, but on
its OWN model (``OLLAMA_SHOPPER_MODEL``, default a larger instruct model than the
pipeline's fast refiner) so the simulated human perceives/steers more reliably.
It mirrors that module's robust JSON-extraction so a slightly-off reply parses.

Two public calls:
  * ``next_message(persona, history, last_recs) -> {"message": str, "stop": bool}``
    — the shopper's next chat line, or a signal that it is satisfied / giving up.
  * ``final_review(persona, history, last_recs) -> {"rating": int 1..5, "reason": str}``
    — the 1–5 review that closes the episode (the primary comparison metric).

On any LLM / parse failure both calls fall back to a safe neutral value (a bland
message that keeps the loop alive, or a rating of 3) so an episode never crashes.

``MAX_TURNS`` caps episode length so a never-satisfied shopper still terminates.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

# ── path setup so we can import the shared Ollama config the way the rest of
#    the pipeline does (weight_agent.py / feature_weighting.py replicate this) ──
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LNIAGIA_DIR = _REPO_ROOT / "LNIAGIA"
_QP_DIR = _LNIAGIA_DIR / "query_parsing"
for _p in (_LNIAGIA_DIR, _QP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logger = logging.getLogger(__name__)

# Hard cap on conversation length: even a shopper that never sets stop=True ends.
MAX_TURNS = 6

# Model used to SIMULATE THE HUMAN shopper — deliberately decoupled from the
# pipeline's OLLAMA_REFINER_MODEL (the fast 7b-q3 the weight agent uses). A
# stronger shopper perceives item attributes more reliably and steers/judges
# more realistically (the 7b-q3 hallucinated item details in run 1). Override
# with OLLAMA_SHOPPER_MODEL; defaults to a larger instruct model.
import os

_DEFAULT_SHOPPER_MODEL = "qwen2.5:14b-instruct"
OLLAMA_SHOPPER_MODEL = (
    (os.getenv("OLLAMA_SHOPPER_MODEL") or "").strip() or _DEFAULT_SHOPPER_MODEL
)

# Number of top recommendations shown to the shopper each turn (keeps the prompt
# small for a small instruct model).
_RECS_SHOWN = 5

# Item attributes the shopper actually reasons about. We deliberately hide
# internal scoring fields (agent_scores / agent_weights) — the shopper only sees
# what a real customer would see on a product card.
_VISIBLE_ATTRS = (
    "color", "type", "fit", "style", "pattern", "material",
    "season", "occasion", "brand", "price",
)


# ── lazy imports of the heavy / optional deps ─────────────────────────────────
# Imported inside the call sites (not at module import) so the module stays
# importable for unit tests that never touch a live Ollama.

def _ollama_chat(messages: list[dict]) -> str:
    """Run one ollama.chat turn and return the raw assistant text.

    Mirrors feature_weighting.analyze_intent's call shape (format="json",
    temperature/num_predict/num_ctx options). Raises on any failure; callers
    catch and fall back.
    """
    import ollama  # local import — keeps the module importable without ollama

    response = ollama.chat(
        model=OLLAMA_SHOPPER_MODEL,
        messages=messages,
        format="json",
        keep_alive="30m",
        options={"temperature": 0.7, "num_predict": 256, "num_ctx": 4096},
    )
    raw = ""
    message_obj = getattr(response, "message", None)
    if message_obj is not None:
        raw = str(getattr(message_obj, "content", "")).strip()
    elif isinstance(response, dict):
        raw = str(response.get("message", {}).get("content", "")).strip()
    return raw


def _extract_json(raw: str) -> dict[str, Any]:
    """Best-effort parse of a JSON object out of an LLM reply.

    Mirrors the defensive approach in feature_weighting.analyze_intent: strip a
    ```` ```json ```` fence if present, try a straight ``json.loads``, and as a
    last resort regex out the first ``{...}`` block. Returns ``{}`` if nothing
    parses (callers translate that into their neutral fallback).
    """
    if not raw:
        return {}

    text = raw.strip()
    if text.startswith("```"):
        # drop the opening fence line, then the trailing fence
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        pass

    # Last resort: grab the first balanced-looking {...} span.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


# ── prompt builders ───────────────────────────────────────────────────────────

def _persona_blurb(persona: dict) -> str:
    blurb = (
        f"You are a {persona.get('temperament', 'neutral')} shopper "
        f"(gender: {persona.get('gender', 'unspecified')}). "
        f"Your hidden goal: {persona.get('hidden_goal', 'find something you like')}. "
    )
    tastes = persona.get("tastes")
    if isinstance(tastes, str) and tastes.strip():
        blurb += f"Your personal tastes: {tastes.strip()} "
    mood = persona.get("mood")
    mood = (mood or "").strip() if isinstance(mood, str) else ""
    mood = mood or "neutral"
    blurb += f"Right now you are {mood}. "
    blurb += (
        f"You were scanned wearing a {persona.get('detected_color', '')} "
        f"{persona.get('detected_type', '')}."
    )
    return blurb


def _format_recs(last_recs: list[dict]) -> str:
    if not last_recs:
        return "  (no recommendations yet)"
    lines = []
    for item in last_recs[:_RECS_SHOWN]:
        attrs = ", ".join(
            f"{k}={item.get(k)}" for k in _VISIBLE_ATTRS if item.get(k) not in (None, "")
        )
        lines.append(f"  {item.get('rank', '?')}. {attrs}")
    return "\n".join(lines)


def _format_history(history: list[dict]) -> str:
    if not history:
        return "  (this is the first turn)"
    lines = []
    for turn in history:
        msg = turn.get("shopper_msg", "")
        if msg:
            lines.append(f"  you said: {msg}")
    return "\n".join(lines) if lines else "  (this is the first turn)"


# ── public API ────────────────────────────────────────────────────────────────

def next_message(
    persona: dict,
    history: list[dict],
    last_recs: list[dict],
) -> dict[str, Any]:
    """Produce the shopper's next chat line (or a stop signal).

    Returns ``{"message": str, "stop": bool}``. ``stop=True`` means the shopper
    is satisfied or has given up — the episode ends after this and goes to the
    review. On any LLM / parse failure, returns a neutral non-stopping message
    so the loop continues (and MAX_TURNS still bounds it).

    The persona's fixed ``mood`` and ``tastes`` (read from the persona dict via
    :func:`_persona_blurb`) lightly colour how the shopper steers.
    """
    system_prompt = (
        _persona_blurb(persona)
        + " You are chatting with a clothing-store assistant that keeps showing you "
        "items. Reply with ONE short, natural shopping message that nudges it toward "
        "your hidden goal (mention a colour, style, fit, occasion, etc. you want). "
        "Let your personal tastes (and, lightly, your current mood) colour how you "
        "steer. "
        "If the current items already satisfy your goal, OR you have clearly given up, "
        'set "stop" to true. '
        'Respond ONLY as JSON: {"message": "<your next line>", "stop": <true|false>}.'
    )
    user_prompt = (
        f"Conversation so far:\n{_format_history(history)}\n\n"
        f"Current top recommendations:\n{_format_recs(last_recs)}\n\n"
        "Give your next message as JSON."
    )

    try:
        raw = _ollama_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        parsed = _extract_json(raw)
    except Exception as exc:  # noqa: BLE001 — never let the episode crash
        logger.warning(f"[shopper] next_message LLM failed ({exc}); using neutral fallback.")
        parsed = {}

    message = parsed.get("message")
    if not isinstance(message, str) or not message.strip():
        message = "Can you show me something else?"
    stop = bool(parsed.get("stop", False))
    return {"message": message.strip(), "stop": stop}


def final_review(
    persona: dict,
    history: list[dict],
    last_recs: list[dict],
) -> dict[str, Any]:
    """Produce the closing 1–5 review of the whole episode.

    Returns ``{"rating": int in 1..5, "reason": str}``. On any LLM / parse
    failure (or an out-of-range rating) falls back to a neutral rating of 3.

    The persona's fixed ``mood`` (read from the persona dict via
    :func:`_persona_blurb`) biases how generous or harsh the rating is — a good
    mood makes it more forgiving, a bad mood harsher — while the persona's
    ``tastes`` shape what is liked, but the stated reason must still cite the
    actual recommended item attributes so the review stays diagnostic.
    """
    system_prompt = (
        _persona_blurb(persona)
        + " The shopping session is over. Rate how well the assistant's FINAL "
        "recommendations met your hidden goal, on a 1-5 scale where: "
        "5 = matches most of what you wanted, only minor misses; "
        "4 = matches several key things but has some clear gaps; "
        "3 = matches only a little, but the features it does match are the ones MOST "
        "important to you; "
        "2 = matches only a little, and the features it matches are minor / less "
        "important to you; "
        "1 = nothing relevant. "
        "Judge the FINAL set as a whole — give credit for the best item and for partial "
        "attribute matches; do NOT require a single perfect item. Be honest and "
        "consistent with your "
        "temperament. Let your current MOOD bias how generous or harsh the number "
        "is (a good mood is more forgiving, a bad mood harsher) and let your personal "
        "TASTES shape what you like, but your one-sentence REASON must still cite the "
        "actual attributes (colour, type, fit, style, etc.) of the recommended items "
        "so the review stays diagnostic. "
        'Respond ONLY as JSON: {"rating": <1-5 integer>, "reason": "<one sentence>"}.'
    )
    user_prompt = (
        f"Conversation so far:\n{_format_history(history)}\n\n"
        f"Final recommendations:\n{_format_recs(last_recs)}\n\n"
        "Give your review as JSON."
    )

    try:
        raw = _ollama_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        parsed = _extract_json(raw)
    except Exception as exc:  # noqa: BLE001 — never let the episode crash
        logger.warning(f"[shopper] final_review LLM failed ({exc}); using neutral rating 3.")
        parsed = {}

    rating = _coerce_rating(parsed.get("rating"))
    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "no reason given"
    return {"rating": rating, "reason": reason.strip()}


def _coerce_rating(value: Any) -> int:
    """Clamp an LLM-supplied rating to an integer in [1, 5]; 3 on failure."""
    try:
        rating = int(round(float(value)))
    except (TypeError, ValueError):
        return 3
    return max(1, min(5, rating))
