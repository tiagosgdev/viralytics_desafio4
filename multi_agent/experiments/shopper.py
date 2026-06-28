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
  * ``final_review(persona, history, last_recs) -> {"ratings": {(item_id, size):
    int 1..5}, "aggregate": int 1..5, "reason": str}`` — the closing review. It
    now rates EACH final item individually (so the RL reward path gets real
    within-round contrast, matching production's per-item feedback) and also
    exposes a backward-compatible episode-level ``aggregate`` (the rounded mean of
    the per-item ratings) that the episode metric / learning-curve point uses.

On any LLM / parse failure both calls fall back to a safe neutral value (a bland
message that keeps the loop alive, or a neutral rating of 3) so an episode never
crashes.

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

from multi_agent import config

_DEFAULT_SHOPPER_MODEL = "qwen2.5:14b-instruct"
OLLAMA_SHOPPER_MODEL = (
    (os.getenv("OLLAMA_SHOPPER_MODEL") or "").strip() or _DEFAULT_SHOPPER_MODEL
)

# Number of top recommendations shown to the shopper during MID-CONVERSATION
# turns (keeps the steering prompt small for a small instruct model).
_RECS_SHOWN = 5

# Number shown for the FINAL review. The shopper must rate ALL final items so the
# RL reward path gets per-item contrast, so this surfaces the full TOP_K set
# (10) rather than just the top few. Kept separate from ``_RECS_SHOWN`` so turn
# behaviour is unchanged.
_FINAL_RECS_SHOWN = config.TOP_K

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


def _format_recs(last_recs: list[dict], limit: int = _RECS_SHOWN) -> str:
    """Render up to ``limit`` recs as a numbered list.

    The leading number is a STABLE 1-based rank derived from the item's position
    in ``last_recs`` (NOT the item's own ``rank`` field), so the final-review
    parser can map a returned rank straight back to ``last_recs[rank - 1]``.
    """
    if not last_recs:
        return "  (no recommendations yet)"
    lines = []
    for idx, item in enumerate(last_recs[:limit], start=1):
        attrs = ", ".join(
            f"{k}={item.get(k)}" for k in _VISIBLE_ATTRS if item.get(k) not in (None, "")
        )
        lines.append(f"  {idx}. {attrs}")
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
    """Produce the closing per-item review of the episode's FINAL recommendations.

    Returns ``{"ratings": {(item_id, size): int 1..5}, "aggregate": int 1..5,
    "reason": str}``:
      * ``ratings`` maps EACH final item (the first ``_FINAL_RECS_SHOWN`` of
        ``last_recs``) to its own 1–5 rating, so the harness can feed each item's
        own reward into the RL path — mirroring production's per-item feedback and
        giving the agent within-round contrast.
      * ``aggregate`` is the rounded mean of those per-item ratings (neutral 3 if
        nothing parsed), kept for backward compatibility as the episode-level
        scalar metric / learning-curve point.
      * ``reason`` is a one-sentence diagnostic justification.

    On any LLM / parse failure (or out-of-range / partial / malformed ratings)
    this degrades gracefully: each rating is clamped to 1..5, any item the model
    forgot to score defaults to the aggregate of the ones it did score (or a
    neutral 3 if it scored none), and the episode never crashes.

    The persona's fixed ``mood`` (read from the persona dict via
    :func:`_persona_blurb`) biases how generous or harsh the ratings are — a good
    mood makes it more forgiving, a bad mood harsher — while the persona's
    ``tastes`` shape what is liked, but the stated reason must still cite the
    actual recommended item attributes so the review stays diagnostic.
    """
    final_recs = last_recs[:_FINAL_RECS_SHOWN]
    system_prompt = (
        _persona_blurb(persona)
        + " The shopping session is over. Rate how well EACH of the assistant's "
        "FINAL recommendations met your hidden goal, scoring every listed item "
        "individually on a 1-5 scale where: "
        "5 = matches most of what you wanted, only minor misses; "
        "4 = matches several key things but has some clear gaps; "
        "3 = matches only a little, but the features it does match are the ones MOST "
        "important to you; "
        "2 = matches only a little, and the features it matches are minor / less "
        "important to you; "
        "1 = nothing relevant. "
        "Judge each item on its OWN merits — different items may deserve very "
        "different scores; give credit for partial attribute matches and do NOT "
        "require a perfect item. Be honest and consistent with your temperament. "
        "Let your current MOOD bias how generous or harsh the numbers are (a good "
        "mood is more forgiving, a bad mood harsher) and let your personal TASTES "
        "shape what you like, but your one-sentence REASON must still cite the "
        "actual attributes (colour, type, fit, style, etc.) of the recommended items "
        "so the review stays diagnostic. "
        'Use each item\'s leading number as its "rank". '
        'Respond ONLY as JSON: '
        '{"ratings": [{"rank": <item number>, "rating": <1-5 integer>}, ...], '
        '"reason": "<one sentence>"}. Include one entry per item shown.'
    )
    user_prompt = (
        f"Conversation so far:\n{_format_history(history)}\n\n"
        f"Final recommendations:\n{_format_recs(last_recs, _FINAL_RECS_SHOWN)}\n\n"
        "Give your per-item review as JSON."
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
        logger.warning(f"[shopper] final_review LLM failed ({exc}); using neutral ratings.")
        parsed = {}

    ratings, aggregate = _coerce_item_ratings(parsed.get("ratings"), final_recs)
    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "no reason given"
    return {"ratings": ratings, "aggregate": aggregate, "reason": reason.strip()}


def _coerce_rating(value: Any) -> int:
    """Clamp an LLM-supplied rating to an integer in [1, 5]; 3 on failure."""
    try:
        rating = int(round(float(value)))
    except (TypeError, ValueError):
        return 3
    return max(1, min(5, rating))


def _item_key(item: dict) -> tuple[int, str] | None:
    """Build the ``(item_id, size)`` reward key for a rec; None if unbuildable.

    Matches the key the harness uses when calling ``submit_feedback`` so the
    per-item ratings line up with the production reward path.
    """
    try:
        return (int(item["item_id"]), str(item["size"]))
    except (KeyError, TypeError, ValueError):
        return None


def _coerce_item_ratings(
    raw: Any,
    final_recs: list[dict],
) -> tuple[dict[tuple[int, str], int], int]:
    """Map a model's per-item rating list back onto ``final_recs`` by rank.

    ``raw`` is expected to be a list of ``{"rank": int, "rating": int}`` (the
    rank being the 1-based position shown to the shopper). Returns
    ``(ratings_by_key, aggregate)`` where:
      * ``ratings_by_key`` maps each rec's ``(item_id, size)`` to a clamped 1..5
        rating. Items the model did not score (or that arrived malformed) default
        to ``aggregate``.
      * ``aggregate`` is the rounded mean of the ratings the model actually
        supplied (each clamped to 1..5), or a neutral ``3`` if it supplied none.

    Fully defensive: a non-list ``raw``, non-dict entries, missing/garbage ranks
    or ratings, and duplicate ranks are all tolerated without raising.
    """
    by_rank: dict[int, int] = {}
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            try:
                rank = int(entry.get("rank"))
            except (TypeError, ValueError):
                continue
            if 1 <= rank <= len(final_recs):
                by_rank[rank] = _coerce_rating(entry.get("rating"))

    if by_rank:
        aggregate = max(1, min(5, int(round(sum(by_rank.values()) / len(by_rank)))))
    else:
        aggregate = 3

    ratings_by_key: dict[tuple[int, str], int] = {}
    for idx, item in enumerate(final_recs, start=1):
        key = _item_key(item)
        if key is None:
            continue
        ratings_by_key[key] = by_rank.get(idx, aggregate)
    return ratings_by_key, aggregate
