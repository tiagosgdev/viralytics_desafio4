"""
Body-shape scoring strategies (personalities)
─────────────────────────────────────────────
Pure functions that score candidate items by how well their designed body
shapes match the detected body shape.

IO CONVENTION (important):
  The body shapes for an item live in clothing.db, but **this module never opens
  a database**. The SPADE agent fetches the shapes (via its DB lookup) and
  *enriches each candidate dict* before calling the strategy, attaching the
  per-item shapes under the ``body_shapes`` key as an iterable of lowercased
  shape strings (empty / missing → no body-shape metadata). The strategy reads
  only that key.

Signature shared by every strategy::

    score(candidates, context, weights_result, params) -> {item_key: float in [0,1]}

Buckets (rewards lifted into ``params``):
  * ``no_detect`` — no detected body shape in context (neutral)
  * ``no_data``   — item has no body-shape metadata
  * ``exact``     — detected shape is in the item's shapes
  * ``adjacent``  — item shapes intersect the adjacency set for detected
  * ``no_match``  — none of the above

Personalities:
  * ``strict``          — baseline behaviour (exact 1.0 / adjacent 0.55 / no_data 0.20 / no_match 0.0)
  * ``lenient``         — more generous adjacent + no-data credit
  * ``flattering_only`` — adjacent penalised hard, no-data ~0
"""

from __future__ import annotations

from multi_agent.strategies.constants import BODY_ADJACENT, item_key

# Baseline ("strict") rewards — reproduce the pre-refactor magic numbers exactly.
DEFAULT_PARAMS: dict[str, float] = {
    "no_detect": 0.5,
    "no_data":   0.20,
    "exact":     1.0,
    "adjacent":  0.55,
    "no_match":  0.0,
}


def score(
    candidates: list[dict],
    context: dict,
    weights_result: dict,
    params: dict,
) -> dict[str, float]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    detected = str(context.get("detected_body_type") or "").lower().strip()

    scores: dict[str, float] = {}
    for c in candidates:
        key = item_key(c["item_id"], c["size"])
        if not detected:
            scores[key] = round(p["no_detect"], 6)
            continue
        shapes = frozenset(
            s.strip().lower()
            for s in (c.get("body_shapes") or ())
            if str(s).strip()
        )
        if not shapes:
            raw = p["no_data"]
        elif detected in shapes:
            raw = p["exact"]
        elif shapes & BODY_ADJACENT.get(detected, frozenset()):
            raw = p["adjacent"]
        else:
            raw = p["no_match"]
        scores[key] = round(raw, 6)
    return scores
