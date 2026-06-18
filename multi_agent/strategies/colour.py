"""
Colour scoring strategies (personalities)
─────────────────────────────────────────
Pure functions that score candidate items by colour relationship to the
detected / weight-refined garment colour. No IO: everything the strategy needs
is already on the candidate dicts (the ``color`` field) and in ``context`` /
``weights_result``.

Signature shared by every strategy::

    score(candidates, context, weights_result, params) -> {item_key: float in [0,1]}

where ``item_key = f"{item_id}:{size}"``.

The detected colour is resolved exactly as the live agent did before this
refactor: prefer ``weights_result.filters.include.color[0]`` if present, else
``context["detected_color"]`` (lowercased / stripped).

Three buckets, three rewards (lifted into ``params``):
  * ``exact``      — item colour equals the detected colour
  * ``compatible`` — item colour is in the compatibility matrix for detected
  * ``unrelated``  — neither of the above
  * ``no_detect``  — no detected colour at all (neutral fallback)

Personalities differ only in how they reward those buckets:
  * ``purist``      — exact highest (= baseline behaviour)
  * ``harmonizer``  — tasteful variation: compatible highest, exact a touch lower
  * ``adventurous`` — contrast/variety: unrelated highest
"""

from __future__ import annotations

from multi_agent.strategies.constants import COLOUR_COMPATIBLE, item_key

# Baseline ("purist") rewards — reproduce the pre-refactor magic numbers exactly.
DEFAULT_PARAMS: dict[str, float] = {
    "exact":      1.0,
    "compatible": 0.65,
    "unrelated":  0.20,
    "no_detect":  0.5,
}


def _resolve_detected(context: dict, weights_result: dict) -> str:
    w_include   = ((weights_result.get("filters") or {}).get("include") or {})
    colour_vals = w_include.get("color") or []
    if colour_vals:
        return str(colour_vals[0])
    return str(context.get("detected_color") or "").lower().strip()


def score(
    candidates: list[dict],
    context: dict,
    weights_result: dict,
    params: dict,
) -> dict[str, float]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    detected = _resolve_detected(context, weights_result)

    scores: dict[str, float] = {}
    for c in candidates:
        key = item_key(c["item_id"], c["size"])
        if not detected:
            scores[key] = p["no_detect"]
            continue
        det = detected.lower().strip()
        item_color = (c.get("color") or "").lower().strip()
        if item_color == det:
            scores[key] = p["exact"]
        elif item_color in COLOUR_COMPATIBLE.get(det, frozenset()):
            scores[key] = p["compatible"]
        else:
            scores[key] = p["unrelated"]
    return scores
