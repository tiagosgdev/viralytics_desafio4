"""
Clothing-intent scoring strategies (personalities)
──────────────────────────────────────────────────
Pure functions that score candidate items by how many of the DB include-filters
(from feature_weighting.analyze_intent, carried in ``weights_result``) each item
satisfies. No IO: candidates already carry the attribute fields being matched.

Signature shared by every strategy::

    score(candidates, context, weights_result, params) -> {item_key: float in [0,1]}

Matching rules (shared by all personalities):
  * the ``age_group`` axis matches by case-insensitive substring
  * every other axis matches by exact membership in its values list
  * ``body_type`` is skipped (handled by the body agent's DB lookup)
  * if there are no include axes → uniform ``no_filters`` score

Personalities:
  * ``match_count``   — baseline: score = hits / n_axes
  * ``weighted_axes`` — per-axis weights (``axis_weights`` in params); score is the
                        weighted fraction of satisfied axes (axes default to weight 1.0)
  * ``strict_type``   — the ``type`` axis dominates: matching type alone scores high,
                        other axes contribute only a small bonus
"""

from __future__ import annotations

from multi_agent.strategies.constants import item_key

# Axes this agent does not evaluate (body_type → BodyRecommenderAgent).
SKIP_AXES: frozenset[str] = frozenset({"body_type"})

# Baseline params.
DEFAULT_PARAMS: dict = {
    "no_filters": 0.5,
}

# weighted_axes default per-axis weights (axes not listed default to 1.0).
WEIGHTED_AXES_PARAMS: dict = {
    "no_filters": 0.5,
    "axis_weights": {
        "type":       2.0,
        "style":      1.5,
        "occasion":   1.5,
        "season":     1.0,
        "age_group":  1.0,
        "gender":     1.0,
        "color":      1.0,
    },
    "default_weight": 1.0,
}

# strict_type params: type match alone gets most of the credit; the remaining
# axes share a small bonus.
STRICT_TYPE_PARAMS: dict = {
    "no_filters":  0.5,
    "type_weight": 0.7,   # credit for matching the `type` axis
    "other_bonus": 0.3,   # total credit shared by all non-type axes
}


def _active_include(weights_result: dict) -> dict[str, list[str]]:
    raw_include = (weights_result.get("filters") or {}).get("include") or {}
    return {k: v for k, v in raw_include.items() if k not in SKIP_AXES}


def _axis_hit(c: dict, key: str, values: list[str]) -> bool:
    item_val = str(c.get(key) or "")
    if key == "age_group":
        return any(str(v).lower() in item_val.lower() for v in values)
    return item_val in values


def score_match_count(
    candidates: list[dict],
    context: dict,
    weights_result: dict,
    params: dict,
) -> dict[str, float]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    include = _active_include(weights_result)
    if not include:
        return {item_key(c["item_id"], c["size"]): p["no_filters"] for c in candidates}

    n_axes = len(include)
    scores: dict[str, float] = {}
    for c in candidates:
        hits = sum(1 for key, values in include.items() if _axis_hit(c, key, values))
        scores[item_key(c["item_id"], c["size"])] = round(hits / n_axes, 6)
    return scores


def score_weighted_axes(
    candidates: list[dict],
    context: dict,
    weights_result: dict,
    params: dict,
) -> dict[str, float]:
    p = {**WEIGHTED_AXES_PARAMS, **(params or {})}
    include = _active_include(weights_result)
    if not include:
        return {item_key(c["item_id"], c["size"]): p["no_filters"] for c in candidates}

    axis_weights: dict = p["axis_weights"]
    default_w: float = p["default_weight"]
    total_w = sum(axis_weights.get(k, default_w) for k in include) or 1.0

    scores: dict[str, float] = {}
    for c in candidates:
        hit_w = sum(
            axis_weights.get(key, default_w)
            for key, values in include.items()
            if _axis_hit(c, key, values)
        )
        scores[item_key(c["item_id"], c["size"])] = round(hit_w / total_w, 6)
    return scores


def score_strict_type(
    candidates: list[dict],
    context: dict,
    weights_result: dict,
    params: dict,
) -> dict[str, float]:
    p = {**STRICT_TYPE_PARAMS, **(params or {})}
    include = _active_include(weights_result)
    if not include:
        return {item_key(c["item_id"], c["size"]): p["no_filters"] for c in candidates}

    other_axes = [k for k in include if k != "type"]
    has_type = "type" in include
    type_w: float = p["type_weight"]
    other_bonus: float = p["other_bonus"]

    scores: dict[str, float] = {}
    for c in candidates:
        total = 0.0
        if has_type and _axis_hit(c, "type", include["type"]):
            total += type_w
        if other_axes:
            other_hits = sum(1 for k in other_axes if _axis_hit(c, k, include[k]))
            total += other_bonus * (other_hits / len(other_axes))
        # When `type` isn't an active axis, fall back to a plain weighted fraction
        # so the score still spans [0, 1] (other axes carry the full budget).
        if not has_type and other_axes:
            other_hits = sum(1 for k in other_axes if _axis_hit(c, k, include[k]))
            total = other_hits / len(other_axes)
        scores[item_key(c["item_id"], c["size"])] = round(min(total, 1.0), 6)
    return scores
