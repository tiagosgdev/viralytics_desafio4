"""
Strategy registry
──────────────────
Maps ``agent_id → {strategy_name: (score_fn, default_params)}`` and exposes a
resolver ``get_strategy(agent_id, name) -> (fn, params)``.

Each registered strategy is a pure function with the shared signature::

    score(candidates, context, weights_result, params) -> {item_key: float in [0,1]}

The ``default_params`` returned by the resolver are a *copy*, so callers can
mutate them freely without affecting the registry.
"""

from __future__ import annotations

import copy

from multi_agent.strategies import body, clothing, colour, stock


def _with_veto(params: dict, veto_threshold: float) -> dict:
    """Return a copy of `params` carrying a `veto_threshold` (veto strictness).

    An agent vetoes any item whose score is strictly below `veto_threshold`
    (see the scorer agents). The score functions themselves are unchanged — the
    veto is derived from the existing per-item score. A threshold of -1 (the
    safe default used when a strategy omits the key) vetoes nothing.
    """
    out = dict(params)
    out.setdefault("veto_threshold", veto_threshold)
    return out


# Per-strategy veto strictness. Higher = the agent eliminates more items.
# Personality maps directly to how aggressively the dimension prunes the slate.
#
#   colour  : purist vetoes anything not exact/compatible; harmonizer allows the
#             complementary palette; adventurous vetoes nothing.
#   body    : strict/flattering veto poor body-shape fits; lenient barely vetoes.
#   clothing: vetoes items that satisfy (almost) none of the requested axes;
#             strict_type prunes hardest, match_count mildly.
#   stock   : a store-push preference, not a relevance gate → never vetoes
#             (threshold -1) so the inventory signal only ever *ranks*.

# agent_id → strategy_name → (score_fn, default_params)
_REGISTRY: dict[str, dict[str, tuple]] = {
    "colour": {
        # purist == baseline. Vetoes unrelated (0.20) and compatible (0.65).
        "purist": (colour.score, _with_veto(dict(colour.DEFAULT_PARAMS), 0.7)),
        # harmonizer: compatible variation rewarded highest, exact a touch lower,
        # unrelated kept low. Vetoes only unrelated (0.30).
        "harmonizer": (
            colour.score,
            _with_veto(
                {"exact": 0.85, "compatible": 1.0, "unrelated": 0.30, "no_detect": 0.5},
                0.4,
            ),
        ),
        # adventurous: contrast/variety rewarded — unrelated highest, exact lowest.
        # Vetoes nothing (every colour is fair game).
        "adventurous": (
            colour.score,
            _with_veto(
                {"exact": 0.40, "compatible": 0.70, "unrelated": 1.0, "no_detect": 0.5},
                0.0,
            ),
        ),
    },
    "body": {
        # strict == baseline. Vetoes no_data (0.20) and no_match (0.0).
        "strict": (body.score, _with_veto(dict(body.DEFAULT_PARAMS), 0.5)),
        # lenient: adjacent shapes and no-data items get more credit. Barely vetoes.
        "lenient": (
            body.score,
            _with_veto(
                {"no_detect": 0.5, "no_data": 0.50, "exact": 1.0, "adjacent": 0.80, "no_match": 0.20},
                0.1,
            ),
        ),
        # flattering_only: only an exact fit is rewarded; adjacent penalised hard,
        # no-data ~0. Vetoes adjacent (0.15), no_data (0.05) and no_match.
        "flattering_only": (
            body.score,
            _with_veto(
                {"no_detect": 0.5, "no_data": 0.05, "exact": 1.0, "adjacent": 0.15, "no_match": 0.0},
                0.5,
            ),
        ),
    },
    "clothing": {
        # match_count == baseline. Vetoes items matching no requested axis (score 0).
        "match_count": (
            clothing.score_match_count,
            _with_veto(dict(clothing.DEFAULT_PARAMS), 0.01),
        ),
        # weighted_axes: type/style/occasion weighted more than other axes.
        "weighted_axes": (
            clothing.score_weighted_axes,
            _with_veto(copy.deepcopy(clothing.WEIGHTED_AXES_PARAMS), 0.01),
        ),
        # strict_type: the `type` axis dominates the score → prune hardest, so an
        # item that misses the type axis (score ≤ other_bonus ≈ 0.3) is vetoed.
        "strict_type": (
            clothing.score_strict_type,
            _with_veto(dict(clothing.STRICT_TYPE_PARAMS), 0.5),
        ),
    },
    "stock": {
        # push == baseline. Stock is a ranking preference, not a gate → never vetoes.
        "push": (stock.score, _with_veto(dict(stock.DEFAULT_PARAMS), -1.0)),
        # overstock_aggressive: favour high stock_count (clear overstock).
        "overstock_aggressive": (
            stock.score, _with_veto(dict(stock.OVERSTOCK_PARAMS), -1.0)
        ),
        # bestsellers: favour sales_velocity (proven sellers).
        "bestsellers": (
            stock.score, _with_veto(dict(stock.BESTSELLERS_PARAMS), -1.0)
        ),
    },
}


def strategy_names(agent_id: str) -> list[str]:
    """Return the registered strategy names for ``agent_id``."""
    return list(_REGISTRY.get(agent_id, {}).keys())


def get_strategy(agent_id: str, name: str) -> tuple:
    """
    Resolve ``(score_fn, default_params)`` for ``agent_id`` / strategy ``name``.

    Raises KeyError with a helpful message if either is unknown. The returned
    params are a deep copy, safe for the caller to mutate.
    """
    agent_strats = _REGISTRY.get(agent_id)
    if agent_strats is None:
        raise KeyError(
            f"unknown agent_id {agent_id!r}; known: {sorted(_REGISTRY)}"
        )
    entry = agent_strats.get(name)
    if entry is None:
        raise KeyError(
            f"unknown strategy {name!r} for agent {agent_id!r}; "
            f"known: {sorted(agent_strats)}"
        )
    fn, default_params = entry
    return fn, copy.deepcopy(default_params)
