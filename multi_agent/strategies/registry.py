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

# agent_id → strategy_name → (score_fn, default_params)
_REGISTRY: dict[str, dict[str, tuple]] = {
    "colour": {
        # purist == baseline
        "purist": (colour.score, dict(colour.DEFAULT_PARAMS)),
        # harmonizer: compatible variation rewarded highest, exact a touch lower,
        # unrelated kept low.
        "harmonizer": (
            colour.score,
            {"exact": 0.85, "compatible": 1.0, "unrelated": 0.30, "no_detect": 0.5},
        ),
        # adventurous: contrast/variety rewarded — unrelated highest, exact lowest.
        "adventurous": (
            colour.score,
            {"exact": 0.40, "compatible": 0.70, "unrelated": 1.0, "no_detect": 0.5},
        ),
    },
    "body": {
        # strict == baseline
        "strict": (body.score, dict(body.DEFAULT_PARAMS)),
        # lenient: adjacent shapes and no-data items get more credit.
        "lenient": (
            body.score,
            {"no_detect": 0.5, "no_data": 0.50, "exact": 1.0, "adjacent": 0.80, "no_match": 0.20},
        ),
        # flattering_only: only an exact fit is rewarded; adjacent penalised hard,
        # no-data ~0.
        "flattering_only": (
            body.score,
            {"no_detect": 0.5, "no_data": 0.05, "exact": 1.0, "adjacent": 0.15, "no_match": 0.0},
        ),
    },
    "clothing": {
        # match_count == baseline
        "match_count": (clothing.score_match_count, dict(clothing.DEFAULT_PARAMS)),
        # weighted_axes: type/style/occasion weighted more than other axes.
        "weighted_axes": (
            clothing.score_weighted_axes,
            copy.deepcopy(clothing.WEIGHTED_AXES_PARAMS),
        ),
        # strict_type: the `type` axis dominates the score.
        "strict_type": (
            clothing.score_strict_type,
            dict(clothing.STRICT_TYPE_PARAMS),
        ),
    },
    "stock": {
        # push == baseline
        "push": (stock.score, dict(stock.DEFAULT_PARAMS)),
        # overstock_aggressive: favour high stock_count (clear overstock).
        "overstock_aggressive": (stock.score, dict(stock.OVERSTOCK_PARAMS)),
        # bestsellers: favour sales_velocity (proven sellers).
        "bestsellers": (stock.score, dict(stock.BESTSELLERS_PARAMS)),
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
