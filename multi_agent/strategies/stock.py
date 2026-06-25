"""
Stock / inventory scoring strategies (personalities)
────────────────────────────────────────────────────
Pure functions that score candidate items by an inventory signal, normalised to
[0, 1] across the candidate set.

IO CONVENTION (important):
  These strategies never touch ``StockStats`` or any DB. The SPADE agent does the
  ``StockStats`` lookups and *enriches each candidate dict* with the raw signals
  before calling the strategy:
    * ``push_score``     — raw push_score for (item_id, size); default 0.0 if missing
    * ``stock_count``    — current units in stock (already on candidate dicts upstream)
    * ``sales_velocity`` — units sold per day (fetched from StockStats.get_row)
  The strategy reads only those keys and min-max normalises the chosen signal
  across the candidate set, so the output is comparable to the baseline.

Signature shared by every strategy::

    score(candidates, context, weights_result, params) -> {item_key: float in [0,1]}

Personalities:
  * ``push``                 — baseline: normalised push_score (store-push signal)
  * ``overstock_aggressive`` — favour high ``stock_count`` (clear overstock)
  * ``bestsellers``          — favour ``sales_velocity`` (proven sellers)
"""

from __future__ import annotations

from multi_agent.strategies.constants import item_key

DEFAULT_PARAMS: dict = {
    "signal":  "push_score",   # which enriched field to rank on
    "missing": 0.0,            # value substituted when the field is absent
}

OVERSTOCK_PARAMS: dict = {"signal": "stock_count", "missing": 0.0}
BESTSELLERS_PARAMS: dict = {"signal": "sales_velocity", "missing": 0.0}


def score(
    candidates: list[dict],
    context: dict,
    weights_result: dict,
    params: dict,
) -> dict[str, float]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    signal: str = p["signal"]
    missing: float = p["missing"]

    raw: dict[str, float] = {}
    for c in candidates:
        key = item_key(c["item_id"], c["size"])
        val = c.get(signal)
        raw[key] = float(val) if val is not None else float(missing)

    values = list(raw.values())
    lo, hi = min(values, default=0.0), max(values, default=1.0)
    span = (hi - lo) or 1.0
    return {key: round((v - lo) / span, 6) for key, v in raw.items()}
