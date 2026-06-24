"""
Candidate retrieval
────────────────────
Shared candidate-retrieval logic for the recommendation round.

`get_candidates` queries a StockAgent for the items that will be debated by the
scorer agents, applying the conversation-driven DB filters and a soft user-gender
include, and returns a list of fully-populated candidate dicts.

This module is deliberately transport-agnostic: it depends only on the StockAgent
interface (passed in explicitly) and `N_CANDIDATES` from config. It imports neither
SPADE nor the orchestrator, so both the live OrchestratorAgent and the (future)
experiment harness can call it to retrieve identical candidate sets.
"""

import logging
import sys
from pathlib import Path

from multi_agent.config import N_CANDIDATES

logger = logging.getLogger(__name__)


# The stock SQL query only accepts the keys in StockAgent's QUERY_KEYS; any other
# field (notably `body_type`, which the intent/weight layer injects for the Qdrant
# vector-search path) makes StockAgent.get_candidates raise ValueError. We strip
# those before querying. Source of truth: stock_agent/stock_stats.py QUERY_KEYS.
# Import it from stock_stats (the dependency-light module where it's defined)
# rather than hardcode, so a schema change there propagates here.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_STOCK_DIR = _REPO_ROOT / "stock_agent"
if str(_STOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_STOCK_DIR))

try:
    from stock_stats import QUERY_KEYS as _STOCK_QUERY_KEYS  # type: ignore[import-untyped]
    _ALLOWED_STOCK_KEYS = frozenset(_STOCK_QUERY_KEYS)
except Exception:  # pragma: no cover - defensive fallback if import is awkward
    # Mirror of stock_agent/stock_stats.py QUERY_KEYS (source of truth).
    _ALLOWED_STOCK_KEYS = frozenset((
        "color", "type", "fit", "size", "style", "pattern", "material",
        "gender", "age_group", "season", "occasion", "brand",
    ))


def _prune_to_stock_keys(sub: dict | None) -> dict:
    """Return a new dict keeping only keys the stock SQL query accepts.

    Drops fields like `body_type` that belong to the vector-search path and would
    otherwise make StockAgent.get_candidates raise. Non-mutating: builds a fresh
    dict so the caller's nested include/exclude dicts are left untouched.
    """
    return {k: v for k, v in (sub or {}).items() if k in _ALLOWED_STOCK_KEYS}


def get_candidates(
    stock_agent,
    weights_result: dict,
    context: dict,
    n: int = N_CANDIDATES,
) -> list[dict]:
    query_filters: dict = dict(weights_result.get("filters") or {})

    # Strip fields the stock SQL query doesn't accept (e.g. body_type, injected
    # for the vector-search path) from both include and exclude. Done first and
    # non-mutating so the conversation's real filters survive while body_type
    # never reaches the stock query. _prune_to_stock_keys copies the nested
    # dicts, so the caller's weights_result is not mutated.
    query_filters["include"] = _prune_to_stock_keys(query_filters.get("include"))
    query_filters["exclude"] = _prune_to_stock_keys(query_filters.get("exclude"))

    # Inject user gender as a soft include so gender-appropriate items rank first
    gender = str(context.get("user_gender") or "").strip().lower()
    if gender in ("male", "female"):
        inc = dict(query_filters.get("include") or {})
        if "gender" not in inc:
            inc["gender"] = [gender, "unisex"]
            query_filters = {**query_filters, "include": inc}

    # get_candidates raises if the query is completely empty (or otherwise
    # invalid). The overstock fallback keeps us returning *something*, but we log
    # it loudly: a silent fallback here is exactly what once hid body_type
    # leaking into the query and degrading every round to the same overstock set.
    try:
        pairs = stock_agent.get_candidates(query_filters, n=n)
    except Exception as exc:
        logger.warning(
            "stock_agent.get_candidates failed (%s); falling back to overstock "
            "items. query_filters=%r",
            exc, query_filters,
        )
        pairs = stock_agent.stats.get_overstock_items(top_k=n)

    info: list[dict] = []
    for iid, sz in pairs:
        try:
            row = stock_agent.stats.get_row(iid, sz)
        except KeyError:
            continue
        info.append({
            "item_id":    int(iid),
            "size":       sz,
            "color":      row.get("color", ""),
            "type":       row.get("type", ""),
            "fit":        row.get("fit", ""),
            "season":     row.get("season", ""),
            "style":      row.get("style", ""),
            "pattern":    row.get("pattern", ""),
            "material":   row.get("material", ""),
            "gender":     row.get("gender", ""),
            "age_group":  row.get("age_group", ""),
            "occasion":   row.get("occasion", ""),
            "brand":      row.get("brand", ""),
            "price":      row.get("price"),
            "stock_count": int(row.get("stock_count", 0)),
            "push_score": float(row.get("push_score", 0.0)),
        })
    return info
