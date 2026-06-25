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

# The stock pool is at (item_id, size) grain. To collect N_CANDIDATES *distinct*
# items we overfetch ~this many rows per wanted item (a garment has up to ~8
# sizes), then keep the best-stocked size of each. Bounded so a small/over-
# filtered catalogue still returns promptly.
_SIZE_OVERFETCH = 8


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


def _prep_query_filters(weights_result: dict, context: dict) -> dict:
    """Build the stock-query filters: prune to stock keys + inject soft gender.

    Shared by the legacy exact-filter path and the broad-random path so both see
    the same conversation filters (minus vector-only fields like body_type) and
    the same gender soft-include. Non-mutating w.r.t. the caller's weights_result.
    """
    query_filters: dict = dict(weights_result.get("filters") or {})
    query_filters["include"] = _prune_to_stock_keys(query_filters.get("include"))
    query_filters["exclude"] = _prune_to_stock_keys(query_filters.get("exclude"))

    gender = str(context.get("user_gender") or "").strip().lower()
    if gender in ("male", "female"):
        inc = dict(query_filters.get("include") or {})
        if "gender" not in inc:
            inc["gender"] = [gender, "unisex"]
            query_filters = {**query_filters, "include": inc}
    return query_filters


def _collapse_pairs_to_candidates(
    stock_agent, pairs, n: int
) -> list[dict]:
    """Collapse (item_id, size) pairs to ≤ n DISTINCT-item candidate dicts.

    Keeps the best-stocked in-stock size of each item_id, preserving the order in
    which each item first appears in `pairs` (match-count order for the legacy
    path, random order for the batch path).
    """
    best_by_item: dict[int, dict] = {}
    order: list[int] = []
    for iid, sz in pairs:
        try:
            row = stock_agent.stats.get_row(iid, sz)
        except KeyError:
            continue
        iid = int(iid)
        cand = {
            "item_id":    iid,
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
        }
        prev = best_by_item.get(iid)
        if prev is None:
            best_by_item[iid] = cand
            order.append(iid)
        elif cand["stock_count"] > prev["stock_count"]:
            best_by_item[iid] = cand

    return [best_by_item[iid] for iid in order[:n]]


def get_candidates(
    stock_agent,
    weights_result: dict,
    context: dict,
    n: int = N_CANDIDATES,
) -> list[dict]:
    # Strip vector-only fields (e.g. body_type) and inject the soft gender include
    # — done first and non-mutating so the caller's weights_result is untouched.
    query_filters = _prep_query_filters(weights_result, context)

    # get_candidates raises if the query is completely empty (or otherwise
    # invalid). The overstock fallback keeps us returning *something*, but we log
    # it loudly: a silent fallback here is exactly what once hid body_type
    # leaking into the query and degrading every round to the same overstock set.
    #
    # The stock pool is at (item_id, size) grain, so a plain top-n collapses to a
    # handful of garments repeated across sizes (~7 distinct items for n=40). We
    # overfetch and collapse to n DISTINCT items below, so the debate — and the
    # final top-k — span n real garments, not the same few in every size.
    raw_n = n * _SIZE_OVERFETCH
    try:
        pairs = stock_agent.get_candidates(query_filters, n=raw_n)
    except Exception as exc:
        logger.warning(
            "stock_agent.get_candidates failed (%s); falling back to overstock "
            "items. query_filters=%r",
            exc, query_filters,
        )
        pairs = stock_agent.stats.get_overstock_items(top_k=raw_n)

    return _collapse_pairs_to_candidates(stock_agent, pairs, n)


def get_random_candidates(
    stock_agent,
    weights_result: dict,
    context: dict,
    n: int = N_CANDIDATES,
    *,
    exclude_item_ids=None,
    seed: int | None = None,
) -> list[dict]:
    """Broad-random candidate retrieval for SELECTION_MODE == "veto_batch".

    Samples `n` random in-stock items from the BROAD relevance band (rows matching
    at least one include signal), excluding any `exclude_item_ids` already seen in
    earlier batches. Same candidate-dict shape and distinct-item collapse as
    `get_candidates`, so the scorer agents are agnostic to which path produced the
    slate. Falls back to overstock on any error.
    """
    # NB: the broad OR-band is built (in get_random_batch) from color/type/price
    # only. body_type is a vector-only signal that _prep_query_filters strips via
    # _prune_to_stock_keys before the stock query, so — despite spec decision #2
    # listing it as a band signal — body_type does NOT contribute to the band here.
    query_filters = _prep_query_filters(weights_result, context)

    raw_n = n * _SIZE_OVERFETCH
    try:
        pairs = stock_agent.get_random_batch(
            query_filters,
            n=raw_n,
            exclude_item_ids=exclude_item_ids,
            seed=seed,
        )
    except Exception as exc:
        logger.warning(
            "stock_agent.get_random_batch failed (%s); falling back to overstock "
            "items. query_filters=%r",
            exc, query_filters,
        )
        pairs = stock_agent.stats.get_overstock_items(top_k=raw_n)

    return _collapse_pairs_to_candidates(stock_agent, pairs, n)
