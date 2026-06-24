"""Unit tests for conversation-driven price/budget handling.

Two layers:
  * `_extract_price_range` — deterministic budget parsing from chat (pure, fast);
  * the SOFT price filter in `StockAgent.get_candidates` (in-budget = +1
    match_count), exercised through the shared `multi_agent.retrieval`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# feature_weighting lives under LNIAGIA/query_parsing (same path setup the
# weight agent uses at runtime).
_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO / "LNIAGIA", _REPO / "LNIAGIA" / "query_parsing"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from query_parsing.feature_weighting import (  # noqa: E402
    PRICE_CHEAP_MAX,
    PRICE_EXPENSIVE_MIN,
    _extract_price_range,
)


# ── deterministic extractor ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text, expected",
    [
        ("under $50", (None, 50.0)),
        ("below 100", (None, 100.0)),
        ("no more than 75", (None, 75.0)),
        ("over 200", (200.0, None)),
        ("at least 150", (150.0, None)),
        ("nothing over $40", (None, 40.0)),   # negated floor = ceiling
        ("not above 75", (None, 75.0)),
        ("no more than 60", (None, 60.0)),
        ("between 30 and 80", (30.0, 80.0)),
        ("$30-80", (30.0, 80.0)),
        ("80 to 30", (30.0, 80.0)),          # normalised low/high
        ("i want a red dress", (None, None)),  # no budget signal
    ],
)
def test_numeric_phrases(text, expected):
    assert _extract_price_range(text) == expected


def test_vague_terms_map_to_thresholds():
    assert _extract_price_range("something cheap") == (None, PRICE_CHEAP_MAX)
    assert _extract_price_range("keep it affordable") == (None, PRICE_CHEAP_MAX)
    assert _extract_price_range("premium designer piece") == (PRICE_EXPENSIVE_MIN, None)
    assert _extract_price_range("mid-range please") == (PRICE_CHEAP_MAX, PRICE_EXPENSIVE_MIN)


def test_negated_expensive_means_cheaper():
    # "not/too expensive" must map to a budget ceiling, not a floor.
    assert _extract_price_range("not too expensive") == (None, PRICE_CHEAP_MAX)
    assert _extract_price_range("nothing expensive") == (None, PRICE_CHEAP_MAX)


def test_numeric_wins_over_vague():
    # An explicit number takes priority over a vague word in the same sentence.
    assert _extract_price_range("cheap, under 25 ideally") == (None, 25.0)


def test_empty_returns_none():
    assert _extract_price_range("") == (None, None)
    assert _extract_price_range(None) == (None, None)


# ── soft price filter through retrieval (integration with real StockAgent) ───

@pytest.fixture(scope="module")
def stock_agent():
    sys.path.insert(0, str(_REPO / "stock_agent"))
    from stock_agent import StockAgent  # noqa: E402
    return StockAgent()


def _red_dress_query(**extra):
    flt = {
        "include": {
            "color": ["red"],
            "type": ["short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"],
        },
        "exclude": {},
    }
    flt.update(extra)
    return {"filters": flt}


def test_impossible_budget_still_backfills_to_n(stock_agent):
    # A budget almost no item meets must NOT shrink the pool: price is soft, so
    # out-of-budget items backfill the n candidates (tiered relaxation).
    from multi_agent.retrieval import get_candidates
    out = get_candidates(stock_agent, _red_dress_query(price_max=1.0), {"user_gender": "female"})
    assert len(out) == 40


def test_in_budget_items_rank_first(stock_agent):
    from multi_agent.retrieval import get_candidates
    out = get_candidates(stock_agent, _red_dress_query(price_max=40.0), {"user_gender": "female"})
    prices = [c["price"] for c in out]
    # Every in-budget item must appear before the first out-of-budget one.
    first_over = next((i for i, p in enumerate(prices) if p > 40.0), len(prices))
    assert all(p <= 40.0 for p in prices[:first_over])
    assert prices[:first_over], "expected at least one in-budget red dress under $40"
