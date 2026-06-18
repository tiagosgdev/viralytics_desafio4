"""Unit tests for build_agent_weights (conversation-driven 4-way weights)."""

from __future__ import annotations

import pytest

from multi_agent.aggregator import build_agent_weights


def _fw(color: float, type_: float, body: float, stock: float) -> dict:
    """Helper: build a feature_weights dict in analyze_intent shape."""
    return {
        "color":    {"importance": color},
        "type":     {"importance": type_},
        "bodyType": {"importance": body},
        "stock":    {"importance": stock},
    }


def test_four_importances_normalise_and_map_correctly():
    # 40/30/20/10 → normalised to sum 1.0, with the documented id mapping.
    fw = _fw(40, 30, 20, 10)
    weights = build_agent_weights(fw)

    assert set(weights) == {"colour", "clothing", "body", "stock"}
    assert sum(weights.values()) == pytest.approx(1.0)

    # color→colour, type→clothing, bodyType→body, stock→stock
    assert weights["colour"]   == pytest.approx(0.40)
    assert weights["clothing"] == pytest.approx(0.30)
    assert weights["body"]     == pytest.approx(0.20)
    assert weights["stock"]    == pytest.approx(0.10)


def test_missing_agent_redistribution():
    # Drop "stock" from the responders; its weight should be redistributed
    # proportionally among the present three, which must still sum to ~1.0.
    fw = _fw(40, 30, 20, 10)
    present = frozenset({"colour", "clothing", "body"})
    weights = build_agent_weights(fw, present_agents=present)

    assert set(weights) == {"colour", "clothing", "body"}
    assert "stock" not in weights
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)

    # stock's 0.10 is split proportionally to the present weights (0.4/0.3/0.2),
    # i.e. each present agent keeps its share of the surviving 0.90 budget.
    assert weights["colour"]   == pytest.approx(0.40 / 0.90, abs=1e-3)
    assert weights["clothing"] == pytest.approx(0.30 / 0.90, abs=1e-3)
    assert weights["body"]     == pytest.approx(0.20 / 0.90, abs=1e-3)


def test_missing_middle_agent_redistribution():
    # Drop "colour" (a non-last agent) — its weight redistributes proportionally
    # among the surviving three, which must still sum to ~1.0 and stay positive.
    fw = _fw(40, 30, 20, 10)
    present = frozenset({"clothing", "body", "stock"})
    weights = build_agent_weights(fw, present_agents=present)

    assert set(weights) == {"clothing", "body", "stock"}
    assert "colour" not in weights
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
    assert all(w > 0.0 for w in weights.values())
    # surviving budget is 0.60; each keeps its share of it.
    assert weights["clothing"] == pytest.approx(0.30 / 0.60, abs=1e-3)
    assert weights["body"]     == pytest.approx(0.20 / 0.60, abs=1e-3)
    assert weights["stock"]    == pytest.approx(0.10 / 0.60, abs=1e-3)


def test_multiple_missing_agents_redistribution():
    # Two agents absent — only colour & stock respond. Their weights must still
    # sum to ~1.0 and remain positive after the pooled weight is redistributed.
    fw = _fw(40, 30, 20, 10)
    present = frozenset({"colour", "stock"})
    weights = build_agent_weights(fw, present_agents=present)

    assert set(weights) == {"colour", "stock"}
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
    assert all(w > 0.0 for w in weights.values())
    # surviving budget is 0.50 (0.40 + 0.10); shares preserved.
    assert weights["colour"] == pytest.approx(0.40 / 0.50, abs=1e-3)
    assert weights["stock"]  == pytest.approx(0.10 / 0.50, abs=1e-3)


def test_fallback_equal_split_when_importances_absent_or_zero():
    # All-zero importances → equal four-way split.
    zero = build_agent_weights(_fw(0, 0, 0, 0))
    assert sum(zero.values()) == pytest.approx(1.0)
    for agent in ("colour", "clothing", "body", "stock"):
        assert zero[agent] == pytest.approx(0.25)

    # Empty dict → equal four-way split too.
    empty = build_agent_weights({})
    assert sum(empty.values()) == pytest.approx(1.0)
    for agent in ("colour", "clothing", "body", "stock"):
        assert empty[agent] == pytest.approx(0.25)


def test_backward_compat_missing_stock_key_does_not_crash():
    # Older callers send only the three user-facing emphases, no stock key.
    # The fallback stock importance (stock_weight) is used for stock so the
    # function still produces a valid 4-way distribution without crashing.
    legacy = {
        "color":    {"importance": 50},
        "type":     {"importance": 30},
        "bodyType": {"importance": 20},
    }
    weights = build_agent_weights(legacy, stock_weight=0.20)

    assert set(weights) == {"colour", "clothing", "body", "stock"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(w >= 0.0 for w in weights.values())
    # stock_weight (0.20) is a *share*, converted to a comparable importance so
    # the legacy caller's stock agent actually receives ~20% of the final budget.
    assert weights["stock"] == pytest.approx(0.20, abs=1e-3)
    # color was the dominant emphasis, so colour should outrank the rest.
    assert weights["colour"] == max(weights.values())


def test_present_agents_none_returns_unredistributed_full():
    fw = _fw(25, 25, 25, 25)
    weights = build_agent_weights(fw, present_agents=None)
    assert weights == {
        "colour":   pytest.approx(0.25),
        "clothing": pytest.approx(0.25),
        "body":     pytest.approx(0.25),
        "stock":    pytest.approx(0.25),
    }
