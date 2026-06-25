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


# ── RL integration: rl_weight is a fixed slice carved off the top ──────────────

def test_rl_weight_carved_off_top_emphases_share_remainder():
    # rl gets its fixed slice; the four emphases share (1 - rl_weight) in
    # proportion to their importances. Everything sums to 1.0.
    fw = _fw(40, 30, 20, 10)
    weights = build_agent_weights(fw, rl_weight=0.15)

    assert set(weights) == {"colour", "clothing", "body", "stock", "rl"}
    assert weights["rl"] == pytest.approx(0.15)
    assert sum(weights.values()) == pytest.approx(1.0)
    # 0.85 split 40/30/20/10
    assert weights["colour"]   == pytest.approx(0.40 * 0.85)
    assert weights["clothing"] == pytest.approx(0.30 * 0.85)
    assert weights["body"]     == pytest.approx(0.20 * 0.85)
    assert weights["stock"]    == pytest.approx(0.10 * 0.85)


def test_rl_weight_zero_is_legacy_four_way_split():
    # rl_weight=0 (RL disabled) → no rl key, four emphases share the whole budget.
    fw = _fw(40, 30, 20, 10)
    weights = build_agent_weights(fw, rl_weight=0.0)

    assert "rl" not in weights
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["colour"] == pytest.approx(0.40)


def test_rl_absent_from_responders_redistributes_its_slice():
    # If the RL agent does not respond, its fixed slice is redistributed
    # proportionally among the present agents (which still sum to ~1.0).
    fw = _fw(40, 30, 20, 10)
    present = frozenset({"colour", "clothing", "body", "stock"})
    weights = build_agent_weights(fw, rl_weight=0.15, present_agents=present)

    assert "rl" not in weights
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)


def test_rl_present_but_scorer_missing_keeps_rl_slice():
    # RL responded but a scorer (stock) did not: stock's weight is redistributed
    # among the present agents, RL keeps a (redistributed-inclusive) slice, and
    # the surviving weights still sum to ~1.0.
    fw = _fw(40, 30, 20, 10)
    present = frozenset({"colour", "clothing", "body", "rl"})
    weights = build_agent_weights(fw, rl_weight=0.15, present_agents=present)

    assert set(weights) == {"colour", "clothing", "body", "rl"}
    assert "stock" not in weights
    assert weights["rl"] > 0.0
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)


# ── Detection-confidence folding ───────────────────────────────────────────────

def test_low_body_confidence_shrinks_body_grows_others_no_rl():
    # A low body detection confidence (0.4) scales the body emphasis down before
    # normalising; body's share shrinks, the other three grow, sum stays 1.0.
    fw = _fw(25, 25, 25, 25)
    base   = build_agent_weights(fw, rl_weight=0.0)
    confid = build_agent_weights(
        fw,
        rl_weight=0.0,
        confidences={"colour": 1.0, "clothing": 1.0, "body": 0.4, "stock": 1.0},
    )

    assert sum(confid.values()) == pytest.approx(1.0)
    assert confid["body"] < base["body"]
    for agent in ("colour", "clothing", "stock"):
        assert confid[agent] > base[agent]

    # Concretely: importances become 25/25/10/25 → body = 10/85.
    assert confid["body"] == pytest.approx(10 / 85)


def test_low_body_confidence_with_rl_slice_fixed():
    # rl_weight is a fixed slice untouched by confidence; the four (confidence-
    # adjusted) emphases share the remaining 0.85, and the total stays 1.0.
    fw = _fw(25, 25, 25, 25)
    weights = build_agent_weights(
        fw,
        rl_weight=0.15,
        confidences={"colour": 1.0, "clothing": 1.0, "body": 0.4, "stock": 1.0},
    )

    assert weights["rl"] == pytest.approx(0.15)
    assert sum(weights.values()) == pytest.approx(1.0)
    # emphasis budget 0.85 shared over importances 25/25/10/25 (sum 85).
    assert weights["body"]     == pytest.approx(10 / 85 * 0.85)
    assert weights["colour"]   == pytest.approx(25 / 85 * 0.85)
    assert weights["clothing"] == pytest.approx(25 / 85 * 0.85)
    assert weights["stock"]    == pytest.approx(25 / 85 * 0.85)
    # the four emphases together fill exactly the 0.85 remainder.
    emphasis_sum = sum(weights[a] for a in ("colour", "clothing", "body", "stock"))
    assert emphasis_sum == pytest.approx(0.85)


def test_confidences_all_one_equals_none_byte_identical():
    # All-1.0 confidences (and confidences=None) must be byte-identical to the
    # legacy emphasis-only split for a representative feature_weights sample.
    fw = _fw(40, 30, 20, 10)
    none_out = build_agent_weights(fw, rl_weight=0.15)
    ones_out = build_agent_weights(
        fw,
        rl_weight=0.15,
        confidences={"colour": 1.0, "clothing": 1.0, "body": 1.0, "stock": 1.0},
    )
    assert ones_out == none_out

    # Also with redistribution active.
    present = frozenset({"colour", "clothing", "body", "stock"})
    none_red = build_agent_weights(fw, present_agents=present)
    ones_red = build_agent_weights(
        fw,
        confidences={"colour": 1.0, "clothing": 1.0, "body": 1.0, "stock": 1.0},
        present_agents=present,
    )
    assert ones_red == none_red


def test_all_zero_confidences_fall_back_to_equal_split():
    # All confidences 0 → every (importance × conf) is 0 → total 0 → equal split
    # of the emphasis budget. Must not crash or produce NaN.
    import math

    fw = _fw(40, 30, 20, 10)
    weights = build_agent_weights(
        fw,
        confidences={"colour": 0.0, "clothing": 0.0, "body": 0.0, "stock": 0.0},
    )
    assert sum(weights.values()) == pytest.approx(1.0)
    for agent in ("colour", "clothing", "body", "stock"):
        assert weights[agent] == pytest.approx(0.25)
        assert not math.isnan(weights[agent])


def test_confidence_with_missing_agent_redistribution():
    # Non-trivial confidence AND a missing agent together: a low body confidence
    # reshapes the conf-adjusted emphases, then the absent "stock" agent's weight
    # is redistributed among the present three — which must still sum to ~1.0.
    fw = _fw(40, 30, 20, 10)
    present = frozenset({"colour", "clothing", "body"})
    weights = build_agent_weights(
        fw,
        confidences={"colour": 1.0, "clothing": 1.0, "body": 0.4, "stock": 1.0},
        present_agents=present,
    )

    assert set(weights) == {"colour", "clothing", "body"}
    assert "stock" not in weights
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
    assert all(w > 0.0 for w in weights.values())
    # body was damped (×0.4), so it should rank below colour and clothing.
    assert weights["body"] < weights["clothing"] < weights["colour"]
