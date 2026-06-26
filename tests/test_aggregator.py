"""Unit tests for the tie-aware weighted Borda + veto selection (aggregator).

Covers:
  * tie-aware Borda: a fully-flat agent is neutral (does not change the order);
    a partially-tied agent separates its groups but adds no order within a group;
  * select_with_veto under weighted (reject-mass ≥ τ) and blackball (any veto)
    elimination, plus the post-veto ranking + item_id dedup.

These exercise the pure aggregator functions only — no SPADE, no DB.
"""

from __future__ import annotations

import pytest

from multi_agent.aggregator import (
    borda_aggregate,
    reject_mass,
    select_with_veto,
    survives_from_mass,
    survives_veto,
)


# ── tie-aware Borda ───────────────────────────────────────────────────────────

def test_flat_agent_is_neutral():
    # `flat` ties everything; `pref` strictly prefers a > b > c. The flat agent
    # must NOT change the order decided by pref, regardless of its weight.
    items = ["1:M", "2:M", "3:M"]
    proposals = {
        "pref": {"1:M": 1.0, "2:M": 0.5, "3:M": 0.0},
        "flat": {k: 0.7 for k in items},
    }
    # Give the flat agent the larger weight — under the old item-id ramp it would
    # have dominated and reordered the result. Tie-aware, it is inert.
    weights = {"pref": 0.3, "flat": 0.7}
    out = borda_aggregate(proposals, weights, k=3)
    assert out == ["1:M", "2:M", "3:M"]


def test_flat_agent_no_item_id_ramp():
    # With ONLY a flat agent, every item is tied → no preference at all. The
    # result must still return all distinct items (order among equals is
    # unspecified, but the composite must be identical for every item).
    items = ["10:M", "2:M", "100:M"]   # item-ids that would sort weirdly as strings
    proposals = {"flat": {k: 1.0 for k in items}}
    weights = {"flat": 1.0}
    out = borda_aggregate(proposals, weights, k=3)
    assert set(out) == set(items)
    assert len(out) == 3


def test_partial_ties_group_average_separates_groups():
    # One agent: 2 items tied high (1.0), 2 items tied low (0.2). The high group
    # must outrank the low group, but items within a group are equal (the input
    # order, not item-id, decides nothing).
    proposals = {
        "a": {"1:M": 1.0, "2:M": 1.0, "3:M": 0.2, "4:M": 0.2},
    }
    weights = {"a": 1.0}
    out = borda_aggregate(proposals, weights, k=4)
    assert set(out[:2]) == {"1:M", "2:M"}   # high group on top
    assert set(out[2:]) == {"3:M", "4:M"}   # low group below


def test_borda_dedups_by_item_id():
    # Two sizes of the same garment; only the best-ranked size survives.
    proposals = {
        "a": {"5:M": 1.0, "5:L": 0.9, "6:M": 0.5},
    }
    out = borda_aggregate(proposals, {"a": 1.0}, k=10)
    ids = [k.split(":", 1)[0] for k in out]
    assert ids == ["5", "6"]                # one entry per item_id


def test_empty_proposals_returns_empty():
    assert borda_aggregate({}, {"a": 1.0}, k=5) == []


# ── select_with_veto ──────────────────────────────────────────────────────────

def _veto_proposals():
    items = ["1:M", "2:M", "3:M", "4:M"]
    proposals = {
        "colour": {k: 1.0 for k in items},
        "body":   {k: 1.0 for k in items},
        "stock":  {k: 1.0 for k in items},
    }
    return items, proposals


def test_weighted_veto_eliminates_when_mass_reaches_tau():
    items, proposals = _veto_proposals()
    weights = {"colour": 0.3, "body": 0.3, "stock": 0.4}
    # 1:M vetoed by colour+body → mass 0.6 ≥ τ=0.5 → eliminated.
    # 2:M vetoed only by colour → mass 0.3 < τ → survives.
    vetoes = {
        "colour": ["1:M", "2:M"],
        "body":   ["1:M"],
        "stock":  [],
    }
    out = select_with_veto(proposals, vetoes, weights, k=10, mode="weighted", tau=0.5)
    ids = {k.split(":", 1)[0] for k in out}
    assert "1" not in ids                    # eliminated (mass 0.6 ≥ τ)
    assert {"2", "3", "4"} <= ids            # all survive


def test_weighted_veto_tau_exact_boundary_eliminates():
    items, proposals = _veto_proposals()
    weights = {"colour": 0.5, "body": 0.3, "stock": 0.2}
    # mass exactly 0.5 must eliminate (≥ τ, not >).
    vetoes = {"colour": ["1:M"], "body": [], "stock": []}
    out = select_with_veto(proposals, vetoes, weights, k=10, mode="weighted", tau=0.5)
    assert "1:M" not in out


def test_blackball_any_single_veto_eliminates():
    items, proposals = _veto_proposals()
    weights = {"colour": 0.1, "body": 0.1, "stock": 0.8}
    # In blackball mode a single low-weight veto still eliminates the item.
    vetoes = {"colour": ["1:M"], "body": [], "stock": []}
    out = select_with_veto(proposals, vetoes, weights, k=10, mode="blackball", tau=0.5)
    ids = {k.split(":", 1)[0] for k in out}
    assert "1" not in ids
    assert {"2", "3", "4"} <= ids


def test_blackball_differs_from_weighted_for_low_weight_veto():
    items, proposals = _veto_proposals()
    weights = {"colour": 0.1, "body": 0.1, "stock": 0.8}
    vetoes = {"colour": ["1:M"], "body": [], "stock": []}
    # weighted (τ=0.5): mass 0.1 < τ → survives.
    w_out = select_with_veto(proposals, vetoes, weights, k=10, mode="weighted", tau=0.5)
    assert "1:M" in w_out
    # blackball: eliminated.
    b_out = select_with_veto(proposals, vetoes, weights, k=10, mode="blackball", tau=0.5)
    assert "1:M" not in b_out


def test_veto_then_rank_uses_tie_aware_borda():
    # Survivors are ranked by the same tie-aware Borda. Here `body` strictly
    # prefers 3 > 2, and there are no vetoes, so 3:M outranks 2:M.
    items, proposals = _veto_proposals()
    proposals["body"] = {"1:M": 0.0, "2:M": 0.5, "3:M": 1.0, "4:M": 0.0}
    weights = {"colour": 0.2, "body": 0.6, "stock": 0.2}
    out = select_with_veto(proposals, {}, weights, k=10, mode="weighted", tau=0.5)
    assert out.index("3:M") < out.index("2:M")


def test_all_items_vetoed_returns_empty():
    items, proposals = _veto_proposals()
    weights = {"colour": 1.0, "body": 0.0, "stock": 0.0}
    vetoes = {"colour": list(items), "body": [], "stock": []}
    out = select_with_veto(proposals, vetoes, weights, k=10, mode="weighted", tau=0.5)
    assert out == []


# ── shared survival rule (reject_mass / survives_veto) ─────────────────────────
# These pure helpers back BOTH select_with_veto and the orchestrator's per-batch
# veto loop, so the survival decision lives in exactly one place.

def test_survives_from_mass_blackball():
    # blackball: zero mass survives, inf (any veto) does not.
    assert survives_from_mass(0.0, mode="blackball", tau=0.5) is True
    assert survives_from_mass(float("inf"), mode="blackball", tau=0.5) is False


def test_survives_from_mass_weighted_tau_boundary():
    # weighted: mass < τ survives, mass == τ does NOT (eliminated at ≥ τ), mass > τ doesn't.
    assert survives_from_mass(0.4, mode="weighted", tau=0.5) is True
    assert survives_from_mass(0.5, mode="weighted", tau=0.5) is False
    assert survives_from_mass(0.6, mode="weighted", tau=0.5) is False


def test_reject_mass_sums_vetoing_agent_weights():
    weights = {"colour": 0.3, "body": 0.3, "stock": 0.4}
    vetoes = {"colour": ["1:M"], "body": ["1:M"], "stock": []}
    # colour + body vetoed 1:M → 0.3 + 0.3 = 0.6
    assert reject_mass("1:M", vetoes, weights) == pytest.approx(0.6)
    # nobody vetoed 2:M → 0.0
    assert reject_mass("2:M", vetoes, weights) == 0.0


def test_reject_mass_blackball_is_inf_on_any_veto():
    weights = {"colour": 0.1, "body": 0.1, "stock": 0.8}
    vetoes = {"colour": ["1:M"], "body": [], "stock": []}
    assert reject_mass("1:M", vetoes, weights, mode="blackball") == float("inf")
    assert reject_mass("2:M", vetoes, weights, mode="blackball") == 0.0


def test_survives_veto_weighted_tau_boundary():
    weights = {"colour": 0.5, "body": 0.3, "stock": 0.2}
    vetoes = {"colour": ["1:M"], "body": [], "stock": []}
    # mass exactly 0.5 must NOT survive (eliminated at ≥ τ).
    assert survives_veto("1:M", vetoes, weights, tau=0.5) is False
    # just below τ survives.
    assert survives_veto("1:M", vetoes, weights, tau=0.6) is True
    # un-vetoed item always survives.
    assert survives_veto("2:M", vetoes, weights, tau=0.5) is True


def test_survives_veto_blackball_any_veto_eliminates():
    weights = {"colour": 0.1, "body": 0.1, "stock": 0.8}
    vetoes = {"colour": ["1:M"], "body": [], "stock": []}
    assert survives_veto("1:M", vetoes, weights, mode="blackball") is False
    assert survives_veto("2:M", vetoes, weights, mode="blackball") is True


def test_survives_veto_matches_select_with_veto_membership():
    # The helper must agree with select_with_veto's survivor set item-for-item.
    items, proposals = _veto_proposals()
    weights = {"colour": 0.3, "body": 0.3, "stock": 0.4}
    vetoes = {"colour": ["1:M", "2:M"], "body": ["1:M"], "stock": []}
    surviving = {
        ik for ik in items
        if survives_veto(ik, vetoes, weights, mode="weighted", tau=0.5)
    }
    out = select_with_veto(proposals, vetoes, weights, k=10, mode="weighted", tau=0.5)
    assert {k.split(":", 1)[0] for k in out} == {k.split(":", 1)[0] for k in surviving}
