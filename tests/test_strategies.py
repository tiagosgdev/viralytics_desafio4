"""Unit tests for the pluggable scorer strategies (Part B).

Two themes:
  * BASELINE REGRESSION — the default strategy of each agent must reproduce the
    exact pre-refactor scores (the documented magic numbers).
  * REGISTRY + PERSONALITY behaviour — every registered strategy returns scores
    in [0, 1], the resolver exposes all expected names, and a few behavioural
    assertions distinguish the personalities.
"""

from __future__ import annotations

import pytest

from multi_agent.strategies import body, clothing, colour, stock
from multi_agent.strategies.registry import get_strategy, strategy_names

EXPECTED_NAMES = {
    "colour":   {"purist", "harmonizer", "adventurous"},
    "body":     {"strict", "lenient", "flattering_only"},
    "clothing": {"match_count", "weighted_axes", "strict_type"},
    "stock":    {"push", "overstock_aggressive", "bestsellers"},
}


# ── colour ───────────────────────────────────────────────────────────────────

def _colour_candidates():
    # detected colour is "black"
    return [
        {"item_id": 1, "size": "M", "color": "black"},   # exact
        {"item_id": 2, "size": "M", "color": "white"},   # compatible (in matrix)
        {"item_id": 3, "size": "M", "color": "teal"},    # unrelated to black
    ]


def test_colour_purist_baseline():
    cands = _colour_candidates()
    ctx = {"detected_color": "black"}
    fn, params = get_strategy("colour", "purist")
    scores = fn(cands, ctx, {}, params)
    assert scores == {"1:M": 1.0, "2:M": 0.65, "3:M": 0.20}


def test_colour_no_detected_is_neutral():
    cands = _colour_candidates()
    fn, params = get_strategy("colour", "purist")
    scores = fn(cands, {}, {}, params)
    assert all(v == 0.5 for v in scores.values())


def test_colour_prefers_weights_result_over_context():
    cands = _colour_candidates()
    ctx = {"detected_color": "black"}
    # Uppercase on purpose: the weights value is used as-is (not pre-lowercased)
    # but compared case-insensitively, so "WHITE" must still match item 2's "white".
    wr = {"filters": {"include": {"color": ["WHITE"]}}}
    fn, params = get_strategy("colour", "purist")
    scores = fn(cands, ctx, wr, params)
    # detected becomes white → item 2 is now the exact match
    assert scores["2:M"] == 1.0


def test_colour_adventurous_scores_unrelated_higher_than_purist():
    cands = _colour_candidates()
    ctx = {"detected_color": "black"}
    purist_fn, pp = get_strategy("colour", "purist")
    adv_fn, ap = get_strategy("colour", "adventurous")
    purist = purist_fn(cands, ctx, {}, pp)
    adv = adv_fn(cands, ctx, {}, ap)
    assert adv["3:M"] > purist["3:M"]          # unrelated rewarded more
    assert adv["3:M"] == max(adv.values())     # unrelated is the top bucket


# ── body ─────────────────────────────────────────────────────────────────────

def _body_candidates():
    # detected = "hourglass"; adjacency includes {pear, rectangle}
    return [
        {"item_id": 1, "size": "M", "body_shapes": ["hourglass"]},  # exact
        {"item_id": 2, "size": "M", "body_shapes": ["pear"]},       # adjacent
        {"item_id": 3, "size": "M", "body_shapes": []},             # no data
        {"item_id": 4, "size": "M", "body_shapes": ["apple"]},      # no match
    ]


def test_body_strict_baseline():
    cands = _body_candidates()
    ctx = {"detected_body_type": "hourglass"}
    fn, params = get_strategy("body", "strict")
    scores = fn(cands, ctx, {}, params)
    assert scores == {"1:M": 1.0, "2:M": 0.55, "3:M": 0.20, "4:M": 0.0}


def test_body_no_detected_is_neutral():
    cands = _body_candidates()
    fn, params = get_strategy("body", "strict")
    scores = fn(cands, {}, {}, params)
    assert all(v == 0.5 for v in scores.values())


def test_body_lenient_gives_adjacent_more_than_strict():
    cands = _body_candidates()
    ctx = {"detected_body_type": "hourglass"}
    strict_fn, sp = get_strategy("body", "strict")
    len_fn, lp = get_strategy("body", "lenient")
    strict = strict_fn(cands, ctx, {}, sp)
    lenient = len_fn(cands, ctx, {}, lp)
    assert lenient["2:M"] > strict["2:M"]   # adjacent
    assert lenient["3:M"] > strict["3:M"]   # no-data credit


def test_body_flattering_only_penalises_adjacent_hard():
    cands = _body_candidates()
    ctx = {"detected_body_type": "hourglass"}
    strict_fn, sp = get_strategy("body", "strict")
    fl_fn, fp = get_strategy("body", "flattering_only")
    strict = strict_fn(cands, ctx, {}, sp)
    flat = fl_fn(cands, ctx, {}, fp)
    assert flat["2:M"] < strict["2:M"]   # adjacent penalised
    assert flat["1:M"] == 1.0            # exact still rewarded


# ── clothing ─────────────────────────────────────────────────────────────────

def _clothing_weights(**include):
    return {"filters": {"include": include}}


def _clothing_candidates():
    return [
        {"item_id": 1, "size": "M", "type": "dress",  "style": "casual"},   # 2 hits
        {"item_id": 2, "size": "M", "type": "dress",  "style": "formal"},   # 1 hit (type)
        {"item_id": 3, "size": "M", "type": "shirt",  "style": "casual"},   # 1 hit (style)
        {"item_id": 4, "size": "M", "type": "shirt",  "style": "formal"},   # 0 hits
    ]


def test_clothing_match_count_baseline():
    cands = _clothing_candidates()
    wr = _clothing_weights(type=["dress"], style=["casual"])
    fn, params = get_strategy("clothing", "match_count")
    scores = fn(cands, {}, wr, params)
    assert scores == {"1:M": 1.0, "2:M": 0.5, "3:M": 0.5, "4:M": 0.0}


def test_clothing_no_filters_uniform():
    cands = _clothing_candidates()
    fn, params = get_strategy("clothing", "match_count")
    scores = fn(cands, {}, {}, params)
    assert all(v == 0.5 for v in scores.values())


def test_clothing_skips_body_type_axis():
    cands = _clothing_candidates()
    # only body_type → treated as no active axes → uniform 0.5
    wr = _clothing_weights(body_type=["hourglass"])
    fn, params = get_strategy("clothing", "match_count")
    scores = fn(cands, {}, wr, params)
    assert all(v == 0.5 for v in scores.values())


def test_clothing_strict_type_type_dominates():
    cands = _clothing_candidates()
    wr = _clothing_weights(type=["dress"], style=["casual"])
    mc_fn, mp = get_strategy("clothing", "match_count")
    st_fn, sp = get_strategy("clothing", "strict_type")
    mc = mc_fn(cands, {}, wr, mp)
    st = st_fn(cands, {}, wr, sp)
    # item 2 matches only type; under strict_type it outranks item 3 (style only),
    # whereas under match_count they tie.
    assert mc["2:M"] == mc["3:M"]
    assert st["2:M"] > st["3:M"]


def test_clothing_weighted_axes_weights_type_higher():
    cands = _clothing_candidates()
    wr = _clothing_weights(type=["dress"], style=["casual"])
    fn, params = get_strategy("clothing", "weighted_axes")
    scores = fn(cands, {}, wr, params)
    # type weight (2.0) > style weight (1.5) → type-only match scores higher
    assert scores["2:M"] > scores["3:M"]


# ── stock ────────────────────────────────────────────────────────────────────

def _stock_candidates():
    return [
        {"item_id": 1, "size": "M", "push_score": 0.0,  "stock_count": 5,   "sales_velocity": 9.0},
        {"item_id": 2, "size": "M", "push_score": 0.5,  "stock_count": 50,  "sales_velocity": 3.0},
        {"item_id": 3, "size": "M", "push_score": 1.0,  "stock_count": 100, "sales_velocity": 1.0},
    ]


def test_stock_push_baseline_normalises():
    cands = _stock_candidates()
    fn, params = get_strategy("stock", "push")
    scores = fn(cands, {}, {}, params)
    # min-max over push_score {0.0, 0.5, 1.0} → {0.0, 0.5, 1.0}
    assert scores == {"1:M": 0.0, "2:M": 0.5, "3:M": 1.0}


def test_stock_overstock_favours_high_stock_count():
    cands = _stock_candidates()
    fn, params = get_strategy("stock", "overstock_aggressive")
    scores = fn(cands, {}, {}, params)
    # highest stock_count (item 3) → top score
    assert scores["3:M"] == 1.0
    assert scores["1:M"] == 0.0


def test_stock_bestsellers_favours_velocity():
    cands = _stock_candidates()
    fn, params = get_strategy("stock", "bestsellers")
    scores = fn(cands, {}, {}, params)
    # highest sales_velocity (item 1) → top score
    assert scores["1:M"] == 1.0
    assert scores["3:M"] == 0.0


# ── registry-wide invariants ─────────────────────────────────────────────────

def test_registry_exposes_all_expected_names():
    for agent, names in EXPECTED_NAMES.items():
        assert set(strategy_names(agent)) == names


def test_get_strategy_rejects_unknown():
    with pytest.raises(KeyError):
        get_strategy("colour", "nope")
    with pytest.raises(KeyError):
        get_strategy("nope", "purist")


def test_get_strategy_returns_param_copy():
    _, p1 = get_strategy("colour", "purist")
    p1["exact"] = 0.123
    _, p2 = get_strategy("colour", "purist")
    assert p2["exact"] == 1.0   # mutation didn't leak into the registry


@pytest.mark.parametrize("agent,names", sorted((a, n) for a, n in EXPECTED_NAMES.items()))
def test_every_strategy_returns_scores_in_unit_interval(agent, names):
    cand_builders = {
        "colour":   (_colour_candidates(), {"detected_color": "black"}, {}),
        "body":     (_body_candidates(),   {"detected_body_type": "hourglass"}, {}),
        "clothing": (_clothing_candidates(), {}, _clothing_weights(type=["dress"], style=["casual"])),
        "stock":    (_stock_candidates(),  {}, {}),
    }
    cands, ctx, wr = cand_builders[agent]
    for name in names:
        fn, params = get_strategy(agent, name)
        scores = fn(cands, ctx, wr, params)
        assert scores, f"{agent}/{name} returned no scores"
        for k, v in scores.items():
            assert 0.0 <= v <= 1.0, f"{agent}/{name} {k}={v} out of [0,1]"
