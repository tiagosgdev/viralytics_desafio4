"""Unit tests for the Part E experiment harness (vertical slice).

Pure-unit only — no live Ollama and no XMPP broker. We test:
  * the OFAT combo generator's SHAPE (baseline + one combo per off-baseline
    strategy, only one agent ever deviating);
  * the SQLite store round-trips a full episode (experiment → episode → turns →
    items) and the per-combo mean-review aggregation;
  * the shopper's JSON parser / fallbacks handle malformed replies (the ollama
    call itself is monkeypatched — never hit live).
"""

from __future__ import annotations

from multi_agent.config import _DEFAULT_AGENT_STRATEGIES
from multi_agent.experiments import shopper
from multi_agent.experiments.spec import SCORER_AGENTS, ofat_combos
from multi_agent.experiments.store import ResultsStore
from multi_agent.strategies import registry


# ── OFAT generator shape ───────────────────────────────────────────────────────

def test_ofat_first_combo_is_baseline():
    combos = list(ofat_combos())
    assert combos[0].name == "baseline"
    assert combos[0].strategies == dict(_DEFAULT_AGENT_STRATEGIES)


def test_ofat_count_matches_off_baseline_strategies():
    # baseline reference + every non-baseline strategy across the four agents.
    expected = 1 + sum(
        len([s for s in registry.strategy_names(a) if s != _DEFAULT_AGENT_STRATEGIES[a]])
        for a in SCORER_AGENTS
    )
    assert len(list(ofat_combos())) == expected


def test_ofat_each_variant_deviates_in_exactly_one_agent():
    for combo in ofat_combos():
        if combo.name == "baseline":
            continue
        diffs = [
            a for a in SCORER_AGENTS
            if combo.strategies[a] != _DEFAULT_AGENT_STRATEGIES[a]
        ]
        assert len(diffs) == 1, f"{combo.name} deviates in {diffs}, expected exactly one"
        # the label encodes the single deviating agent=strategy
        agent, strategy = combo.name.split("=", 1)
        assert diffs[0] == agent
        assert combo.strategies[agent] == strategy


def test_ofat_combos_are_full_four_agent_mappings():
    for combo in ofat_combos():
        assert set(combo.strategies) == set(SCORER_AGENTS)


# ── store round-trip ───────────────────────────────────────────────────────────

def _sample_items() -> list[dict]:
    return [
        {
            "rank": 1, "item_id": 101, "size": "M",
            "color": "red", "type": "short_sleeve_dress", "price": 49.99,
            "agent_scores": {"colour": 1.0, "body": 0.55, "clothing": 0.5, "stock": 0.3},
            "agent_weights": {"colour": 0.4, "body": 0.2, "clothing": 0.2, "stock": 0.2},
        },
        {
            "rank": 2, "item_id": 102, "size": "S",
            "color": "pink", "type": "short_sleeve_dress", "price": 39.0,
            "agent_scores": {"colour": 0.65, "body": 0.5, "clothing": 0.5, "stock": 0.5},
            "agent_weights": {"colour": 0.4, "body": 0.2, "clothing": 0.2, "stock": 0.2},
        },
    ]


def test_store_round_trips_an_episode(tmp_path):
    store = ResultsStore(db_path=tmp_path / "results.db")

    experiment_id = store.create_experiment(
        name="t", spec={"foo": "bar"}, git_sha="deadbeef"
    )
    assert experiment_id > 0

    episode_id = store.create_episode(
        experiment_id=experiment_id,
        customer_id="party_maya",
        combo={"name": "colour=adventurous",
               "strategies": {**_DEFAULT_AGENT_STRATEGIES, "colour": "adventurous"}},
        repeat_idx=0,
        n_turns=2,
        final_review=4,
        review_reason="bold and fun",
        abandoned=False,
    )
    assert episode_id > 0

    items = _sample_items()
    store.add_turn(episode_id, idx=0, shopper_msg="",
                   agent_weights=items[0]["agent_weights"], items=items)
    store.add_turn(episode_id, idx=1, shopper_msg="something bolder",
                   agent_weights=items[0]["agent_weights"], items=items)

    rows = store.episode_rows(episode_id)
    ep = rows["episode"]
    assert ep[2] == "party_maya"          # customer_id
    assert ep[6] == 4                     # final_review
    assert len(rows["turns"]) == 2
    assert len(rows["items"]) == 4        # 2 items per turn × 2 turns

    # mean-review aggregation groups by the combo label.
    summary = store.mean_review_per_combo(experiment_id)
    assert summary == [("colour=adventurous", 4.0, 1)]
    store.close()


def test_store_mean_review_excludes_null_reviews(tmp_path):
    store = ResultsStore(db_path=tmp_path / "results.db")
    exp = store.create_experiment(name="t", spec={})
    combo = {"name": "baseline", "strategies": dict(_DEFAULT_AGENT_STRATEGIES)}
    store.create_episode(exp, "c", combo, 0, 1, 5, "great", False)
    store.create_episode(exp, "c", combo, 1, 1, None, "abandoned", True)
    summary = store.mean_review_per_combo(exp)
    assert summary == [("baseline", 5.0, 1)]   # the NULL review is not averaged in
    store.close()


# ── shopper JSON parsing / fallbacks ───────────────────────────────────────────

def test_extract_json_strips_fence():
    raw = '```json\n{"message": "hi", "stop": false}\n```'
    assert shopper._extract_json(raw) == {"message": "hi", "stop": False}


def test_extract_json_recovers_embedded_object():
    raw = 'Sure! Here you go: {"rating": 4, "reason": "nice"} hope that helps'
    assert shopper._extract_json(raw) == {"rating": 4, "reason": "nice"}


def test_extract_json_returns_empty_on_garbage():
    assert shopper._extract_json("not json at all") == {}
    assert shopper._extract_json("") == {}


def test_coerce_rating_clamps_and_defaults():
    assert shopper._coerce_rating(4) == 4
    assert shopper._coerce_rating("5") == 5
    assert shopper._coerce_rating(9) == 5      # clamped up-bound
    assert shopper._coerce_rating(0) == 1      # clamped low-bound
    assert shopper._coerce_rating(None) == 3   # neutral fallback
    assert shopper._coerce_rating("nope") == 3


def test_next_message_falls_back_on_malformed_llm(monkeypatch):
    # Force the LLM to return junk → neutral, non-stopping message.
    monkeypatch.setattr(shopper, "_ollama_chat", lambda messages: "<<<not json>>>")
    out = shopper.next_message({"temperament": "picky"}, [], [])
    assert isinstance(out["message"], str) and out["message"]
    assert out["stop"] is False


def test_next_message_parses_valid_llm(monkeypatch):
    monkeypatch.setattr(
        shopper, "_ollama_chat",
        lambda messages: '{"message": "show me red dresses", "stop": false}',
    )
    out = shopper.next_message({"temperament": "picky"}, [], [])
    assert out == {"message": "show me red dresses", "stop": False}


def test_final_review_falls_back_to_three(monkeypatch):
    monkeypatch.setattr(shopper, "_ollama_chat", lambda messages: "garbage")
    out = shopper.final_review({"temperament": "picky"}, [], [])
    assert out["rating"] == 3
    assert isinstance(out["reason"], str)


def test_final_review_clamps_out_of_range(monkeypatch):
    monkeypatch.setattr(
        shopper, "_ollama_chat",
        lambda messages: '{"rating": 99, "reason": "loved everything"}',
    )
    out = shopper.final_review({"temperament": "easygoing"}, [], [])
    assert out["rating"] == 5
    assert out["reason"] == "loved everything"
