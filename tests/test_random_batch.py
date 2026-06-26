"""Unit tests for the random-batch + veto loop selection path.

Two layers, both broker-free:

  * ``StockAgent.get_random_batch`` — the broad OR-band random sampler — is
    exercised against the REAL stock DB (the same module-scoped fixture pattern
    ``tests/test_price_extraction.py`` uses), asserting band membership, exclusion,
    short-band behaviour, and seed reproducibility.

  * ``OrchestratorBehaviour._run_veto_batch`` — the batch loop — is exercised with
    its three I/O methods (retrieve / broadcast / collect) MOCKED, so no XMPP
    broker and no DB are touched. We assert the loop terminates at MAX_BATCHES,
    stops early once ≥ TOP_K distinct survivors exist, and best-effort fills up to
    TOP_K when survivors are short — and never exceeds MAX_BATCHES (the test
    completing at all is itself the no-infinite-loop guard).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from multi_agent.config import MAX_BATCHES, TOP_K

_REPO = Path(__file__).resolve().parent.parent


# ── get_random_batch against the real stock DB ────────────────────────────────

@pytest.fixture(scope="module")
def stock_agent():
    sys.path.insert(0, str(_REPO / "stock_agent"))
    from stock_agent import StockAgent  # noqa: E402
    return StockAgent()


_INCLUDE_COLORS = ["red"]
_INCLUDE_TYPES = ["short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"]


def _broad_query():
    # Broad OR-band: any red item OR any dress-type item is in the band.
    return {"include": {"color": _INCLUDE_COLORS, "type": _INCLUDE_TYPES}}


def test_random_batch_items_in_broad_or_band(stock_agent):
    # Every returned item must match AT LEAST ONE include signal (the OR-band),
    # not necessarily all of them (that is get_candidates' AND-style ranking).
    pairs = stock_agent.get_random_batch(_broad_query(), n=40, seed=7)
    assert pairs, "expected a non-empty sample from a broad band"
    for iid, sz in pairs:
        row = stock_agent.stats.get_row(iid, sz)
        in_band = row["color"] in _INCLUDE_COLORS or row["type"] in _INCLUDE_TYPES
        assert in_band, f"item {(iid, sz)} (color={row['color']}, type={row['type']}) not in band"


def test_random_batch_band_is_or_not_and(stock_agent):
    # Strengthen the band check: the above "at least one signal" assertion also
    # passes under an accidental AND-narrowing (AND ⊂ OR). To PROVE OR semantics
    # we draw the WHOLE band (huge n → the full pool, deterministic membership,
    # no flaky random draw) and require at least one item matching the `type`
    # signal but NOT the `color` signal — a dress that is not red, which can only
    # be in the band under OR, never under AND.
    huge_n = 10 ** 7  # far larger than the band → returns the entire band once
    pairs = stock_agent.get_random_batch(_broad_query(), n=huge_n)
    assert pairs, "expected a non-empty band"
    type_only = [
        (iid, sz) for iid, sz in pairs
        if (row := stock_agent.stats.get_row(iid, sz))["type"] in _INCLUDE_TYPES
        and row["color"] not in _INCLUDE_COLORS
    ]
    assert type_only, (
        "band contained no type-only item (dress that is not red) — looks like "
        "AND-narrowing, not the OR-band"
    )


def test_random_batch_excludes_seen_item_ids(stock_agent):
    first = stock_agent.get_random_batch(_broad_query(), n=40, seed=1)
    exclude = {iid for iid, _ in first}
    assert exclude
    second = stock_agent.get_random_batch(
        _broad_query(), n=40, exclude_item_ids=exclude, seed=2
    )
    second_ids = {iid for iid, _ in second}
    assert second_ids.isdisjoint(exclude), "excluded item_ids leaked into the next batch"


def test_random_batch_band_smaller_than_n_returns_what_exists(stock_agent):
    # A narrow band (one rare-ish color, no dress backfill) smaller than the
    # requested n must return every band member once — no crash, no duplicates.
    narrow = {"include": {"color": ["red"]}}
    huge_n = 10 ** 7  # far larger than the red band
    pairs = stock_agent.get_random_batch(narrow, n=huge_n)
    assert 0 < len(pairs) < huge_n               # returned what exists, capped at band size
    assert len(set(pairs)) == len(pairs)         # no duplicate (item_id, size) rows


def test_random_batch_seed_is_reproducible(stock_agent):
    a = stock_agent.get_random_batch(_broad_query(), n=25, seed=42)
    b = stock_agent.get_random_batch(_broad_query(), n=25, seed=42)
    assert a == b, "same seed must yield the identical sample"
    c = stock_agent.get_random_batch(_broad_query(), n=25, seed=99)
    assert a != c, "a different seed should (with overwhelming probability) differ"


# ── _run_veto_batch loop, with mocked I/O ─────────────────────────────────────

def _candidate(item_id: int, size: str = "M") -> dict:
    return {"item_id": item_id, "size": size}


def _weights_result(color=100, type_=1, body=1, stock=1) -> dict:
    # color dominates so the colour agent's weight alone exceeds VETO_TAU=0.5 —
    # a single colour veto then eliminates an item under the weighted rule.
    return {
        "weights": {
            "color":    {"importance": color},
            "type":     {"importance": type_},
            "bodyType": {"importance": body},
            "stock":    {"importance": stock},
        }
    }


def _make_behaviour(batches: list[list[dict]], vetoer):
    """Build an OrchestratorBehaviour with retrieve/broadcast/collect mocked.

    batches : the candidate list returned for each successive batch draw; once
              exhausted, an empty list is returned (signalling no more stock).
    vetoer  : fn(batch_keys) -> list[str] of item_keys the `colour` agent vetoes
              for that batch. All four scorers always PROPOSE flat 1.0 scores.
    """
    from multi_agent.agents.orchestrator import OrchestratorBehaviour

    beh = OrchestratorBehaviour()
    state: dict = {"draw": 0, "retrieve_calls": 0, "last_keys": []}

    async def fake_retrieve(context, weights_result, *, random_batch=False, exclude_item_ids=None):
        state["retrieve_calls"] += 1
        idx = state["draw"]
        state["draw"] += 1
        batch = batches[idx] if idx < len(batches) else []
        state["last_keys"] = [f"{c['item_id']}:{c['size']}" for c in batch]
        return list(batch)

    async def fake_broadcast(conv_id, candidates_info, weights_result, context):
        return None

    async def fake_collect(conv_id):
        keys = state["last_keys"]
        scores = {k: 1.0 for k in keys}
        proposals = {a: dict(scores) for a in ("colour", "body", "clothing", "stock")}
        vetoes = {"colour": list(vetoer(keys)), "body": [], "clothing": [], "stock": []}
        return proposals, vetoes

    beh._retrieve_candidates = fake_retrieve      # type: ignore[assignment]
    beh._broadcast_cfp = fake_broadcast           # type: ignore[assignment]
    beh._collect_proposals = fake_collect         # type: ignore[assignment]
    return beh, state


def _run(beh):
    return asyncio.run(
        beh._run_veto_batch({}, "convtest", _weights_result())
    )


def test_loop_terminates_at_max_batches_when_never_enough():
    # Every item is vetoed by the high-weight colour agent → 0 survivors every
    # batch → the loop must run exactly MAX_BATCHES draws and then stop.
    batches = [
        [_candidate(1000 * b + i) for i in range(5)]
        for b in range(MAX_BATCHES + 3)  # supply more than enough, loop must self-limit
    ]
    beh, state = _make_behaviour(batches, vetoer=lambda keys: list(keys))
    outcome = _run(beh)

    assert state["retrieve_calls"] == MAX_BATCHES        # bounded — no infinite loop
    assert outcome is not None
    top_k_keys = outcome[0]
    # Pool was empty (all vetoed), so the result is best-effort fill only.
    assert 0 < len(top_k_keys) <= TOP_K


def test_loop_stops_early_once_top_k_distinct_survivors():
    # One batch with ≥ TOP_K distinct, un-vetoed items → break after one draw.
    big_batch = [_candidate(i) for i in range(TOP_K + 5)]
    beh, state = _make_behaviour([big_batch], vetoer=lambda keys: [])
    outcome = _run(beh)

    assert state["retrieve_calls"] == 1                  # stopped early
    top_k_keys = outcome[0]
    assert len(top_k_keys) == TOP_K                      # exactly the top-k, deduped


def test_best_effort_fill_tops_up_to_top_k_when_short():
    # Each batch has 3 distinct items; colour vetoes 2 of them → only 1 survives
    # per batch. After MAX_BATCHES that is < TOP_K survivors, so the eliminated
    # items must backfill (from the fallback pool) up to TOP_K.
    batches = [
        [_candidate(100 * b + i) for i in range(3)]
        for b in range(MAX_BATCHES)
    ]

    def vetoer(keys):
        return list(keys[:2])  # veto the first two of each batch, keep one

    beh, state = _make_behaviour(batches, vetoer=vetoer)
    outcome = _run(beh)

    assert state["retrieve_calls"] == MAX_BATCHES        # never reached TOP_K survivors
    top_k_keys = outcome[0]
    # 3*MAX_BATCHES distinct items seen ≥ TOP_K, so fill reaches exactly TOP_K.
    assert len(top_k_keys) == TOP_K
    assert len(set(top_k_keys)) == len(top_k_keys)       # distinct keys, no dupes
