"""Unit tests for the shared candidate-retrieval function (Part D).

`get_candidates` is exercised with a STUB stock_agent — no real DB and no SPADE.
The stub mimics the slice of the StockAgent interface the function touches:
  * ``.get_candidates(filters, n)`` → list of (item_id, size) pairs
  * ``.stats.get_row(iid, sz)``     → attribute dict for a pair
  * ``.stats.get_overstock_items(top_k)`` → fallback pairs

Coverage:
  * happy path — candidate dicts have the exact expected keys/values;
  * gender injection — adds include["gender"] == [gender, "unisex"] only when
    no gender filter is already present;
  * fallback — get_candidates raising falls back to get_overstock_items;
  * get_row KeyError on one pair → that pair is skipped.
"""

from __future__ import annotations

import pytest

from multi_agent.retrieval import get_candidates


# ── Stub stock agent ──────────────────────────────────────────────────────────

class _StubStats:
    def __init__(self, rows: dict, overstock: list | None = None):
        # rows maps (item_id, size) -> attribute dict; missing keys raise KeyError
        self._rows = rows
        self._overstock = overstock or []

    def get_row(self, iid, sz):
        return self._rows[(iid, sz)]

    def get_overstock_items(self, top_k):
        return list(self._overstock)[:top_k]


class _StubStockAgent:
    def __init__(self, pairs=None, rows=None, overstock=None, raise_on_query=False):
        self._pairs = pairs or []
        self._raise_on_query = raise_on_query
        self.stats = _StubStats(rows or {}, overstock)
        self.received_filters = None
        self.received_n = None

    def get_candidates(self, filters, n):
        self.received_filters = filters
        self.received_n = n
        if self._raise_on_query:
            raise ValueError("empty query")
        return list(self._pairs)


def _full_row(**overrides):
    row = {
        "color": "black", "type": "shirt", "fit": "slim", "season": "summer",
        "style": "casual", "pattern": "solid", "material": "cotton",
        "gender": "male", "age_group": "adult", "occasion": "everyday",
        "brand": "acme", "price": 19.99, "stock_count": 7, "push_score": 0.42,
    }
    row.update(overrides)
    return row


EXPECTED_KEYS = {
    "item_id", "size", "color", "type", "fit", "season", "style", "pattern",
    "material", "gender", "age_group", "occasion", "brand", "price",
    "stock_count", "push_score",
}


# ── happy path ────────────────────────────────────────────────────────────────

def test_happy_path_exact_keys_and_values():
    rows = {("11", "M"): _full_row()}
    agent = _StubStockAgent(pairs=[("11", "M")], rows=rows)

    out = get_candidates(agent, {"filters": {}}, {})

    assert len(out) == 1
    cand = out[0]
    assert set(cand.keys()) == EXPECTED_KEYS
    assert cand["item_id"] == 11 and isinstance(cand["item_id"], int)
    assert cand["size"] == "M"
    assert cand["color"] == "black"
    assert cand["type"] == "shirt"
    assert cand["price"] == 19.99
    assert cand["stock_count"] == 7 and isinstance(cand["stock_count"], int)
    assert cand["push_score"] == 0.42 and isinstance(cand["push_score"], float)


def test_missing_row_fields_use_defaults():
    rows = {(1, "S"): {}}  # no attributes at all
    agent = _StubStockAgent(pairs=[(1, "S")], rows=rows)

    out = get_candidates(agent, {}, {})

    cand = out[0]
    assert set(cand.keys()) == EXPECTED_KEYS
    assert cand["color"] == "" and cand["brand"] == ""
    assert cand["price"] is None
    assert cand["stock_count"] == 0
    assert cand["push_score"] == 0.0


# ── gender injection ──────────────────────────────────────────────────────────

def test_gender_injection_when_absent():
    agent = _StubStockAgent(pairs=[], rows={})
    get_candidates(agent, {"filters": {"include": {}}}, {"user_gender": "Male"})

    assert agent.received_filters["include"]["gender"] == ["male", "unisex"]


def test_gender_injection_female_with_no_include_key():
    agent = _StubStockAgent(pairs=[], rows={})
    get_candidates(agent, {"filters": {}}, {"user_gender": "female"})

    assert agent.received_filters["include"]["gender"] == ["female", "unisex"]


def test_gender_not_overwritten_when_present():
    agent = _StubStockAgent(pairs=[], rows={})
    filters = {"filters": {"include": {"gender": ["unisex"]}}}
    get_candidates(agent, filters, {"user_gender": "male"})

    assert agent.received_filters["include"]["gender"] == ["unisex"]


def test_no_gender_injection_for_unknown_gender():
    agent = _StubStockAgent(pairs=[], rows={})
    get_candidates(agent, {"filters": {"include": {}}}, {"user_gender": "other"})

    assert "gender" not in agent.received_filters.get("include", {})


def test_exclude_filters_preserved_through_gender_injection():
    # Injecting a gender include must not drop sibling exclude filters.
    agent = _StubStockAgent(pairs=[], rows={})
    filters = {"filters": {"include": {}, "exclude": {"color": ["red"]}}}
    get_candidates(agent, filters, {"user_gender": "male"})

    assert agent.received_filters["include"]["gender"] == ["male", "unisex"]
    assert agent.received_filters["exclude"] == {"color": ["red"]}


def test_caller_weights_result_not_mutated():
    # get_candidates must copy filters, never mutate the caller's dict.
    agent = _StubStockAgent(pairs=[], rows={})
    weights_result = {"filters": {"include": {}}}
    get_candidates(agent, weights_result, {"user_gender": "male"})

    # Original dict is untouched; the gender include only lives on the copy
    # the function passed to the stock agent.
    assert weights_result["filters"]["include"] == {}
    assert agent.received_filters is not weights_result["filters"]
    assert agent.received_filters["include"]["gender"] == ["male", "unisex"]


# ── fallback ──────────────────────────────────────────────────────────────────

def test_fallback_to_overstock_on_query_error():
    rows = {(99, "L"): _full_row(color="red")}
    agent = _StubStockAgent(
        rows=rows, overstock=[(99, "L")], raise_on_query=True
    )

    out = get_candidates(agent, {"filters": {}}, {})

    assert len(out) == 1
    assert out[0]["item_id"] == 99
    assert out[0]["color"] == "red"


# ── get_row KeyError skip ─────────────────────────────────────────────────────

def test_get_row_keyerror_skips_pair():
    rows = {(1, "M"): _full_row(color="blue")}  # (2, "M") intentionally missing
    agent = _StubStockAgent(pairs=[(1, "M"), (2, "M")], rows=rows)

    out = get_candidates(agent, {"filters": {}}, {})

    assert [c["item_id"] for c in out] == [1]
    assert out[0]["color"] == "blue"


# ── n forwarding ──────────────────────────────────────────────────────────────

def test_n_defaults_and_forwards():
    agent = _StubStockAgent(pairs=[], rows={})
    get_candidates(agent, {}, {})
    assert agent.received_n == 40  # N_CANDIDATES default

    agent2 = _StubStockAgent(pairs=[], rows={})
    get_candidates(agent2, {}, {}, n=5)
    assert agent2.received_n == 5
