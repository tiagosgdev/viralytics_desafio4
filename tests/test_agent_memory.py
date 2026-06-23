"""Unit tests for AgentMemory (per-agent, write-only autonomy log).

No spade — memory.py imports only stdlib, so these run standalone with tmp_path.
"""

from __future__ import annotations

from pathlib import Path

from multi_agent.memory import AgentMemory


# ── round-trip ──────────────────────────────────────────────────────────────

def test_record_then_recent_and_summary_roundtrip(tmp_path: Path):
    mem = AgentMemory("body", db_dir=tmp_path)
    context = {
        "detected_color": "blue",
        "detected_type": "dress",
        "detected_body_type": "hourglass",
    }
    scores = {"a": 0.9, "b": 0.4, "c": 0.7}
    mem.record("conv-1", context, scores)

    rows = mem.recent()
    assert len(rows) == 1
    row = rows[0]
    assert row["conv_id"] == "conv-1"
    assert row["context"] == context            # context_json deserialises
    assert row["top_scores"]["a"] == 0.9        # top_scores_json deserialises
    assert isinstance(row["timestamp"], float)

    summary = mem.summary()
    assert "conv-1"[:8] in summary
    assert "memory/body" in summary


def test_top_n_trims_to_highest_scores(tmp_path: Path):
    mem = AgentMemory("colour", db_dir=tmp_path)
    scores = {f"item{i}": float(i) / 10 for i in range(10)}  # 0.0 .. 0.9
    mem.record("conv-x", {}, scores, top_n=3)

    top = mem.recent()[0]["top_scores"]
    assert len(top) == 3
    # the three highest: item9=0.9, item8=0.8, item7=0.7
    assert set(top) == {"item9", "item8", "item7"}
    assert min(top.values()) == 0.7


# ── separate files per agent ────────────────────────────────────────────────

def test_separate_agents_write_separate_files(tmp_path: Path):
    body = AgentMemory("body", db_dir=tmp_path)
    colour = AgentMemory("colour", db_dir=tmp_path)

    body.record("conv-body", {"detected_type": "shirt"}, {"x": 1.0})
    colour.record("conv-colour", {"detected_color": "red"}, {"y": 0.5})

    assert (tmp_path / "body.db").exists()
    assert (tmp_path / "colour.db").exists()

    body_rows = body.recent()
    colour_rows = colour.recent()
    assert [r["conv_id"] for r in body_rows] == ["conv-body"]
    assert [r["conv_id"] for r in colour_rows] == ["conv-colour"]
    # no cross-contamination
    assert all(r["conv_id"] != "conv-colour" for r in body_rows)
    assert all(r["conv_id"] != "conv-body" for r in colour_rows)


# ── graceful degradation ────────────────────────────────────────────────────

def test_init_failure_does_not_raise_and_record_noops(tmp_path: Path):
    # Point db_dir at a path that cannot be a directory (a regular file blocks
    # mkdir), forcing _init_db to fail. Construction and record() must not raise.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")

    mem = AgentMemory("stock", db_dir=blocker / "sub")  # mkdir under a file fails
    assert mem._db is None                              # degraded to no-op

    mem.record("conv-z", {"k": "v"}, {"a": 0.1})        # must not raise
    assert mem.recent() == []
    assert mem.summary() == ""


# ── empty store ─────────────────────────────────────────────────────────────

def test_empty_store_returns_sensibly(tmp_path: Path):
    mem = AgentMemory("clothing", db_dir=tmp_path)
    assert mem.recent() == []
    assert mem.summary() == ""


def test_record_never_raises_on_unserialisable_context(tmp_path: Path):
    # Part F's whole point: record() must never crash a round. A context that
    # isn't natively JSON-serialisable must be tolerated, not raised.
    mem = AgentMemory("body", db_dir=tmp_path)

    class _Weird:  # no __dict__ json handling
        def __repr__(self) -> str:
            return "<weird>"

    # Must not raise.
    mem.record("c1", {"obj": _Weird(), "n": 1}, {"7:M": 0.9})
    rows = mem.recent()
    assert len(rows) == 1  # row still written (serialised defensively)


def test_record_never_raises_on_unsortable_scores(tmp_path: Path):
    # Mixed/None score values must not blow up the top-N sort; record() should
    # degrade gracefully (write a row or no-op) without raising.
    mem = AgentMemory("colour", db_dir=tmp_path)
    mem.record("c2", {}, {"1:S": None, "2:M": 0.5})  # must not raise
    # No assertion on contents — the contract under test is "never raises".
