"""
multi_agent/experiments/store.py
────────────────────────────────
SQLite results store for the experiment harness.

Follows the connection pattern used across the project (``history.py`` /
``memory.py``): one long-lived connection opened with
``check_same_thread=False``, tables created with ``CREATE TABLE IF NOT EXISTS``,
and an explicit ``.commit()`` after every write. Writes are guarded by a
``threading.Lock`` so the harness can persist from executor threads safely.

Schema (one row per …):
  * ``experiments``  — one experiment run (a spec + git sha + timestamp).
  * ``episodes``     — one simulated conversation (customer × combo × repeat),
                       its turn count and the final 1–5 review.
  * ``turns``        — one recommendation round inside an episode (the shopper
                       message that produced it + the round's agent weights).
  * ``turn_items``   — one ranked item within a turn (its scores + attributes).

The store imports only stdlib, so it is unit-testable without spade / ollama.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Results DB lives next to this module: multi_agent/experiments/results.db
_DEFAULT_DB_PATH = Path(__file__).resolve().parent / "results.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT,
    spec_json     TEXT,
    git_sha       TEXT,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    customer_id   TEXT,
    combo_json    TEXT,
    repeat_idx    INTEGER,
    n_turns       INTEGER,
    final_review  INTEGER,
    review_reason TEXT,
    abandoned     INTEGER DEFAULT 0,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id   INTEGER NOT NULL,
    idx          INTEGER,
    shopper_msg  TEXT,
    weights_json TEXT
);

CREATE TABLE IF NOT EXISTS turn_items (
    turn_id           INTEGER NOT NULL,
    rank              INTEGER,
    item_id           INTEGER,
    size              TEXT,
    final_score       REAL,
    agent_scores_json TEXT,
    item_attrs_json   TEXT
);
"""


class ResultsStore:
    """Thread-safe SQLite store for experiment / episode / turn / item rows."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._lock = threading.Lock()
        self._db: Optional[sqlite3.Connection] = None
        self._db_path = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
        self._init_db()

    # ── Persistence helpers ───────────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._db.executescript(_CREATE_SQL)
            self._db.commit()
        except Exception as exc:
            logger.warning(f"[results] DB init failed ({exc}); store disabled (no-op).")
            self._db = None

    # ── Writers ───────────────────────────────────────────────────────────────

    def create_experiment(
        self,
        name: str,
        spec: dict,
        git_sha: str = "",
    ) -> int:
        """Insert one experiment row and return its ``experiment_id`` (-1 on failure)."""
        if self._db is None:
            return -1
        try:
            with self._lock:
                cur = self._db.execute(
                    "INSERT INTO experiments (name, spec_json, git_sha, created_at) "
                    "VALUES (?,?,?,?)",
                    (name, json.dumps(spec, default=str), git_sha, time.time()),
                )
                self._db.commit()
                return int(cur.lastrowid)
        except Exception as exc:
            logger.warning(f"[results] Failed to create experiment {name!r}: {exc}")
            return -1

    def create_episode(
        self,
        experiment_id: int,
        customer_id: str,
        combo: dict,
        repeat_idx: int,
        n_turns: int,
        final_review: Optional[int],
        review_reason: str,
        abandoned: bool,
    ) -> int:
        """Insert one episode row and return its ``episode_id`` (-1 on failure)."""
        if self._db is None:
            return -1
        try:
            with self._lock:
                cur = self._db.execute(
                    "INSERT INTO episodes (experiment_id, customer_id, combo_json, "
                    "repeat_idx, n_turns, final_review, review_reason, abandoned, "
                    "created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        experiment_id,
                        customer_id,
                        json.dumps(combo, default=str),
                        repeat_idx,
                        n_turns,
                        final_review,
                        review_reason,
                        int(bool(abandoned)),
                        time.time(),
                    ),
                )
                self._db.commit()
                return int(cur.lastrowid)
        except Exception as exc:
            logger.warning(f"[results] Failed to create episode: {exc}")
            return -1

    def add_turn(
        self,
        episode_id: int,
        idx: int,
        shopper_msg: str,
        agent_weights: dict,
        items: list[dict],
    ) -> int:
        """Insert one turn + all its ranked items. Returns ``turn_id`` (-1 on failure).

        ``items`` are the result dicts from ``RecommendationSystem.recommend()``:
        each has ``rank``, ``item_id``, ``size``, ``agent_scores``, ``agent_weights``
        plus the full item attributes. ``final_score`` is read from a ``final_score``
        key if present (the orchestrator does not currently expose one, so it is
        usually ``None``); the rest of the item attrs are stored whole.
        """
        if self._db is None:
            return -1
        try:
            with self._lock:
                cur = self._db.execute(
                    "INSERT INTO turns (episode_id, idx, shopper_msg, weights_json) "
                    "VALUES (?,?,?,?)",
                    (episode_id, idx, shopper_msg, json.dumps(agent_weights, default=str)),
                )
                turn_id = int(cur.lastrowid)

                rows = []
                for item in items:
                    attrs = {
                        k: v
                        for k, v in item.items()
                        if k not in (
                            "rank", "item_id", "size", "agent_scores",
                            "agent_weights", "final_score",
                        )
                    }
                    rows.append(
                        (
                            turn_id,
                            item.get("rank"),
                            item.get("item_id"),
                            item.get("size"),
                            item.get("final_score"),
                            json.dumps(item.get("agent_scores", {}), default=str),
                            json.dumps(attrs, default=str),
                        )
                    )
                if rows:
                    self._db.executemany(
                        "INSERT INTO turn_items (turn_id, rank, item_id, size, "
                        "final_score, agent_scores_json, item_attrs_json) "
                        "VALUES (?,?,?,?,?,?,?)",
                        rows,
                    )
                self._db.commit()
                return turn_id
        except Exception as exc:
            logger.warning(f"[results] Failed to add turn {idx} of episode {episode_id}: {exc}")
            return -1

    # ── Queries (summary / tests) ─────────────────────────────────────────────

    def mean_review_per_combo(self, experiment_id: int) -> list[tuple[str, float, int]]:
        """Return ``(combo_name, mean_review, n_episodes)`` rows for an experiment.

        ``combo_name`` is the OFAT label stored in each episode's ``combo_json``
        under the ``name`` key (see ``run_experiment``). Episodes with a NULL
        review are excluded from the mean but counted is left to the caller.
        """
        if self._db is None:
            return []
        try:
            with self._lock:
                rows = self._db.execute(
                    "SELECT combo_json, final_review FROM episodes "
                    "WHERE experiment_id = ?",
                    (experiment_id,),
                ).fetchall()
        except Exception as exc:
            logger.warning(f"[results] Failed to read episodes: {exc}")
            return []

        # Aggregate in Python so the combo label can come from combo_json.
        buckets: dict[str, list[int]] = {}
        for combo_json, review in rows:
            try:
                combo = json.loads(combo_json or "{}")
            except Exception:
                combo = {}
            label = combo.get("name") or json.dumps(combo.get("strategies", combo))
            buckets.setdefault(label, [])
            if review is not None:
                buckets[label].append(int(review))

        out: list[tuple[str, float, int]] = []
        for label, reviews in buckets.items():
            n = len(reviews)
            mean = sum(reviews) / n if n else 0.0
            out.append((label, mean, n))
        out.sort(key=lambda r: r[1], reverse=True)
        return out

    def episode_rows(self, episode_id: int) -> dict:
        """Fetch one episode + its turns + items (used by the round-trip test)."""
        if self._db is None:
            return {}
        with self._lock:
            ep = self._db.execute(
                "SELECT episode_id, experiment_id, customer_id, combo_json, "
                "repeat_idx, n_turns, final_review, review_reason, abandoned "
                "FROM episodes WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            turns = self._db.execute(
                "SELECT turn_id, idx, shopper_msg, weights_json FROM turns "
                "WHERE episode_id = ? ORDER BY idx",
                (episode_id,),
            ).fetchall()
            items = []
            for turn in turns:
                turn_id = turn[0]
                rows = self._db.execute(
                    "SELECT turn_id, rank, item_id, size, final_score, "
                    "agent_scores_json, item_attrs_json FROM turn_items "
                    "WHERE turn_id = ? ORDER BY rank",
                    (turn_id,),
                ).fetchall()
                items.extend(rows)
        return {"episode": ep, "turns": turns, "items": items}

    def close(self) -> None:
        if self._db is not None:
            try:
                self._db.close()
            finally:
                self._db = None
