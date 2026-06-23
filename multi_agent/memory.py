"""
multi_agent/memory.py
─────────────────────
Per-agent, WRITE-ONLY autonomy log.

Each scorer agent (body / clothing / colour / stock) owns its *own* SQLite file
``multi_agent/memory/<agent_id>.db`` and writes one row per round right after it
sends its PROPOSE: the conversation id, a timestamp, the context it saw, and its
top-N proposed scores.

IMPORTANT — this store is intentionally minimal and is **never read to make a
decision**. The scorers are deterministic and nothing in the recommender
consumes these records to influence scoring, weights, aggregation or the
message protocol. The per-agent duplication exists purely to satisfy a course
requirement (explicit agent autonomy: "each agent has its own memory"). The only
read path is the human-readable ``summary()`` an agent may log about *itself* on
``setup()`` — logging only, never decisions.

This generalises the SQLite-persistence pattern in ``history.py``:
  * one connection per store, ``check_same_thread=False`` (agents may write from
    executor threads),
  * thread-safe writes guarded by a ``threading.Lock``,
  * graceful degradation — if the DB can't be opened or written, log a warning
    and no-op (the round must never crash because of memory).

It imports only stdlib (sqlite3, json, time, threading, pathlib, logging) so it
can be unit-tested without spade.
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

# Default location for the per-agent stores: multi_agent/memory/<agent_id>.db
_DEFAULT_DB_DIR = Path(__file__).resolve().parent / "memory"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS agent_rounds (
    conv_id         TEXT,
    timestamp       REAL NOT NULL,
    context_json    TEXT,
    top_scores_json TEXT
)
"""


class AgentMemory:
    """A single scorer agent's own write-only round log (one SQLite file).

    Design notes:
      * Append-only: ``conv_id`` is NOT a primary key. Each ``record()`` inserts
        a new row, so an agent that (re)processes the same conversation produces
        one row per processing. This keeps the log a faithful history of what the
        agent actually did rather than collapsing re-runs.
      * ``record()`` stores only the TOP-N scores (default 5) sorted descending,
        to keep each row small — these are an autonomy breadcrumb, not the full
        candidate set.
    """

    def __init__(self, agent_id: str, db_dir: Optional[Path] = None) -> None:
        self.agent_id = agent_id
        self._lock = threading.Lock()
        self._db: Optional[sqlite3.Connection] = None

        db_dir = Path(db_dir) if db_dir is not None else _DEFAULT_DB_DIR
        self._db_path = db_dir / f"{agent_id}.db"
        self._init_db(db_dir)

    # ── Persistence helpers ───────────────────────────────────────────────────

    def _init_db(self, db_dir: Path) -> None:
        try:
            db_dir.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._db.execute(_CREATE_SQL)
            self._db.commit()
        except Exception as exc:
            logger.warning(
                f"[memory/{self.agent_id}] DB init failed ({exc}); "
                f"memory disabled (no-op)."
            )
            self._db = None

    # ── Writer (called by the agent right after sending PROPOSE) ──────────────

    def record(
        self,
        conv_id: str,
        context: dict,
        scores: dict[str, float],
        top_n: int = 5,
    ) -> None:
        """Persist one round: conv_id, time, the context seen, and top-N scores.

        ``context`` is stored whole (it is already JSON-safe by the time agents
        hold it). ``scores`` is trimmed to the ``top_n`` highest values (sorted
        descending) to keep the row small. Never raises: if the store failed to
        initialise or the write errors, it logs a warning and no-ops.
        """
        if self._db is None:
            return

        try:
            top_scores = dict(
                sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
            )
            context_json = json.dumps(context, default=str)
            top_scores_json = json.dumps(top_scores)
        except Exception as exc:
            logger.warning(
                f"[memory/{self.agent_id}] Failed to serialise round "
                f"{conv_id}: {exc}"
            )
            return

        try:
            with self._lock:
                self._db.execute(
                    "INSERT INTO agent_rounds "
                    "(conv_id, timestamp, context_json, top_scores_json) "
                    "VALUES (?,?,?,?)",
                    (conv_id, time.time(), context_json, top_scores_json),
                )
                self._db.commit()
        except Exception as exc:
            logger.warning(
                f"[memory/{self.agent_id}] Failed to persist round "
                f"{conv_id}: {exc}"
            )

    # ── Queries (logging only — never used for decisions) ─────────────────────

    def recent(self, n: int = 5) -> list[dict]:
        """Return the last ``n`` records (most recent first) as dicts.

        Each dict: ``{conv_id, timestamp, context, top_scores}``. Returns an
        empty list on a fresh/failed store.
        """
        if self._db is None:
            return []
        try:
            with self._lock:
                rows = self._db.execute(
                    "SELECT conv_id, timestamp, context_json, top_scores_json "
                    "FROM agent_rounds ORDER BY timestamp DESC LIMIT ?",
                    (n,),
                ).fetchall()
        except Exception as exc:
            logger.warning(
                f"[memory/{self.agent_id}] Failed to read recent records: {exc}"
            )
            return []

        out: list[dict] = []
        for conv_id, ts, ctx_json, scores_json in rows:
            try:
                context = json.loads(ctx_json or "{}")
            except Exception:
                context = {}
            try:
                top_scores = json.loads(scores_json or "{}")
            except Exception:
                top_scores = {}
            out.append(
                {
                    "conv_id": conv_id,
                    "timestamp": ts,
                    "context": context,
                    "top_scores": top_scores,
                }
            )
        return out

    def summary(self, n: int = 5) -> str:
        """Human-readable summary of this agent's own last ``n`` rounds.

        Logged by the agent on ``setup()`` (replacing the shared-history
        comeback log). Returns "" when there are no records.
        """
        records = self.recent(n)
        if not records:
            return ""

        lines = [
            f"  [memory/{self.agent_id}] My last {len(records)} rounds "
            f"(write-only autonomy log):"
        ]
        for rec in records:
            ctx = rec["context"]
            top = rec["top_scores"]
            top_str = "  ".join(f"{k}={v:.2f}" for k, v in list(top.items())[:3])
            conv = str(rec["conv_id"] or "")[:8]
            lines.append(
                f"    [{conv}] "
                f"color={str(ctx.get('detected_color', ''))!r:12} "
                f"type={str(ctx.get('detected_type', ''))!r:18} "
                f"body={str(ctx.get('detected_body_type', ''))!r:12} "
                f"▸ {top_str}"
            )
        return "\n".join(lines)
