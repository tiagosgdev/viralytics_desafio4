"""
multi_agent/history.py
──────────────────────
Shared round history — a process-level singleton visible to every agent.

Two purposes:
  1. Orchestrator staleness check: rounds that have been sitting in the queue
     longer than QUEUE_TTL_S are considered stale (user has moved on) and are
     dropped without execution.

  2. Agent context on comeback: when a scorer agent restarts after a failure it
     calls history.agent_context_summary(agent_id) to see what happened while
     it was down — which rounds ran, how many it missed, and what was detected.
     Completed/failed rounds are persisted to a local SQLite file so agents
     that recover after a full process restart still have context.

All public methods are thread-safe (agents may call from executor threads).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from multi_agent.config import QUEUE_TTL_S

MAX_HISTORY = 50   # keep last N round records in memory

_HISTORY_DB = Path(__file__).resolve().parent / "history.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS round_history (
    conv_id            TEXT PRIMARY KEY,
    wall_time          REAL NOT NULL,
    status             TEXT NOT NULL,
    detected_color     TEXT,
    detected_type      TEXT,
    detected_body_type TEXT,
    result_count       INTEGER DEFAULT 0,
    agents_responded   TEXT,
    agents_missing     TEXT,
    error              TEXT,
    duration_s         REAL
)
"""

logger = logging.getLogger(__name__)


# ── Record ────────────────────────────────────────────────────────────────────

@dataclass
class RoundRecord:
    conv_id:           str
    queued_at:         float            # time.monotonic() when trigger_round() was called
    context:           dict             # detected_color, detected_type, detected_body_type, …
    status:            str  = "queued"  # queued | running | complete | failed | stale
    agents_responded:  list = field(default_factory=list)
    agents_missing:    list = field(default_factory=list)
    result_count:      int  = 0
    error:             Optional[str]   = None
    started_at:        Optional[float] = None
    completed_at:      Optional[float] = None

    def age_s(self) -> float:
        """Seconds since this round was queued."""
        return time.monotonic() - self.queued_at

    def duration_s(self) -> Optional[float]:
        """Wall time from first processing to completion, or None if not done."""
        if self.started_at is not None and self.completed_at is not None:
            return self.completed_at - self.started_at
        return None


# ── History store ─────────────────────────────────────────────────────────────

class RoundHistory:
    """Thread-safe log of recommendation rounds with optional SQLite persistence."""

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._records: dict[str, RoundRecord] = {}
        self._order:   deque[str] = deque(maxlen=MAX_HISTORY)
        self._db: Optional[sqlite3.Connection] = None
        self._init_db()
        self._load_from_db()

    # ── Persistence helpers ───────────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            self._db = sqlite3.connect(str(_HISTORY_DB), check_same_thread=False)
            self._db.execute(_CREATE_SQL)
            self._db.commit()
        except Exception as exc:
            logger.warning(f"[history] DB init failed ({exc}); running in-memory only.")
            self._db = None

    def _load_from_db(self) -> None:
        """Populate in-memory store from the last MAX_HISTORY persisted records."""
        if self._db is None:
            return
        try:
            rows = self._db.execute(
                "SELECT conv_id, status, wall_time, detected_color, detected_type, "
                "detected_body_type, result_count, agents_responded, agents_missing, "
                "error, duration_s FROM round_history "
                "ORDER BY wall_time DESC LIMIT ?",
                (MAX_HISTORY,),
            ).fetchall()
        except Exception as exc:
            logger.warning(f"[history] Failed to load persisted history: {exc}")
            return

        for row in reversed(rows):  # oldest first so _order is chronological
            (conv_id, status, _wall_time, det_color, det_type, det_body,
             result_count, responded_json, missing_json, error, duration_s) = row

            rec = RoundRecord(
                conv_id  = conv_id,
                queued_at = 0.0,  # monotonic from dead process; treat as very old
                context   = {
                    "detected_color":     det_color or "",
                    "detected_type":      det_type  or "",
                    "detected_body_type": det_body  or "",
                },
                status            = status,
                result_count      = result_count or 0,
                agents_responded  = json.loads(responded_json or "[]"),
                agents_missing    = json.loads(missing_json   or "[]"),
                error             = error,
            )
            if duration_s is not None:
                rec.started_at   = 0.0
                rec.completed_at = duration_s  # relative; duration is still correct
            with self._lock:
                self._records[conv_id] = rec
                self._order.append(conv_id)

    def _persist(self, rec: RoundRecord) -> None:
        """Write or update one record in the SQLite store."""
        if self._db is None:
            return
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO round_history "
                "(conv_id, wall_time, status, detected_color, detected_type, "
                "detected_body_type, result_count, agents_responded, agents_missing, "
                "error, duration_s) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rec.conv_id,
                    time.time(),
                    rec.status,
                    rec.context.get("detected_color",     ""),
                    rec.context.get("detected_type",      ""),
                    rec.context.get("detected_body_type", ""),
                    rec.result_count,
                    json.dumps(rec.agents_responded),
                    json.dumps(rec.agents_missing),
                    rec.error,
                    rec.duration_s(),
                ),
            )
            self._db.commit()
        except Exception as exc:
            logger.warning(f"[history] Failed to persist round {rec.conv_id}: {exc}")

    # ── Writers (called by orchestrator) ─────────────────────────────────────

    def record_enqueued(self, conv_id: str, context: dict) -> None:
        rec = RoundRecord(
            conv_id   = conv_id,
            queued_at = time.monotonic(),
            context   = {k: v for k, v in context.items()
                         if k not in ("result_future", "conv_id")},
        )
        with self._lock:
            self._records[conv_id] = rec
            self._order.append(conv_id)
            self._evict()

    def record_started(self, conv_id: str) -> None:
        with self._lock:
            rec = self._records.get(conv_id)
            if rec:
                rec.status     = "running"
                rec.started_at = time.monotonic()

    def record_stale(self, conv_id: str) -> None:
        with self._lock:
            rec = self._records.get(conv_id)
            if rec:
                rec.status       = "stale"
                rec.completed_at = time.monotonic()

    def record_complete(
        self,
        conv_id:          str,
        result_count:     int,
        agents_responded: list[str],
        agents_missing:   list[str],
    ) -> None:
        with self._lock:
            rec = self._records.get(conv_id)
            if rec:
                rec.status           = "complete"
                rec.result_count     = result_count
                rec.agents_responded = list(agents_responded)
                rec.agents_missing   = list(agents_missing)
                rec.completed_at     = time.monotonic()
                self._persist(rec)

    def record_failed(self, conv_id: str, error: str) -> None:
        with self._lock:
            rec = self._records.get(conv_id)
            if rec:
                rec.status       = "failed"
                rec.error        = error
                rec.completed_at = time.monotonic()
                self._persist(rec)

    # ── Queries ───────────────────────────────────────────────────────────────

    def is_stale(self, conv_id: str) -> bool:
        """
        True if this round has been waiting in the queue longer than QUEUE_TTL_S.
        Also returns True if the conv_id is unknown (defensive).
        """
        with self._lock:
            rec = self._records.get(conv_id)
            if rec is None:
                return True
            return rec.age_s() > QUEUE_TTL_S

    def recent_records(self, n: int = 10) -> list[RoundRecord]:
        """Return last n records, oldest first."""
        with self._lock:
            keys = list(self._order)[-n:]
            return [self._records[k] for k in keys if k in self._records]

    def agent_context_summary(self, agent_id: str, n: int = 5) -> str:
        """
        Human-readable summary of recent rounds for an agent that is coming back
        online after a failure.  Logged by each scorer agent in setup().
        """
        records = self.recent_records(n)
        if not records:
            return ""

        lines = [
            f"  [history/{agent_id}] Context from last {len(records)} rounds:"
        ]
        for rec in records:
            ctx  = rec.context
            dur  = f"{rec.duration_s():.1f}s" if rec.duration_s() is not None else "?"
            note = f"  ⚠ YOU WERE ABSENT" if agent_id in rec.agents_missing else ""
            lines.append(
                f"    [{rec.conv_id[:8]}] {rec.status:<8}  "
                f"type={str(ctx.get('detected_type',''))!r:22}  "
                f"body={str(ctx.get('detected_body_type',''))!r:14}  "
                f"results={rec.result_count}  dur={dur}{note}"
            )

        missed = sum(1 for r in records if agent_id in r.agents_missing)
        if missed:
            lines.append(
                f"  [history/{agent_id}] ⚠  Absent for {missed}/{len(records)} recent rounds — "
                f"weights were redistributed to other agents."
            )
        return "\n".join(lines)

    def queue_depth(self) -> int:
        """Number of rounds currently in 'queued' status."""
        with self._lock:
            return sum(1 for r in self._records.values() if r.status == "queued")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evict(self) -> None:
        while len(self._records) > MAX_HISTORY:
            oldest = next(iter(self._order), None)
            if oldest:
                self._order.popleft()
                self._records.pop(oldest, None)
            else:
                break


# ── Module-level singleton ────────────────────────────────────────────────────
# All agents import this directly — safe because all agents run in the same
# Python process and share the same module namespace.
history: RoundHistory = RoundHistory()
