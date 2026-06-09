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

All public methods are thread-safe (agents may call from executor threads).
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from multi_agent.config import QUEUE_TTL_S

MAX_HISTORY = 50   # keep last N round records in memory


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
    """Thread-safe in-memory log of recommendation rounds."""

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._records: dict[str, RoundRecord] = {}
        self._order:   deque[str] = deque(maxlen=MAX_HISTORY)

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

    def record_failed(self, conv_id: str, error: str) -> None:
        with self._lock:
            rec = self._records.get(conv_id)
            if rec:
                rec.status       = "failed"
                rec.error        = error
                rec.completed_at = time.monotonic()

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
