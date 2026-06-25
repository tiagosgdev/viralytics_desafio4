"""
BodyRecommenderAgent
────────────────────
Receives a CFP from the Orchestrator with 40 candidate items.
Scores each item by how well its `body_type` field (stored in clothing.db)
matches the detected body shape, and returns a sealed PROPOSE.

The agent owns the DB IO: it fetches each item's body shapes from clothing.db
and enriches the candidate dicts with a ``body_shapes`` field, then hands the
enriched candidates to a swappable scoring strategy (personality) selected via
``AGENT_STRATEGIES["body"]``. The pure strategy never opens the DB.

Scoring (strict / baseline):
  exact match   → 1.0   (item is designed for this body shape)
  adjacent shape → 0.55  (neighbouring shape on the body-shape graph)
  no data        → 0.20  (item has no body_type metadata — neutral low score)
  no match       → 0.0
"""

import asyncio
import logging
import sqlite3
import sys
from pathlib import Path

from spade.behaviour import CyclicBehaviour
from spade.template import Template

from multi_agent import config
from multi_agent.agents.base import BaseRecommenderAgent
from multi_agent.memory import AgentMemory
from multi_agent.messages import parse, make_propose, comm_log, CFP
from multi_agent.strategies.registry import get_strategy

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DB_PATH   = _REPO_ROOT / "LNIAGIA" / "DB" / "SQLLite" / "clothing.db"

logger = logging.getLogger(__name__)

# Module-level persistent connection — opened once and reused across rounds.
# check_same_thread=False because _load_body_types is called from a thread-pool
# executor while the agent lives on the asyncio event loop thread.
_db_conn: sqlite3.Connection | None = None


def _get_db_conn() -> sqlite3.Connection | None:
    global _db_conn
    if not _DB_PATH.exists():
        return None
    if _db_conn is None:
        _db_conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    return _db_conn

def _load_body_types(item_ids: list[int]) -> dict[int, frozenset[str]]:
    """Batch-fetch body_type column from clothing.db for the given item ids."""
    if not item_ids:
        return {}
    conn = _get_db_conn()
    if conn is None:
        return {}

    placeholders = ",".join("?" * len(item_ids))
    try:
        rows = conn.execute(
            f"SELECT id, body_type FROM items WHERE id IN ({placeholders})",
            item_ids,
        ).fetchall()
    except Exception as exc:
        logger.error(f"body_type DB query failed: {exc}")
        return {}

    result: dict[int, frozenset[str]] = {}
    for row_id, raw in rows:
        if raw:
            shapes = frozenset(s.strip().lower() for s in raw.split(",") if s.strip())
        else:
            shapes = frozenset()
        result[int(row_id)] = shapes
    return result


class BodyScoreBehaviour(CyclicBehaviour):
    async def run(self) -> None:
        msg = await self.receive(timeout=60)
        if msg is None:
            return

        data            = parse(msg)
        conv_id         = data.get("conv_id", "")
        candidates_info = data.get("candidates", [])
        context         = data.get("context", {})
        weights_result  = data.get("weights_result", {})
        detected        = str(context.get("detected_body_type") or "").lower().strip()

        item_ids = [int(c["item_id"]) for c in candidates_info]

        # IO stays in the agent: fetch body shapes and enrich each candidate so
        # the pure strategy can read them off the candidate dict (body_shapes).
        loop = asyncio.get_event_loop()
        body_types: dict[int, frozenset] = await loop.run_in_executor(
            None, lambda: _load_body_types(item_ids)
        )
        for c in candidates_info:
            c["body_shapes"] = list(body_types.get(int(c["item_id"]), frozenset()))

        strategy_name = config.AGENT_STRATEGIES["body"]
        score_fn, params = get_strategy("body", strategy_name)
        params.update(config.AGENT_STRATEGY_PARAMS.get("body", {}))

        scores: dict[str, float] = score_fn(
            candidates_info, context, weights_result, params
        )

        propose = make_propose(
            to_jid   = str(msg.sender),
            conv_id  = conv_id,
            agent_id = "body",
            scores   = scores,
        )
        await self.send(propose)
        # Write-only per-agent memory (course requirement; never read for
        # decisions). Defensive: must not break the round if _memory is absent.
        if mem := getattr(self.agent, "_memory", None):
            mem.record(conv_id, context, scores)
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        top_str = "  ".join(f"{k}={v:.2f}" for k, v in top3)
        comm_log("body", "orchestrator", "PROPOSE", conv_id,
                 f"{len(scores)} scores  ▸ {top_str}")
        logger.info(f"[{conv_id}] BodyAgent PROPOSE sent ({len(scores)} items, detected={detected!r}).")


class BodyRecommenderAgent(BaseRecommenderAgent):
    async def setup(self) -> None:
        template = Template()
        template.set_metadata("performative", CFP)
        self.add_behaviour(BodyScoreBehaviour(), template)
        self._memory = AgentMemory("body")
        summary = self._memory.summary()
        if summary:
            logger.info(summary)
        logger.info("BodyRecommenderAgent ready.")
