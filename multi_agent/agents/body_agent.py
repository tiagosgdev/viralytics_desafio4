"""
BodyRecommenderAgent
────────────────────
Receives a CFP from the Orchestrator with 40 candidate items.
Scores each item by how well its `body_type` field (stored in clothing.db)
matches the detected body shape, and returns a sealed PROPOSE.

Scoring:
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

from multi_agent.agents.base import BaseRecommenderAgent
from multi_agent.messages import parse, make_propose, comm_log, CFP

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DB_PATH   = _REPO_ROOT / "LNIAGIA" / "DB" / "SQLLite" / "clothing.db"

logger = logging.getLogger(__name__)

# Shapes considered "close enough" to get partial credit.
# Symmetric: if A is adjacent to B then B is adjacent to A.
_ADJACENT: dict[str, frozenset] = {
    "hourglass":         frozenset({"pear", "rectangle"}),
    "pear":              frozenset({"hourglass", "triangle"}),
    "triangle":          frozenset({"pear", "rectangle"}),
    "rectangle":         frozenset({"hourglass", "triangle", "trapezoid", "inverted_triangle"}),
    "inverted_triangle": frozenset({"trapezoid", "rectangle"}),
    "apple":             frozenset({"rectangle", "oval"}),
    "trapezoid":         frozenset({"rectangle", "inverted_triangle"}),
    "oval":              frozenset({"apple", "rectangle"}),
}


def _item_key(item_id: int, size: str) -> str:
    return f"{item_id}:{size}"


def _load_body_types(item_ids: list[int]) -> dict[int, frozenset[str]]:
    """Batch-fetch body_type column from clothing.db for the given item ids."""
    if not item_ids or not _DB_PATH.exists():
        return {}

    placeholders = ",".join("?" * len(item_ids))
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        rows = conn.execute(
            f"SELECT id, body_type FROM items WHERE id IN ({placeholders})",
            item_ids,
        ).fetchall()
        conn.close()
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


def _score(detected: str, item_shapes: frozenset[str]) -> float:
    if not item_shapes:
        return 0.20
    if detected in item_shapes:
        return 1.0
    if item_shapes & _ADJACENT.get(detected, frozenset()):
        return 0.55
    return 0.0


class BodyScoreBehaviour(CyclicBehaviour):
    async def run(self) -> None:
        msg = await self.receive(timeout=60)
        if msg is None:
            return

        data            = parse(msg)
        conv_id         = data.get("conv_id", "")
        candidates_info = data.get("candidates", [])
        context         = data.get("context", {})
        detected        = str(context.get("detected_body_type") or "").lower().strip()

        item_ids = [int(c["item_id"]) for c in candidates_info]

        loop = asyncio.get_event_loop()
        body_types: dict[int, frozenset] = await loop.run_in_executor(
            None, lambda: _load_body_types(item_ids)
        )

        scores: dict[str, float] = {}
        for c in candidates_info:
            iid = int(c["item_id"])
            sz  = c["size"]
            if detected:
                raw = _score(detected, body_types.get(iid, frozenset()))
            else:
                raw = 0.5   # no body type context → neutral
            scores[_item_key(iid, sz)] = round(raw, 6)

        propose = make_propose(
            to_jid   = str(msg.sender),
            conv_id  = conv_id,
            agent_id = "body",
            scores   = scores,
        )
        await self.send(propose)
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
        logger.info("BodyRecommenderAgent ready.")
