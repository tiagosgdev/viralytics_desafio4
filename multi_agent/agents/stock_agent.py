"""
StockRecommenderAgent
─────────────────────
Receives a CFP from the Orchestrator with 40 candidate items.
Scores each item by its normalised push_score (inventory health signal) and
returns a sealed PROPOSE.

push_score combines: stock age, days since last sale, sales velocity, and
stock count — precomputed by StockStats. A high push_score means the store
wants to shift this item. Values are normalised to [0, 1] across the 40
candidates before submission.
"""

import asyncio
import logging
import sys
from pathlib import Path

from spade.behaviour import CyclicBehaviour
from spade.template import Template

from multi_agent.agents.base import BaseRecommenderAgent
from multi_agent.messages import parse, make_propose, comm_log, CFP

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STOCK_DIR = _REPO_ROOT / "stock_agent"
if str(_STOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_STOCK_DIR))

from stock_stats import StockStats  # noqa: E402

logger = logging.getLogger(__name__)


def _item_key(item_id: int, size: str) -> str:
    return f"{item_id}:{size}"


class StockScoreBehaviour(CyclicBehaviour):
    async def run(self) -> None:
        msg = await self.receive(timeout=60)
        if msg is None:
            return

        data            = parse(msg)
        conv_id         = data.get("conv_id", "")
        candidates_info = data.get("candidates", [])

        pairs = [(int(c["item_id"]), c["size"]) for c in candidates_info]

        loop = asyncio.get_event_loop()
        try:
            raw_scores: dict = await loop.run_in_executor(
                None,
                lambda: self.agent._stats.get_push_scores(pairs, missing=0.0),
            )
        except Exception as exc:
            logger.error(f"[{conv_id}] Stock scoring failed: {exc}")
            raw_scores = {pair: 0.0 for pair in pairs}

        # Normalise to [0, 1] across this candidate set
        values = list(raw_scores.values())
        lo, hi = min(values, default=0.0), max(values, default=1.0)
        span   = (hi - lo) or 1.0
        scores = {
            _item_key(iid, sz): round((v - lo) / span, 6)
            for (iid, sz), v in raw_scores.items()
        }

        propose = make_propose(
            to_jid   = str(msg.sender),
            conv_id  = conv_id,
            agent_id = "stock",
            scores   = scores,
        )
        await self.send(propose)
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        top_str = "  ".join(f"{k}={v:.2f}" for k, v in top3)
        comm_log("stock", "orchestrator", "PROPOSE", conv_id,
                 f"{len(scores)} scores  ▸ {top_str}")
        logger.info(f"[{conv_id}] StockAgent PROPOSE sent ({len(scores)} items).")


class StockRecommenderAgent(BaseRecommenderAgent):
    def __init__(self, jid: str, password: str) -> None:
        super().__init__(jid, password)
        self._stats: StockStats | None = None

    async def setup(self) -> None:
        loop = asyncio.get_event_loop()
        self._stats = await loop.run_in_executor(None, StockStats)
        template = Template()
        template.set_metadata("performative", CFP)
        self.add_behaviour(StockScoreBehaviour(), template)
        logger.info("StockRecommenderAgent ready.")
