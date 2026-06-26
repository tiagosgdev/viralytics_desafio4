"""
StockRecommenderAgent
─────────────────────
Receives a CFP from the Orchestrator with 40 candidate items.
Scores each item by its normalised push_score (inventory health signal) and
returns a sealed PROPOSE.

push_score combines: stock age, days since last sale, sales velocity, and
stock count — precomputed by StockStats. A high push_score means the store
wants to shift this item.

The agent owns the StockStats IO: it fetches the raw inventory signals
(push_score, and — for the ``bestsellers`` personality — sales_velocity) and
enriches the candidate dicts, then hands them to a swappable scoring strategy
(personality) selected via ``AGENT_STRATEGIES["stock"]``. The pure strategy
min-max normalises the chosen signal to [0, 1] across the candidate set; it
never calls StockStats. The default ``push`` strategy reproduces the historic
normalised-push_score behaviour exactly.
"""

import asyncio
import logging
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
_STOCK_DIR = _REPO_ROOT / "stock_agent"
if str(_STOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_STOCK_DIR))

from stock_stats import StockStats  # noqa: E402

logger = logging.getLogger(__name__)


def _enrich(stats: "StockStats", candidates_info: list[dict], signal: str) -> None:
    """Attach the raw inventory signal each strategy scores on (IO lives here).

    Always sets ``push_score`` (missing → 0.0) for the baseline ``push`` strategy.
    When the active strategy scores on a different ``signal`` — ``sales_velocity``
    (bestsellers) or ``stock_count`` (overstock_aggressive) — that field is fetched
    per-row from StockStats. The stock CFP payload only carries ``item_id``/``size``
    (orchestrator ``_CFP_FIELDS["stock"]``), so these heavier signals must be fetched
    here rather than read off the candidate dicts.
    """
    pairs = [(int(c["item_id"]), c["size"]) for c in candidates_info]
    try:
        push: dict = stats.get_push_scores(pairs, missing=0.0)
    except Exception as exc:
        logger.error(f"Stock push_score lookup failed: {exc}")
        push = {pair: 0.0 for pair in pairs}

    need_row = signal not in ("push_score", "")
    for c in candidates_info:
        iid, sz = int(c["item_id"]), c["size"]
        c["push_score"] = push.get((iid, sz), 0.0)
        if need_row and signal not in c:
            try:
                c[signal] = float(stats.get_row(iid, sz)[signal])
            except Exception:
                c[signal] = 0.0


class StockScoreBehaviour(CyclicBehaviour):
    async def run(self) -> None:
        msg = await self.receive(timeout=60)
        if msg is None:
            return

        data            = parse(msg)
        conv_id         = data.get("conv_id", "")
        candidates_info = data.get("candidates", [])
        context         = data.get("context", {})
        weights_result  = data.get("weights_result", {})

        strategy_name = config.AGENT_STRATEGIES["stock"]
        score_fn, params = get_strategy("stock", strategy_name)
        params.update(config.AGENT_STRATEGY_PARAMS.get("stock", {}))
        signal = params.get("signal", "push_score")

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: _enrich(self.agent._stats, candidates_info, signal),
            )
        except Exception as exc:
            logger.error(f"[{conv_id}] Stock scoring failed: {exc}")
            for c in candidates_info:
                c.setdefault("push_score", 0.0)

        scores = score_fn(candidates_info, context, weights_result, params)

        # Derive the veto set from the existing scores (default -1.0 vetoes nothing).
        veto_threshold = float(params.get("veto_threshold", -1.0))
        vetoes = [k for k, v in scores.items() if v < veto_threshold]

        propose = make_propose(
            to_jid   = str(msg.sender),
            conv_id  = conv_id,
            agent_id = "stock",
            scores   = scores,
            vetoes   = vetoes,
        )
        await self.send(propose)
        # Write-only per-agent memory (course requirement; never read for
        # decisions). Defensive: must not break the round if _memory is absent.
        if mem := getattr(self.agent, "_memory", None):
            mem.record(conv_id, context, scores)
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        top_str = "  ".join(f"{k}={v:.2f}" for k, v in top3)
        comm_log("stock", "orchestrator", "PROPOSE", conv_id,
                 f"{len(scores)} scores  ▸ {top_str}")
        logger.info(f"[{conv_id}] StockAgent PROPOSE sent ({len(scores)} items).")


class StockRecommenderAgent(BaseRecommenderAgent):
    def __init__(
        self,
        jid: str,
        password: str,
        stats: "StockStats | None" = None,
    ) -> None:
        super().__init__(jid, password)
        self._stats: StockStats | None = stats  # shared instance or None

    async def setup(self) -> None:
        if self._stats is None:
            loop = asyncio.get_event_loop()
            self._stats = await loop.run_in_executor(None, StockStats)
        template = Template()
        template.set_metadata("performative", CFP)
        self.add_behaviour(StockScoreBehaviour(), template)
        self._memory = AgentMemory("stock")
        summary = self._memory.summary()
        if summary:
            logger.info(summary)
        logger.info("StockRecommenderAgent ready.")
