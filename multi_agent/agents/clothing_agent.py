"""
ClothingRecommenderAgent
────────────────────────
Receives a CFP from the Orchestrator with 40 candidate items.
Scores each item by how many of the DB include-filters (from
feature_weighting.analyze_intent) it satisfies, normalised to [0, 1].

This captures the garment-type / style / occasion / season match — the
"clothing intent" dimension of the user's request.
"""

import asyncio
import logging

from spade.behaviour import CyclicBehaviour
from spade.template import Template

from multi_agent import config
from multi_agent.agents.base import BaseRecommenderAgent
from multi_agent.messages import parse, make_propose, comm_log, CFP
from multi_agent.strategies.registry import get_strategy

logger = logging.getLogger(__name__)


class ClothingScoreBehaviour(CyclicBehaviour):
    async def run(self) -> None:
        msg = await self.receive(timeout=60)
        if msg is None:
            return

        data            = parse(msg)
        conv_id         = data.get("conv_id", "")
        candidates_info = data.get("candidates", [])
        context         = data.get("context", {})
        weights_result  = data.get("weights_result", {})

        strategy_name = config.AGENT_STRATEGIES["clothing"]
        score_fn, params = get_strategy("clothing", strategy_name)
        params.update(config.AGENT_STRATEGY_PARAMS.get("clothing", {}))

        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: score_fn(candidates_info, context, weights_result, params),
        )

        propose = make_propose(
            to_jid   = str(msg.sender),
            conv_id  = conv_id,
            agent_id = "clothing",
            scores   = scores,
        )
        await self.send(propose)
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        top_str = "  ".join(f"{k}={v:.2f}" for k, v in top3)
        comm_log("clothing", "orchestrator", "PROPOSE", conv_id,
                 f"{len(scores)} scores  ▸ {top_str}")
        logger.info(f"[{conv_id}] ClothingAgent PROPOSE sent ({len(scores)} items).")


class ClothingRecommenderAgent(BaseRecommenderAgent):
    async def setup(self) -> None:
        template = Template()
        template.set_metadata("performative", CFP)
        self.add_behaviour(ClothingScoreBehaviour(), template)
        from multi_agent.history import history
        summary = history.agent_context_summary("clothing")
        if summary:
            logger.info(summary)
        logger.info("ClothingRecommenderAgent ready.")
