"""
ColourRecommenderAgent  (stub implementation)
──────────────────────────────────────────────
Receives a CFP from the Orchestrator with 40 candidate items.
Scores each item by colour compatibility with the detected garment colour.

Current implementation: rule-based colour compatibility matrix, factored out
into a swappable scoring strategy (personality) selected via config
(``AGENT_STRATEGIES["colour"]``). The default ``purist`` strategy reproduces the
historic behaviour exactly:

Score values (purist / baseline):
  1.0  — exact colour match (user confirmed / detected colour)
  0.65 — compatible colour (adjacent on the compatibility matrix)
  0.20 — unrelated colour
  0.50 — no detected colour (context unknown → neutral)

Future implementation: replace with a colour-wheel embedding model or
a trained colour-harmony classifier (still as a registry strategy).
"""

import logging

from spade.behaviour import CyclicBehaviour
from spade.template import Template

from multi_agent import config
from multi_agent.agents.base import BaseRecommenderAgent
from multi_agent.memory import AgentMemory
from multi_agent.messages import parse, make_propose, comm_log, CFP
from multi_agent.strategies.colour import _resolve_detected
from multi_agent.strategies.registry import get_strategy

logger = logging.getLogger(__name__)


class ColourScoreBehaviour(CyclicBehaviour):
    async def run(self) -> None:
        msg = await self.receive(timeout=60)
        if msg is None:
            return

        data            = parse(msg)
        conv_id         = data.get("conv_id", "")
        candidates_info = data.get("candidates", [])
        context         = data.get("context", {})
        weights_result  = data.get("weights_result", {})

        strategy_name = config.AGENT_STRATEGIES["colour"]
        score_fn, params = get_strategy("colour", strategy_name)
        params.update(config.AGENT_STRATEGY_PARAMS.get("colour", {}))

        scores: dict[str, float] = score_fn(
            candidates_info, context, weights_result, params
        )
        detected = _resolve_detected(context, weights_result)

        propose = make_propose(
            to_jid   = str(msg.sender),
            conv_id  = conv_id,
            agent_id = "colour",
            scores   = scores,
        )
        await self.send(propose)
        # Write-only per-agent memory (course requirement; never read for
        # decisions). Defensive: must not break the round if _memory is absent.
        if mem := getattr(self.agent, "_memory", None):
            mem.record(conv_id, context, scores)
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        top_str = "  ".join(f"{k}={v:.2f}" for k, v in top3)
        comm_log("colour", "orchestrator", "PROPOSE", conv_id,
                 f"{len(scores)} scores  ▸ {top_str}")
        logger.info(f"[{conv_id}] ColourAgent PROPOSE sent ({len(scores)} items, detected={detected!r}).")


class ColourRecommenderAgent(BaseRecommenderAgent):
    async def setup(self) -> None:
        template = Template()
        template.set_metadata("performative", CFP)
        self.add_behaviour(ColourScoreBehaviour(), template)
        self._memory = AgentMemory("colour")
        summary = self._memory.summary()
        if summary:
            logger.info(summary)
        logger.info("ColourRecommenderAgent ready.")
