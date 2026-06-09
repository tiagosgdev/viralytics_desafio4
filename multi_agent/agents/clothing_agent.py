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

from multi_agent.agents.base import BaseRecommenderAgent
from multi_agent.messages import parse, make_propose, comm_log, CFP

logger = logging.getLogger(__name__)


def _item_key(item_id: int, size: str) -> str:
    return f"{item_id}:{size}"


# Axes to skip in clothing scoring — body_type is handled with a proper
# per-item DB lookup by BodyRecommenderAgent; penalising clothing candidates
# for a field that isn't in their CFP payload deflates all scores uniformly.
_SKIP_AXES: frozenset[str] = frozenset({"body_type"})


def _score_candidates(
    candidates_info: list[dict],
    filters: dict,
) -> dict[str, float]:
    """
    Match-count score: how many include axes does each item satisfy?
    Normalised by the number of active filter axes so the result is in [0, 1].
    """
    raw_include: dict[str, list[str]] = (
        (filters.get("filters") or {}).get("include") or {}
    )
    # Drop axes this agent doesn't evaluate (body_type → BodyRecommenderAgent).
    include = {k: v for k, v in raw_include.items() if k not in _SKIP_AXES}

    # No meaningful filters → uniform score of 0.5
    if not include:
        return {_item_key(c["item_id"], c["size"]): 0.5 for c in candidates_info}

    n_axes = len(include)
    scores: dict[str, float] = {}

    for c in candidates_info:
        hits = 0
        for key, values in include.items():
            item_val = str(c.get(key) or "")
            if key == "age_group":
                if any(v.lower() in item_val.lower() for v in values):
                    hits += 1
            elif item_val in values:
                hits += 1
        scores[_item_key(c["item_id"], c["size"])] = round(hits / n_axes, 6)

    return scores


class ClothingScoreBehaviour(CyclicBehaviour):
    async def run(self) -> None:
        msg = await self.receive(timeout=60)
        if msg is None:
            return

        data            = parse(msg)
        conv_id         = data.get("conv_id", "")
        candidates_info = data.get("candidates", [])
        weights_result  = data.get("weights_result", {})

        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: _score_candidates(candidates_info, weights_result),
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
