"""
ColourRecommenderAgent  (stub implementation)
──────────────────────────────────────────────
Receives a CFP from the Orchestrator with 40 candidate items.
Scores each item by colour compatibility with the detected garment colour.

Current implementation: rule-based colour compatibility matrix.
Future implementation: replace with a colour-wheel embedding model or
a trained colour-harmony classifier.

Score values:
  1.0  — exact colour match (user confirmed / detected colour)
  0.65 — compatible colour (adjacent on the compatibility matrix)
  0.20 — unrelated colour
  0.50 — no detected colour (context unknown → neutral)
"""

import asyncio
import logging

from spade.behaviour import CyclicBehaviour
from spade.template import Template

from multi_agent.agents.base import BaseRecommenderAgent
from multi_agent.messages import parse, make_propose, comm_log, CFP

logger = logging.getLogger(__name__)

# ── Colour compatibility matrix ───────────────────────────────────────────────
# Symmetric: if A is compatible with B, B is compatible with A.
_COMPATIBLE: dict[str, frozenset[str]] = {
    "black":     frozenset({"white", "gray", "red", "blue", "yellow", "green", "purple", "orange", "beige", "cream", "pink"}),
    "white":     frozenset({"black", "navy", "blue", "red", "green", "gray", "pink", "beige", "burgundy", "olive"}),
    "gray":      frozenset({"white", "black", "navy", "blue", "red", "pink", "purple", "yellow"}),
    "navy":      frozenset({"white", "beige", "gray", "red", "yellow", "cream", "burgundy"}),
    "blue":      frozenset({"white", "gray", "beige", "navy", "brown", "yellow", "orange"}),
    "red":       frozenset({"white", "black", "gray", "navy", "beige", "pink"}),
    "green":     frozenset({"white", "beige", "brown", "navy", "black", "cream", "olive"}),
    "yellow":    frozenset({"white", "black", "navy", "gray", "blue", "brown"}),
    "orange":    frozenset({"white", "black", "navy", "brown", "blue"}),
    "pink":      frozenset({"white", "gray", "navy", "black", "beige", "cream"}),
    "purple":    frozenset({"white", "black", "gray", "beige", "pink"}),
    "brown":     frozenset({"white", "beige", "green", "navy", "cream", "yellow", "orange"}),
    "beige":     frozenset({"white", "brown", "navy", "blue", "green", "black", "pink"}),
    "cream":     frozenset({"brown", "navy", "black", "beige", "green", "burgundy"}),
    "burgundy":  frozenset({"white", "beige", "gray", "black", "navy", "cream"}),
    "olive":     frozenset({"white", "beige", "brown", "black", "cream"}),
    "teal":      frozenset({"white", "beige", "gray", "navy", "black", "coral"}),
    "coral":     frozenset({"white", "beige", "navy", "gray", "teal"}),
    "multicolor": frozenset({"black", "white", "navy", "gray"}),
}


def _item_key(item_id: int, size: str) -> str:
    return f"{item_id}:{size}"


def _colour_score(detected: str, item_color: str) -> float:
    if not detected:
        return 0.5
    detected   = detected.lower().strip()
    item_color = (item_color or "").lower().strip()
    if item_color == detected:
        return 1.0
    if item_color in _COMPATIBLE.get(detected, frozenset()):
        return 0.65
    return 0.20


class ColourScoreBehaviour(CyclicBehaviour):
    async def run(self) -> None:
        msg = await self.receive(timeout=60)
        if msg is None:
            return

        data            = parse(msg)
        conv_id         = data.get("conv_id", "")
        candidates_info = data.get("candidates", [])
        context         = data.get("context", {})

        # Prefer the weight-refined colour if available, else fall back to detected
        weights_result = data.get("weights_result", {})
        w_include      = ((weights_result.get("filters") or {}).get("include") or {})
        colour_vals    = w_include.get("color") or []
        detected = (
            colour_vals[0]
            if colour_vals
            else str(context.get("detected_color") or "").lower().strip()
        )

        scores: dict[str, float] = {
            _item_key(c["item_id"], c["size"]): _colour_score(detected, c.get("color", ""))
            for c in candidates_info
        }

        propose = make_propose(
            to_jid   = str(msg.sender),
            conv_id  = conv_id,
            agent_id = "colour",
            scores   = scores,
        )
        await self.send(propose)
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
        logger.info("ColourRecommenderAgent ready.")
