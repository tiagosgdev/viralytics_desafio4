"""
FeatureWeightAgent
──────────────────
Receives a REQUEST from the Orchestrator containing the round context
(detected_color, detected_type, detected_body_type, user_answer), calls
feature_weighting.analyze_intent() in a thread pool, and sends an INFORM
back with the resulting weights + DB filters.

The weights (color / type / bodyType / stock importances summing to 100) are
used by the Orchestrator to derive the four scorer-agent weights — each agent
(colour / clothing / body / stock) gets a share proportional to its emphasis.
"""

import asyncio
import logging
import sys
from pathlib import Path

from spade.behaviour import CyclicBehaviour
from spade.template import Template

from multi_agent.agents.base import BaseRecommenderAgent
from multi_agent.messages import parse, make_inform, comm_log, REQUEST

# ── path setup so feature_weighting can import its own sub-modules ────────────
_REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
_LNIAGIA_DIR  = _REPO_ROOT / "LNIAGIA"
_QP_DIR       = _LNIAGIA_DIR / "query_parsing"
for _p in (_LNIAGIA_DIR, _QP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from query_parsing.feature_weighting import analyze_intent  # noqa: E402

logger = logging.getLogger(__name__)

# Default four-way emphasis used whenever no LLM signal is available. The three
# user-facing features sit slightly above stock (the shopper's explicit garment /
# colour / fit intent should usually outweigh a generic inventory push), and all
# four are > 0 and sum to 100 so build_agent_weights normalises them cleanly.
_FALLBACK_WEIGHTS = {
    "color":    {"importance": 30},
    "type":     {"importance": 30},
    "bodyType": {"importance": 25},
    "stock":    {"importance": 15},
}


def _rule_based_weights(ctx: dict) -> dict:
    """
    Deterministic weights for rounds where user_answer is empty.

    When the user has not answered an intent question yet, the three detected
    features are treated as equally important.  This path takes microseconds
    and avoids an unnecessary Ollama call.
    """
    detected_color     = str(ctx.get("detected_color")     or "").strip()
    detected_type      = str(ctx.get("detected_type")      or "").strip()
    detected_body_type = str(ctx.get("detected_body_type") or "").strip()

    include: dict = {}
    if detected_color:
        include["color"] = [detected_color]
    if detected_type:
        include["type"] = [detected_type]
    if detected_body_type:
        include["body_type"] = [detected_body_type]

    query = " ".join(filter(None, [detected_color, detected_type])) or "clothing"
    return {
        "query":   query,
        "filters": {"include": include, "exclude": {}},
        "weights": {
            "color":    {"importance": 30},
            "type":     {"importance": 30},
            "bodyType": {"importance": 25},
            "stock":    {"importance": 15},
        },
    }


class WeightBehaviour(CyclicBehaviour):
    async def run(self) -> None:
        msg = await self.receive(timeout=60)
        if msg is None:
            return

        data    = parse(msg)
        conv_id = data.get("conv_id", "")
        ctx     = data.get("context", {})

        user_answer = str(ctx.get("user_answer") or "").strip()

        if not user_answer:
            # Fast path: no user text to interpret — deterministic rules, ~0 ms.
            result = _rule_based_weights(ctx)
            logger.info(f"[{conv_id}] FeatureWeightAgent: fast-path (no user answer).")
        else:
            # LLM path: interpret the user's text to derive importance weights.
            logger.info(f"[{conv_id}] FeatureWeightAgent: calling analyze_intent.")
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: analyze_intent(
                        detected_color     = str(ctx.get("detected_color",     "") or ""),
                        detected_type      = str(ctx.get("detected_type",      "") or ""),
                        detected_body_type = str(ctx.get("detected_body_type", "") or ""),
                        user_answer        = user_answer,
                    ),
                )
            except Exception as exc:
                logger.error(f"[{conv_id}] analyze_intent failed: {exc}")
                result = {
                    "query":   str(ctx.get("detected_type") or "clothing"),
                    "filters": {"include": {}, "exclude": {}},
                    "weights": _FALLBACK_WEIGHTS,
                }

        reply = make_inform(
            to_jid  = str(msg.sender),
            conv_id = conv_id,
            payload = result,
        )
        await self.send(reply)
        w = result.get("weights", {})
        comm_log(
            "weights", "orchestrator", "INFORM", conv_id,
            f"color={w.get('color',{}).get('importance')}%  "
            f"type={w.get('type',{}).get('importance')}%  "
            f"body={w.get('bodyType',{}).get('importance')}%  "
            f"stock={w.get('stock',{}).get('importance')}%",
        )
        logger.info(f"[{conv_id}] Weights sent: {w}")


class FeatureWeightAgent(BaseRecommenderAgent):
    async def setup(self) -> None:
        template = Template()
        template.set_metadata("performative", REQUEST)
        self.add_behaviour(WeightBehaviour(), template)
        logger.info("FeatureWeightAgent ready.")
