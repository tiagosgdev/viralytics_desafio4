"""
multi_agent/run.py
──────────────────
RecommendationSystem: starts all six SPADE agents, runs recommendation rounds,
and stops cleanly on shutdown.

Usage — embedded in the FastAPI app (persistent mode):
    system = RecommendationSystem()
    await system.start()                # once, on app startup
    top10  = await system.recommend(…)  # per user request
    await system.stop()                 # once, on app shutdown

Usage — standalone CLI demo:
    python -m multi_agent.run

XMPP broker must be running first:
    docker compose up -d xmpp
"""

import asyncio
import logging
from typing import Optional

import spade

from multi_agent.agents.body_agent     import BodyRecommenderAgent
from multi_agent.agents.clothing_agent import ClothingRecommenderAgent
from multi_agent.agents.colour_agent   import ColourRecommenderAgent
from multi_agent.agents.orchestrator   import OrchestratorAgent
from multi_agent.agents.stock_agent    import StockRecommenderAgent
from multi_agent.agents.weight_agent   import FeatureWeightAgent
from multi_agent.config import JIDS, ROUND_TIMEOUT_S, TOP_K, XMPP_PASSWORD

logger = logging.getLogger(__name__)


class RecommendationSystem:
    """
    Lifecycle wrapper for all six SPADE agents.

    Agents are started once and kept alive for the lifetime of the process.
    Rounds are serialised through the OrchestratorAgent's internal asyncio.Queue,
    so multiple concurrent recommend() calls are safely queued.
    """

    def __init__(self) -> None:
        self._orchestrator: Optional[OrchestratorAgent] = None
        self._agents: list = []
        self._started = False

    async def start(self) -> None:
        """Start all agents and register them with the XMPP broker."""
        if self._started:
            return

        # Scorer agents must be up before the Orchestrator sends CFPs.
        # FeatureWeightAgent must be up before Orchestrator sends REQUEST.
        # We start them all and rely on SPADE's async connection setup.
        weight_agent   = FeatureWeightAgent(JIDS["weights"],      XMPP_PASSWORD)
        body_agent     = BodyRecommenderAgent(JIDS["body"],       XMPP_PASSWORD)
        clothing_agent = ClothingRecommenderAgent(JIDS["clothing"], XMPP_PASSWORD)
        colour_agent   = ColourRecommenderAgent(JIDS["colour"],   XMPP_PASSWORD)
        stock_agent    = StockRecommenderAgent(JIDS["stock"],     XMPP_PASSWORD)
        orchestrator   = OrchestratorAgent(JIDS["orchestrator"],  XMPP_PASSWORD)

        self._agents = [
            weight_agent, body_agent, clothing_agent,
            colour_agent, stock_agent, orchestrator,
        ]

        for agent in self._agents:
            await agent.start(auto_register=True)
            logger.info(f"Agent started: {agent.jid}")

        self._orchestrator = orchestrator
        self._started = True
        logger.info("RecommendationSystem: all agents online.")

    async def recommend(
        self,
        *,
        detected_color:     str = "",
        detected_type:      str = "",
        detected_body_type: str = "",
        user_answer:        str = "",
        user_gender:        str = "",
        user_height_cm:     float | None = None,
    ) -> list[dict]:
        """
        Run one recommendation round and return the top-10 list.

        Each dict in the result contains:
            rank, item_id, size, color, type, fit, season, style, pattern,
            material, gender, age_group, occasion, brand, price, stock_count,
            push_score, agent_scores {body, clothing, colour, stock},
            agent_weights {body, clothing, colour, stock}
        """
        if not self._started:
            raise RuntimeError("Call start() before recommend().")

        loop          = asyncio.get_event_loop()
        result_future = loop.create_future()

        self._orchestrator.trigger_round(
            detected_color     = detected_color,
            detected_type      = detected_type,
            detected_body_type = detected_body_type,
            user_answer        = user_answer,
            user_gender        = user_gender,
            user_height_cm     = user_height_cm,
            result_future      = result_future,
        )

        return await asyncio.wait_for(result_future, timeout=ROUND_TIMEOUT_S)

    async def stop(self) -> None:
        """Stop all agents gracefully."""
        for agent in reversed(self._agents):
            try:
                await agent.stop()
            except Exception as exc:
                logger.warning(f"Error stopping {agent.jid}: {exc}")
        self._agents.clear()
        self._orchestrator = None
        self._started = False
        logger.info("RecommendationSystem: all agents stopped.")

    @property
    def is_ready(self) -> bool:
        return self._started


# ── CLI demo ──────────────────────────────────────────────────────────────────

async def _cli_demo() -> None:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    system = RecommendationSystem()

    print("\n━━  Viralytics Multi-Agent Recommendation Demo  ━━")
    print("(XMPP broker must be running: docker compose up -d xmpp)\n")

    print("Starting agents…", flush=True)
    await system.start()
    print("All agents online.\n")

    color     = input("Detected color      (e.g. red)              > ").strip() or "red"
    type_     = input("Detected type       (e.g. short_sleeve_top) > ").strip() or "short_sleeve_top"
    body_type = input("Detected body type  (e.g. hourglass)        > ").strip() or "hourglass"
    gender    = input("User gender         (male / female / blank) > ").strip() or ""
    answer    = input("User intent answer  (free text)             > ").strip() or "I want something casual"

    print("\nRunning sealed-bid round…", flush=True)
    try:
        results = await system.recommend(
            detected_color     = color,
            detected_type      = type_,
            detected_body_type = body_type,
            user_answer        = answer,
            user_gender        = gender,
        )
        print(f"\nTop {TOP_K} recommendations:\n")
        for item in results:
            w = item.get("agent_weights", {})
            s = item.get("agent_scores",  {})
            print(
                f"  {item['rank']:>2}. id={item['item_id']:>5}  size={item['size']:<3}  "
                f"color={item.get('color','?'):<12}  type={item.get('type','?'):<22}\n"
                f"      body={s.get('body',0):.2f}(w={w.get('body',0):.2f})  "
                f"cloth={s.get('clothing',0):.2f}(w={w.get('clothing',0):.2f})  "
                f"colour={s.get('colour',0):.2f}(w={w.get('colour',0):.2f})  "
                f"stock={s.get('stock',0):.2f}(w={w.get('stock',0):.2f})"
            )
    except asyncio.TimeoutError:
        print(f"Timeout: round did not complete within {ROUND_TIMEOUT_S}s.")
    except Exception as exc:
        print(f"Error: {exc}")

    await system.stop()
    print("\nAgents stopped.")


if __name__ == "__main__":
    spade.run(_cli_demo())
