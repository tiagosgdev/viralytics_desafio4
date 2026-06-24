"""
RLRecommenderAgent  (PPO)
─────────────────────────
A 5th sealed-bid scorer that *learns* with Proximal Policy Optimisation.  It plugs
into the same CFP → PROPOSE → Borda flow as the deterministic scorers, but its
scores are sampled from a PyTorch actor-critic policy (multi_agent/rl/policy.py).
It improves that policy from two reward signals:

  1. Pass-rate (automatic, every round) — of the agent's own top-K picks, how many
     reached the final top-K.  Anchored at 10% (see store.passrate_reward).
  2. Emoji satisfaction (delayed) — applied outside this agent, in
     RecommendationSystem.submit_feedback(), because the rating arrives over HTTP.

Both rewards are accumulated onto the round's transitions; once
PPO_ROLLOUT_ROUNDS rounds have settled, the collected on-policy batch is handed
to the policy for one PPO update.

Two behaviours:
  - RLScoreBehaviour     (template: CFP)    — sample scores, cache transitions, PROPOSE
  - RoundResultBehaviour (template: INFORM) — settle pass-rate; trigger PPO update
"""

import logging

from spade.behaviour import CyclicBehaviour
from spade.template import Template

from multi_agent.agents.base import BaseRecommenderAgent
from multi_agent.messages import CFP, INFORM, comm_log, make_propose, parse
from multi_agent.rl.policy import extract_features, rl_policy
from multi_agent.rl.store import rl_store

logger = logging.getLogger(__name__)


class RLScoreBehaviour(CyclicBehaviour):
    """Sealed-bid scoring: identical contract to the other scorer agents."""

    async def run(self) -> None:
        msg = await self.receive(timeout=60)
        if msg is None:
            return

        data            = parse(msg)
        conv_id         = data.get("conv_id", "")
        candidates_info = data.get("candidates", [])
        context         = data.get("context", {})

        if not candidates_info:
            return

        # Forward pass + sample (sampling = exploration); cache the transitions so
        # the reward passes can run PPO over them later.
        features = extract_features(candidates_info, context)
        scores, transitions = rl_policy.act(features)
        rl_store.record_transitions(conv_id, transitions)

        propose = make_propose(
            to_jid   = str(msg.sender),
            conv_id  = conv_id,
            agent_id = "rl",
            scores   = scores,
        )
        await self.send(propose)

        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
        top_str = "  ".join(f"{k}={v:.2f}" for k, v in top3)
        comm_log("rl", "orchestrator", "PROPOSE", conv_id,
                 f"{len(scores)} scores  ▸ {top_str}  (σ={rl_policy.snapshot()['sigma']})")
        logger.info(f"[{conv_id}] RLAgent PROPOSE sent ({len(scores)} items).")


class RoundResultBehaviour(CyclicBehaviour):
    """
    Listens for the orchestrator's end-of-round INFORM (the final top-K), applies
    the pass-rate reward, and triggers a PPO update once enough rounds have
    accumulated into an on-policy rollout.
    """

    async def run(self) -> None:
        msg = await self.receive(timeout=60)
        if msg is None:
            return

        data = parse(msg)
        if data.get("event") != "round_result":
            return   # not the message we care about

        conv_id    = data.get("conv_id", "")
        final_keys = set(data.get("final_keys", []))
        if not final_keys:
            return

        # Apply the pass-rate reward and, if the rollout is full, get the batch.
        batch = rl_store.settle_round(conv_id, final_keys)
        if batch:
            stats = rl_policy.learn(batch)
            logger.info(f"[{conv_id}] RLAgent PPO update: {stats}")


class RLRecommenderAgent(BaseRecommenderAgent):
    async def setup(self) -> None:
        cfp_template = Template()
        cfp_template.set_metadata("performative", CFP)
        self.add_behaviour(RLScoreBehaviour(), cfp_template)

        inform_template = Template()
        inform_template.set_metadata("performative", INFORM)
        self.add_behaviour(RoundResultBehaviour(), inform_template)

        from multi_agent.history import history
        if summary := history.agent_context_summary("rl"):
            logger.info(summary)
        logger.info(f"RLRecommenderAgent ready (PPO). policy={rl_policy.snapshot()}")
