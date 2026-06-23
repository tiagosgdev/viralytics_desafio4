"""
multi_agent.rl
──────────────
Reinforcement-learning support for the RLRecommenderAgent (PPO):

  - policy.py : PPOPolicy — a PyTorch actor-critic trained with Proximal Policy
                Optimisation; the agent's learnable, persisted "brain".
  - store.py  : RLRoundStore — on-policy transition cache + rollout batching,
                plus the two reward functions (pass-rate and emoji satisfaction).

Both expose a process-level singleton (`rl_policy`, `rl_store`) shared by the
SPADE agent (acting + PPO updates) and the API feedback path (emoji rewards).
"""

from multi_agent.rl.policy import rl_policy, PPOPolicy, FEATURE_NAMES, extract_features
from multi_agent.rl.store import (
    rl_store,
    RLRoundStore,
    Transition,
    passrate_reward,
    rating_reward,
)

__all__ = [
    "rl_policy",
    "PPOPolicy",
    "FEATURE_NAMES",
    "extract_features",
    "rl_store",
    "RLRoundStore",
    "Transition",
    "passrate_reward",
    "rating_reward",
]
