"""Central configuration for the SPADE multi-agent recommendation system."""

import json
import os
from pathlib import Path

# ── XMPP broker ───────────────────────────────────────────────────────────────
# Prosody runs in Docker, exposed to the host on localhost:5222.
XMPP_HOST     = "localhost"
XMPP_PORT     = 5222
XMPP_PASSWORD = "viralytics_dev"   # shared secret; all agents use the same password

# ── Agent JIDs ────────────────────────────────────────────────────────────────
JIDS: dict[str, str] = {
    "orchestrator": f"orchestrator@{XMPP_HOST}",
    "body":         f"body@{XMPP_HOST}",
    "clothing":     f"clothing@{XMPP_HOST}",
    "colour":       f"colour@{XMPP_HOST}",
    "stock":        f"stock@{XMPP_HOST}",
    "weights":      f"weights@{XMPP_HOST}",
    "rl":           f"rl@{XMPP_HOST}",
}

# ── Reinforcement-learning agent ───────────────────────────────────────────────
# Master switch.  When False the system behaves exactly as before the RL agent was
# added (four scorers, no learning, weights identical to the legacy split).
RL_ENABLED        = True

# Names of the agents that score candidates and submit sealed proposals.
# The RL agent is appended only when enabled so the rest of the pipeline (CFP
# broadcast, proposal collection, Borda weighting) picks it up automatically.
SCORER_NAMES: list[str] = ["body", "clothing", "colour", "stock"]
if RL_ENABLED:
    SCORER_NAMES.append("rl")

# ── Round parameters ──────────────────────────────────────────────────────────
N_CANDIDATES      = 40    # items retrieved from DB before the debate round
TOP_K             = 10    # final recommendations returned to the user

# ── Selection mechanism (legacy Borda vs random-batch + weighted veto) ────────
# SELECTION_MODE switches the whole selection path. "borda" (default) is the
# legacy single-round retrieve→CFP→Borda behaviour, byte-identical to before
# (modulo the tie-aware Borda fix). "veto_batch" enables the new random-batch +
# weighted-veto loop. All flags are env-overridable so the experiment harness
# can A/B test without code changes.
SELECTION_MODE    = os.environ.get("SELECTION_MODE", "borda")          # "borda" | "veto_batch"

# How vetoes combine to eliminate an item. "weighted" (default): eliminate when
# Σ(weight of vetoing agents) ≥ VETO_TAU. "blackball": any single veto eliminates
# (the τ→0+ special case).
VETO_MODE         = os.environ.get("VETO_MODE", "weighted")            # "weighted" | "blackball"

# Weighted-veto elimination threshold (reject-mass cutoff).
VETO_TAU          = float(os.environ.get("VETO_TAU", "0.5"))

# Batch loop bounds (veto_batch mode): draw up to MAX_BATCHES random batches of
# BATCH_SIZE in-stock items from the broad relevance band until ≥ TOP_K survive.
MAX_BATCHES       = int(os.environ.get("MAX_BATCHES", "5"))
BATCH_SIZE        = int(os.environ.get("BATCH_SIZE", str(N_CANDIDATES)))

# Fallback emphases (NOT a hard budget split any more). The four scorer weights
# are conversation-driven: build_agent_weights normalises the four emphases
# (color/type/bodyType/stock) returned by FeatureWeightAgent across the budget
# left after the RL slice. STOCK_WEIGHT is only used as a FALLBACK stock
# importance when a caller omits the `stock` emphasis; USER_WEIGHT is kept for
# backward-compatibility / reference only.
STOCK_WEIGHT      = 0.20  # fallback stock importance (when no stock emphasis supplied)
USER_WEIGHT       = 0.80  # legacy reference; no longer a fixed user-preference budget

# RL_WEIGHT is the ONE fixed-slice budget: it is carved off the top for the RL
# agent's learned signal, and the four conversation emphases share the remaining
# 1 - RL_WEIGHT. 0 (or RL_ENABLED=False) omits the RL agent entirely.
RL_WEIGHT         = 0.15  # fixed weight for the RLRecommenderAgent (learned signal)

# ── PPO hyperparameters ──────────────────────────────────────────────────────────
# The RL agent is a PyTorch actor-critic trained with Proximal Policy Optimisation.
PPO_HIDDEN         = 32      # hidden units per layer in the shared actor-critic trunk
PPO_LR             = 3e-4    # Adam learning rate
PPO_CLIP_EPS       = 0.2     # PPO clipped-surrogate ε (trust-region width)
PPO_EPOCHS         = 4       # optimisation passes over each collected rollout
PPO_MINIBATCH      = 64      # minibatch size within an epoch
PPO_VALUE_COEF     = 0.5     # weight of the critic (value) loss
PPO_ENTROPY_COEF   = 0.01    # weight of the entropy bonus (drives exploration)
PPO_MAX_GRAD_NORM  = 0.5     # global gradient-norm clip
PPO_INIT_LOG_STD   = -0.5    # initial log σ of the Gaussian action head (σ ≈ 0.61)
PPO_ROLLOUT_ROUNDS = 8       # rounds collected before each on-policy PPO update

# Per-item reward shaping (see rl/store.py)
PASSRATE_ANCHOR   = 0.10  # fraction of the agent's picks that must advance for reward 0
ZERO_PASS_PENALTY = -1.5  # extra-negative reward when 0% of the agent's picks advance

# ── RL persistence ─────────────────────────────────────────────────────────────────
# ONE global policy, shared by every customer.  Checkpointed (network + optimizer
# state) and reloaded on startup so the learned pattern is never lost on restart.
_REPO_ROOT         = Path(__file__).resolve().parent.parent
RL_CHECKPOINT_PATH = _REPO_ROOT / "models" / "weights" / "agents" / "rl_ppo.pt"
RL_ROUND_CACHE     = 200  # recent rounds kept in the in-memory transition cache

# Timeouts (seconds)
WEIGHTS_TIMEOUT_S = 120   # max wait for FeatureWeightAgent INFORM reply
                           # (fast-path ~0 ms; Ollama LLM path ~48 s on CPU)
COLLECT_TIMEOUT_S = 60    # max wait to collect all sealed proposals
ROUND_TIMEOUT_S   = 200   # total round deadline (WEIGHTS + COLLECT + margin)
QUEUE_TTL_S       = 60    # max time a round may sit in the orchestrator queue;
                           # older rounds are dropped (user has likely moved on)

# Back-pressure: maximum number of rounds that may sit in the orchestrator
# queue at once.  Rounds beyond this cap are dropped immediately and return [].
QUEUE_MAX_SIZE    = 10

# ── Agent personalities (strategies) ──────────────────────────────────────────
# Each scorer agent resolves its scoring "personality" from
# multi_agent.strategies.registry. The defaults below are the BASELINE strategies
# and reproduce the pre-refactor scoring byte-for-byte. Each is env-overridable
# via AGENT_STRATEGY_<NAME> (e.g. AGENT_STRATEGY_COLOUR=adventurous) so the
# experiment harness (and ad-hoc runs) can mix-and-match without code changes.
_DEFAULT_AGENT_STRATEGIES: dict[str, str] = {
    "colour":   "purist",
    "body":     "strict",
    "clothing": "match_count",
    "stock":    "push",
}

AGENT_STRATEGIES: dict[str, str] = {
    agent: os.environ.get(f"AGENT_STRATEGY_{agent.upper()}", default)
    for agent, default in _DEFAULT_AGENT_STRATEGIES.items()
}

# Optional per-agent param overrides layered on top of the strategy's defaults.
# Supplied as a JSON object via AGENT_STRATEGY_PARAMS, e.g.
#   AGENT_STRATEGY_PARAMS='{"colour": {"unrelated": 0.1}}'
# An empty / unset / malformed value leaves all strategies on their defaults.
def _load_strategy_params() -> dict[str, dict]:
    raw = os.environ.get("AGENT_STRATEGY_PARAMS", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


AGENT_STRATEGY_PARAMS: dict[str, dict] = _load_strategy_params()
