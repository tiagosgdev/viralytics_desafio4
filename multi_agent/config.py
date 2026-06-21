"""Central configuration for the SPADE multi-agent recommendation system."""

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

# Budget split between inventory health and user-preference signals
STOCK_WEIGHT      = 0.20  # fixed weight for the StockAgent (inventory pressure)
RL_WEIGHT         = 0.15  # fixed weight for the RLRecommenderAgent (learned signal)
USER_WEIGHT       = 0.80  # split among body/clothing/colour by feature importances
                          # (note: the effective user budget is 1 - STOCK_WEIGHT
                          #  - RL_WEIGHT when both fixed-slice agents are present)

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
