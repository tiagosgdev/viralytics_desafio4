"""Central configuration for the SPADE multi-agent recommendation system."""

import json
import os

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
}

# Names of the four agents that score candidates and submit sealed proposals
SCORER_NAMES: list[str] = ["body", "clothing", "colour", "stock"]

# ── Round parameters ──────────────────────────────────────────────────────────
N_CANDIDATES      = 40    # items retrieved from DB before the debate round
TOP_K             = 10    # final recommendations returned to the user

# Fallback emphases (NOT a hard budget split any more). All four agent weights
# are now conversation-driven: build_agent_weights normalises the four emphases
# (color/type/bodyType/stock) returned by FeatureWeightAgent. STOCK_WEIGHT is
# only used as a FALLBACK stock importance when a caller omits the `stock`
# emphasis; USER_WEIGHT is kept for backward-compatibility / reference only.
STOCK_WEIGHT      = 0.20  # fallback stock importance (when no stock emphasis supplied)
USER_WEIGHT       = 0.80  # legacy reference; no longer a fixed user-preference budget

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
