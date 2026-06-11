"""Central configuration for the SPADE multi-agent recommendation system."""

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

# Budget split between inventory health and user-preference signals
STOCK_WEIGHT      = 0.20  # fixed weight for the StockAgent (inventory pressure)
USER_WEIGHT       = 0.80  # split among body/clothing/colour by feature importances

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
