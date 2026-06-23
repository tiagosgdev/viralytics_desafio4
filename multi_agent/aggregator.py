"""
Weighted Borda Count aggregation — the game-theoretic "closed letter" mechanism.

Protocol:
  1. Each agent independently ranks all N candidates by its domain score
     (sealed bid — agents do not see each other's rankings before submitting).
  2. Borda points: rank-1 item gets N pts, rank-2 gets N-1 pts, … rank-N gets 1 pt.
  3. Each agent's Borda vector is scaled by that agent's weight.
  4. Vectors are summed → composite score per item.
  5. Top-k items by composite score are returned.

The weight split:
  - The RL agent (when enabled) receives a fixed `rl_weight` (default 0.15) for
    its learned signal, carved off the top; 0 omits it entirely (legacy behaviour).
  - The remaining budget (1 - rl_weight) is shared by the FOUR conversation-driven
    agents — colour/clothing/body/stock — each getting a share proportional to the
    corresponding emphasis returned by FeatureWeightAgent (color/type/bodyType/stock
    importances).  There is no fixed stock budget any more; a strong "popular / on
    sale / trending" intent can push the stock weight up, a very specific request
    can push it down.
"""

from __future__ import annotations


def borda_aggregate(
    proposals: dict[str, dict[str, float]],
    agent_weights: dict[str, float],
    k: int = 10,
) -> list[str]:
    """
    Aggregate sealed proposals via weighted Borda count.

    proposals     : {agent_id: {item_key: raw_score}}  — higher score = agent prefers item
    agent_weights : {agent_id: weight}                 — should sum to ~1.0
    k             : number of top items to return

    Returns a list of item_keys (f"{item_id}:{size}") sorted best-first.
    """
    all_items = {key for scores in proposals.values() for key in scores}
    if not all_items:
        return []

    n = len(all_items)
    composite: dict[str, float] = {key: 0.0 for key in all_items}

    for agent_id, scores in proposals.items():
        weight = agent_weights.get(agent_id, 0.0)
        if weight <= 0:
            continue

        # Rank items by this agent's score (descending); ties broken by item_key
        ranked = sorted(all_items, key=lambda ik: (scores.get(ik, 0.0), ik), reverse=True)

        for rank_0, item_key in enumerate(ranked):
            borda_pts = n - rank_0       # n pts for rank 1, down to 1 pt for rank n
            composite[item_key] += weight * borda_pts

    top_items = sorted(composite.keys(), key=lambda ik: composite[ik], reverse=True)
    return top_items[:k]


def build_agent_weights(
    feature_weights: dict,
    stock_weight: float = 0.20,
    rl_weight: float = 0.0,
    present_agents: frozenset[str] | None = None,
) -> dict[str, float]:
    """
    Convert FeatureWeightAgent output to per-agent budget.

    The four scorer agents are conversation-driven: each derives from one of four
    emphases returned by FeatureWeightAgent. The feature-id → agent-id mapping is:

        color    → "colour"
        type     → "clothing"
        bodyType → "body"
        stock    → "stock"

    The RL agent (when `rl_weight > 0`) is the one fixed-slice agent: it takes a
    constant `rl_weight` off the top for its learned signal, and the four emphases
    share the remaining `1 - rl_weight`.  With `rl_weight = 0` the RL agent never
    appears and the four emphases share the whole budget (legacy behaviour).

    feature_weights : {"color":    {"importance": N},
                       "type":     {"importance": N},
                       "bodyType": {"importance": N},
                       "stock":    {"importance": N}}
                      The four importances are normalised to share the emphasis
                      budget (1 - rl_weight). A missing key (e.g. an older caller
                      without a `stock` emphasis) falls back to `stock_weight` for
                      stock, so the function never crashes.
    stock_weight    : fallback stock *share* (fraction, 0.20 = 20%) used only when
                      the `feature_weights` dict carries no `stock` emphasis. It is
                      converted to a comparable importance internally; it is no
                      longer a fixed budget carve-out.
    rl_weight       : fixed budget reserved for the RL agent (learned signal),
                      carved off the top before the emphases are normalised. 0
                      disables / omits the RL agent.
    present_agents  : set of agent ids that actually responded this round.
                      When an agent is absent its weight is redistributed
                      proportionally among the agents that did respond.
                      Pass None (default) to assume all expected agents are present.
    """
    # feature-id → agent-id mapping (ordering fixed for the equal-split fallback).
    feature_to_agent = (
        ("color",    "colour"),
        ("type",     "clothing"),
        ("bodyType", "body"),
        ("stock",    "stock"),
    )

    def _read(feature_id: str) -> float | None:
        """Return the importance for a feature, or None when it is absent."""
        block = feature_weights.get(feature_id) if isinstance(feature_weights, dict) else None
        if isinstance(block, dict) and "importance" in block:
            try:
                return max(0.0, float(block["importance"]))
            except (TypeError, ValueError):
                return None
        return None

    raw = {agent_id: _read(feature_id) for feature_id, agent_id in feature_to_agent}

    # Backward-compat: an older caller may send only color/type/bodyType and no
    # `stock` emphasis.
    # TODO(part-A follow-up): once every caller always emits a `stock` importance
    # (weight_agent paths + orchestrator fallback already do; verify once the chat
    # agent forwards intent too), this branch is dead code — re-verify it is
    # unreachable and remove it. Until then it only affects deprecated callers.
    # If there is a real signal but stock is missing, synthesise a stock importance
    # from `stock_weight`. NOTE: `stock_weight` is a *fraction* (0.20 = 20% of the
    # final budget) while the other emphases are *importances* on a 0–100 scale, so
    # we convert: to give stock a `frac` share of the total, its importance must be
    # frac/(1-frac) × (sum of the other importances).
    has_signal = any(v and v > 0 for v in raw.values())
    if has_signal and (raw["stock"] is None):
        others = sum(v for k, v in raw.items() if k != "stock" and v)
        frac   = min(max(float(stock_weight), 0.0), 0.95)
        raw["stock"] = (frac / (1.0 - frac)) * others if others > 0 else 0.0

    importances = {agent_id: (raw[agent_id] or 0.0) for _, agent_id in feature_to_agent}
    total = sum(importances.values())

    # The RL agent (when enabled) takes a fixed slice off the top; the four
    # conversation emphases share whatever budget remains.
    rl_slice        = max(0.0, float(rl_weight)) if rl_weight and rl_weight > 0 else 0.0
    emphasis_budget = max(0.0, 1.0 - rl_slice)

    if total <= 0:
        # No usable emphases at all → equal split of the emphasis budget.
        share = emphasis_budget / len(feature_to_agent)
        full = {agent_id: share for _, agent_id in feature_to_agent}
    else:
        full = {
            agent_id: imp / total * emphasis_budget
            for agent_id, imp in importances.items()
        }

    # The RL agent is the only fixed-slice agent, included when it carries weight.
    if rl_slice > 0:
        full["rl"] = rl_slice

    if present_agents is None:
        return full

    # Identify which agents are missing and pool their weights
    missing      = {a for a in full if a not in present_agents}
    missing_pool = sum(full[a] for a in missing)

    if not missing or not missing_pool:
        return {a: v for a, v in full.items() if a in present_agents}

    # Redistribute the pooled weight proportionally among present agents
    present      = {a: v for a, v in full.items() if a in present_agents}
    present_sum  = sum(present.values()) or 1.0
    redistributed = {
        a: v + missing_pool * (v / present_sum)
        for a, v in present.items()
    }
    return {a: round(v, 4) for a, v in redistributed.items()}
