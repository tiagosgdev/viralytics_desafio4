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
  - Stock agent receives a fixed `stock_weight` (default 0.20) for inventory health.
  - RL agent receives a fixed `rl_weight` (default 0.15 when enabled) for its
    learned signal; 0 omits it entirely (legacy behaviour).
  - The remaining budget (1 - stock_weight - rl_weight) is distributed among
    body/clothing/colour agents in proportion to the feature importances
    returned by FeatureWeightAgent.
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

    feature_weights : {"color": {"importance": N}, "type": {...}, "bodyType": {...}}
    present_agents  : set of agent ids that actually responded this round.
                      When an agent is absent its weight is redistributed
                      proportionally among the agents that did respond.
                      Pass None (default) to assume all expected agents are present.

    Fixed-slice agents get a constant weight regardless of feature importances:
      - stock agent → `stock_weight`   (inventory health signal)
      - RL agent    → `rl_weight`       (learned signal; 0 disables / omits it)
    The remaining budget (1 - stock_weight - rl_weight) is split among
    body/clothing/colour in proportion to the feature importances (which sum to 100).
    """
    try:
        total = sum(
            float(v["importance"])
            for v in feature_weights.values()
            if isinstance(v, dict) and "importance" in v
        )
    except (TypeError, KeyError):
        total = 0.0

    user_budget = max(0.0, 1.0 - stock_weight - rl_weight)

    if total <= 0:
        share = user_budget / 3.0
        full = {"colour": share, "clothing": share, "body": share}
    else:
        color_imp = float((feature_weights.get("color")    or {}).get("importance", 0) or 0)
        type_imp  = float((feature_weights.get("type")     or {}).get("importance", 0) or 0)
        body_imp  = float((feature_weights.get("bodyType") or {}).get("importance", 0) or 0)
        full = {
            "colour":   color_imp / total * user_budget,
            "clothing": type_imp  / total * user_budget,
            "body":     body_imp  / total * user_budget,
        }

    # Fixed-slice agents are only included when they carry a positive weight, so a
    # disabled RL agent (rl_weight=0) never appears and the legacy split is exact.
    if stock_weight > 0:
        full["stock"] = stock_weight
    if rl_weight > 0:
        full["rl"] = rl_weight

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
