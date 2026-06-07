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
  - The remaining budget (0.80) is distributed among body/clothing/colour agents
    in proportion to the feature importances returned by FeatureWeightAgent.
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
) -> dict[str, float]:
    """
    Convert FeatureWeightAgent output to per-agent budget.

    feature_weights : {"color": {"importance": N}, "type": {...}, "bodyType": {...}}

    The stock agent gets a fixed `stock_weight` (inventory health signal).
    The remaining 1 - stock_weight is split among body/clothing/colour agents
    in proportion to the feature importances (which sum to 100).
    """
    try:
        total = sum(
            float(v["importance"])
            for v in feature_weights.values()
            if isinstance(v, dict) and "importance" in v
        )
    except (TypeError, KeyError):
        total = 0.0

    user_budget = 1.0 - stock_weight

    if total <= 0:
        # Equal split among the three user-preference agents
        share = user_budget / 3.0
        return {"colour": share, "clothing": share, "body": share, "stock": stock_weight}

    color_imp    = float((feature_weights.get("color")    or {}).get("importance", 0) or 0)
    type_imp     = float((feature_weights.get("type")     or {}).get("importance", 0) or 0)
    body_imp     = float((feature_weights.get("bodyType") or {}).get("importance", 0) or 0)

    return {
        "colour":   color_imp / total * user_budget,
        "clothing": type_imp  / total * user_budget,
        "body":     body_imp  / total * user_budget,
        "stock":    stock_weight,
    }
