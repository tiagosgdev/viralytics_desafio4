"""
Candidate retrieval
────────────────────
Shared candidate-retrieval logic for the recommendation round.

`get_candidates` queries a StockAgent for the items that will be debated by the
scorer agents, applying the conversation-driven DB filters and a soft user-gender
include, and returns a list of fully-populated candidate dicts.

This module is deliberately transport-agnostic: it depends only on the StockAgent
interface (passed in explicitly) and `N_CANDIDATES` from config. It imports neither
SPADE nor the orchestrator, so both the live OrchestratorAgent and the (future)
experiment harness can call it to retrieve identical candidate sets.
"""

from multi_agent.config import N_CANDIDATES


def get_candidates(
    stock_agent,
    weights_result: dict,
    context: dict,
    n: int = N_CANDIDATES,
) -> list[dict]:
    query_filters: dict = dict(weights_result.get("filters") or {})

    # Inject user gender as a soft include so gender-appropriate items rank first
    gender = str(context.get("user_gender") or "").strip().lower()
    if gender in ("male", "female"):
        inc = dict(query_filters.get("include") or {})
        if "gender" not in inc:
            inc["gender"] = [gender, "unisex"]
            query_filters = {**query_filters, "include": inc}

    # get_candidates raises if the query is completely empty
    try:
        pairs = stock_agent.get_candidates(query_filters, n=n)
    except Exception:
        pairs = stock_agent.stats.get_overstock_items(top_k=n)

    info: list[dict] = []
    for iid, sz in pairs:
        try:
            row = stock_agent.stats.get_row(iid, sz)
        except KeyError:
            continue
        info.append({
            "item_id":    int(iid),
            "size":       sz,
            "color":      row.get("color", ""),
            "type":       row.get("type", ""),
            "fit":        row.get("fit", ""),
            "season":     row.get("season", ""),
            "style":      row.get("style", ""),
            "pattern":    row.get("pattern", ""),
            "material":   row.get("material", ""),
            "gender":     row.get("gender", ""),
            "age_group":  row.get("age_group", ""),
            "occasion":   row.get("occasion", ""),
            "brand":      row.get("brand", ""),
            "price":      row.get("price"),
            "stock_count": int(row.get("stock_count", 0)),
            "push_score": float(row.get("push_score", 0.0)),
        })
    return info
