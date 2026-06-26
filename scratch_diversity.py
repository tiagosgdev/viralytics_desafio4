"""Diversity + review metrics for SPADE experiments (read-only on results.db).

For each experiment id given, computes:
  - overall mean review (sanity vs report)
  - TOTAL distinct item_ids surfaced across every turn_items row
  - CONTROLLED-CELL diversity (matches the exp_7 methodology in memory):
      customer=party_maya, repeat_idx=0, turn idx=0, across all 81 combos
      → distinct top-10 SETS, distinct items, mean pairwise Jaccard
  - same controlled-cell numbers for office_daniel & casual_sofia (turn0/repeat0)
"""
import sqlite3, sys, statistics, itertools
from pathlib import Path

DB = Path(__file__).resolve().parent / "multi_agent/experiments/results.db"


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def controlled_cell(db, exp_id, customer, repeat=0, turn_idx=0):
    """Return {combo_name: frozenset(top-10 item_ids)} for the fixed cell."""
    rows = db.execute(
        """
        SELECT e.combo_json, ti.item_id
        FROM episodes e
        JOIN turns t   ON t.episode_id = e.episode_id
        JOIN turn_items ti ON ti.turn_id = t.turn_id
        WHERE e.experiment_id=? AND e.customer_id=? AND e.repeat_idx=? AND t.idx=?
        ORDER BY e.combo_json, ti.rank
        """,
        (exp_id, customer, repeat, turn_idx),
    ).fetchall()
    import json
    sets = {}
    for combo_json, item_id in rows:
        name = json.loads(combo_json or "{}").get("name", "?")
        sets.setdefault(name, []).append(item_id)
    return {k: frozenset(v) for k, v in sets.items()}


def summarize(db, exp_id):
    name = db.execute("SELECT name FROM experiments WHERE experiment_id=?", (exp_id,)).fetchone()
    if not name:
        print(f"\n### exp {exp_id}: NOT FOUND")
        return
    reviews = [r[0] for r in db.execute(
        "SELECT final_review FROM episodes WHERE experiment_id=? AND final_review IS NOT NULL",
        (exp_id,)).fetchall()]
    total_distinct = db.execute(
        """SELECT COUNT(DISTINCT ti.item_id)
           FROM episodes e JOIN turns t ON t.episode_id=e.episode_id
           JOIN turn_items ti ON ti.turn_id=t.turn_id
           WHERE e.experiment_id=?""", (exp_id,)).fetchone()[0]
    print(f"\n{'='*70}\n### exp {exp_id} — {name[0]}")
    print(f"episodes reviewed : {len(reviews)}")
    print(f"overall mean review: {statistics.mean(reviews):.3f} (σ={statistics.pstdev(reviews):.3f})")
    print(f"TOTAL distinct items surfaced (all turns): {total_distinct}")

    for cust in ("party_maya", "office_daniel", "casual_sofia"):
        cell = controlled_cell(db, exp_id, cust)
        if not cell:
            print(f"  [{cust}] no controlled-cell data")
            continue
        sets = list(cell.values())
        distinct_sets = len(set(sets))
        distinct_items = len(set().union(*sets)) if sets else 0
        pairs = list(itertools.combinations(sets, 2))
        mean_j = statistics.mean(jaccard(a, b) for a, b in pairs) if pairs else 1.0
        print(f"  [{cust}] controlled cell (repeat0,turn0): "
              f"{len(sets)} combos → {distinct_sets} distinct top-10 sets, "
              f"{distinct_items} distinct items, mean pairwise Jaccard {mean_j:.3f}")


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]] or [7, 8]
    db = sqlite3.connect(str(DB))
    try:
        for i in ids:
            summarize(db, i)
    finally:
        db.close()
