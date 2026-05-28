from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DEFAULT_DB_PATH = REPO_ROOT / "LNIAGIA" / "DB" / "SQLLite" / "clothing.db"
DEFAULT_CONFIG_PATH = BASE_DIR / "stock_config.json"

PIVOT_KEYS = ("color", "type", "fit", "size")

# Customer-facing query keys for retrieval. Superset of PIVOT_KEYS; the
# extras narrow the 40-candidate retrieval but don't affect voting /
# attribute_pressure (those stay on PIVOT_KEYS = the 4-agent axes).
QUERY_KEYS = (
    "color", "type", "fit", "size",
    "style", "pattern", "material",
    "gender", "age_group", "season",
    "occasion", "brand",
)

# Numeric range keys (hard filter, not part of match_count).
RANGE_KEYS = ("price_min", "price_max")

# NOTE: julianday('now') is UTC. Seeder writes UTC ISO timestamps; keep them
# UTC-aligned end-to-end or age_days drifts by the local tz offset.
_SCORED_QUERY = """
SELECT
  i.id AS item_id, s.size,
  i.color, i.type, i.fit, i.season,
  i.style, i.pattern, i.material,
  i.gender, i.age_group, i.occasion, i.brand, i.price,
  s.stock_count, s.total_sold, s.last_sold_at, s.active,
  (julianday('now') - julianday(i.created_at)) AS age_days,
  CASE WHEN s.last_sold_at IS NULL
       THEN (julianday('now') - julianday(i.created_at))
       ELSE (julianday('now') - julianday(s.last_sold_at))
  END AS days_since_last_sale,
  (s.total_sold * 1.0) / MAX(julianday('now') - julianday(i.created_at), 1.0)
       AS sales_velocity
FROM items i
JOIN item_stock s ON s.item_id = i.id
WHERE i.created_at IS NOT NULL
"""


class StockStats:
    """Read-only stats API for the StockAgent. Loads once, queries from cache."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.db_path = db_path
        self.config_path = config_path
        self._load_config()
        self.df, self._push_by_key, self._row_by_key = self._load_frame()

    # ─── loading ─────────────────────────────────────────────────────────

    def _load_config(self) -> None:
        with self.config_path.open() as fp:
            cfg = json.load(fp)
        self.weights = cfg["push_score"]["weights"]
        self.refs = cfg["push_score"]["refs"]
        self.sum_weights = sum(self.weights.values())

    def _load_frame(self) -> tuple[pd.DataFrame, dict, dict]:
        # Read-only connection; no pragmas needed (WAL already persisted).
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(_SCORED_QUERY, conn)
        finally:
            conn.close()
        # items.price is TEXT in the DB (e.g. "381.0"); coerce for numeric
        # range filters in get_candidates.
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        return self._score(df)

    def _score(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
        r = self.refs
        w = self.weights

        stock_score = np.clip(df["stock_count"] / r["STOCK_REF"], 0.0, 1.0)
        age_score = np.clip(df["age_days"] / r["AGE_REF"], 0.0, 1.0)
        stag_score = np.clip(df["days_since_last_sale"] / r["AGE_REF"], 0.0, 1.0)
        perf_score = 1.0 - np.clip(
            df["sales_velocity"] / r["VELOCITY_REF"], 0.0, 1.0
        )

        df["stock_score"] = stock_score
        df["age_score"] = age_score
        df["stag_score"] = stag_score
        df["perf_score"] = perf_score
        df["push_score"] = (
            w["w_stock"] * stock_score
            + w["w_age"] * age_score
            + w["w_stag"] * stag_score
            + w["w_perf"] * perf_score
        )
        # active=0 → no push (applied after sum so it survives downstream agg)
        df.loc[df["active"] == 0, "push_score"] = 0.0

        keys = list(zip(df["item_id"].tolist(), df["size"].tolist()))
        push_by_key = dict(zip(keys, df["push_score"].tolist()))
        row_by_key = {k: i for i, k in enumerate(keys)}
        return df, push_by_key, row_by_key

    def reload(self) -> None:
        """Re-query the DB and atomically swap df + lookup dicts."""
        new_df, new_push, new_row = self._load_frame()
        # Atomic-ish swap: all three assignments happen together; readers
        # touching only one are fine. For Phase-2 SPADE concurrency, wrap
        # callers in a lock.
        self.df = new_df
        self._push_by_key = new_push
        self._row_by_key = new_row

    # ─── public API ──────────────────────────────────────────────────────

    def get_stock_stats(self, by: list[str]) -> pd.DataFrame:
        """Aggregate stats grouped by any subset of color/type/fit/size."""
        bad = [k for k in by if k not in PIVOT_KEYS]
        if bad:
            raise ValueError(f"unknown pivot keys: {bad}; allowed: {PIVOT_KEYS}")

        grouped = self.df.groupby(by, sort=False).agg(
            n_rows=("item_id", "count"),
            stock_count=("stock_count", "sum"),
            total_sold=("total_sold", "sum"),
            sales_velocity=("sales_velocity", "mean"),
            mean_age_days=("age_days", "mean"),
            mean_push_score=("push_score", "mean"),
        )
        total_stock = self.df["stock_count"].sum()
        grouped["stock_share"] = (
            grouped["stock_count"] / total_stock if total_stock else 0.0
        )
        return grouped.reset_index()

    def get_overstock_items(self, top_k: int = 50) -> list[tuple[int, str]]:
        """Top-N (item_id, size) pairs ranked by push_score desc.

        Inactive rows (active=0) have push_score=0 by construction (§2.4),
        so they're effectively excluded — no explicit filter needed.
        Asymmetric with get_top_performers, which excludes active=0 explicitly.
        """
        top = self.df.nlargest(top_k, "push_score")
        return list(zip(top["item_id"].tolist(), top["size"].tolist()))

    def get_push_score(self, item_id: int, size: str) -> float:
        key = (item_id, size)
        if key not in self._push_by_key:
            raise KeyError(f"no row for item_id={item_id}, size={size!r}")
        return self._push_by_key[key]

    def get_push_scores(
        self,
        candidates: Iterable[tuple[int, str]],
        *,
        missing: float | None = None,
    ) -> dict[tuple[int, str], float]:
        """Batch lookup for voting on the 40-candidate shortlist.

        missing=None (default): raise KeyError on any unknown (item_id, size).
        missing=<float>:        substitute that value for unknown keys.
                                Use 0.0 if "not in stock" should suppress votes.
        """
        if missing is None:
            return {key: self._push_by_key[key] for key in candidates}
        return {key: self._push_by_key.get(key, missing) for key in candidates}

    def get_row(self, item_id: int, size: str) -> pd.Series:
        """Full row for an (item_id, size) — handy for debugging / printing."""
        key = (item_id, size)
        if key not in self._row_by_key:
            raise KeyError(f"no row for item_id={item_id}, size={size!r}")
        return self.df.iloc[self._row_by_key[key]]

    def get_attribute_pressure(self) -> dict[str, dict[str, float]]:
        """For each pivot key, mean push_score per attribute value.

        Caveat: the `size` axis reflects the seeder's size_mult skew — small
        sizes (XXL/XS) have low stock but also low velocity, inflating their
        mean push_score. The signal still ranks values correctly within each
        attribute, but cross-attribute magnitudes aren't comparable.
        """
        out: dict[str, dict[str, float]] = {}
        for k in PIVOT_KEYS:
            out[k] = (
                self.df.groupby(k, sort=False)["push_score"]
                .mean()
                .sort_values(ascending=False)
                .to_dict()
            )
        return out

    def get_top_performers(self, top_k: int = 20) -> list[tuple[int, str]]:
        """Top-N (item_id, size) by sales_velocity desc, active rows only."""
        active = self.df[self.df["active"] == 1]
        top = active.nlargest(top_k, "sales_velocity")
        return list(zip(top["item_id"].tolist(), top["size"].tolist()))


# ─── memoized accessor ──────────────────────────────────────────────────

_default: StockStats | None = None


def default_stats() -> StockStats:
    global _default
    if _default is None:
        _default = StockStats()
    return _default


# ─── CLI smoke run ──────────────────────────────────────────────────────

def _print_overstock(stats: StockStats, n: int = 20) -> None:
    print(f"\n── top {n} overstock (push_score) ──")
    pairs = stats.get_overstock_items(top_k=n)
    for (iid, sz) in pairs:
        ps = stats.get_push_score(iid, sz)
        row = stats.get_row(iid, sz)
        print(
            f"  ({iid:>5}, {sz:<3})  push={ps:.3f}  "
            f"stock={int(row['stock_count']):>3}  sold={int(row['total_sold']):>4}  "
            f"age={row['age_days']:>6.1f}d  color={row['color']:<10} type={row['type']}"
        )


def _print_top_performers(stats: StockStats, n: int = 10) -> None:
    print(f"\n── top {n} performers (sales_velocity) ──")
    for (iid, sz) in stats.get_top_performers(top_k=n):
        row = stats.get_row(iid, sz)
        print(
            f"  ({iid:>5}, {sz:<3})  vel={row['sales_velocity']:.3f}/d  "
            f"sold={int(row['total_sold']):>4}  age={row['age_days']:>6.1f}d"
        )


def _print_pressure(stats: StockStats, top_n: int = 3) -> None:
    print(f"\n── attribute_pressure (top {top_n} per key) ──")
    pressure = stats.get_attribute_pressure()
    for attr, values in pressure.items():
        head = list(values.items())[:top_n]
        joined = ", ".join(f"{v}={ps:.3f}" for v, ps in head)
        print(f"  {attr:<8} {joined}")


def _self_checks(stats: StockStats) -> None:
    print("\n── self-checks ──")
    df = stats.df

    # 1. push_score ∈ [0, sum_weights]
    assert df["push_score"].min() >= 0.0
    assert df["push_score"].max() <= stats.sum_weights + 1e-9
    print(f"  [1] push_score in [0, {stats.sum_weights}] OK")

    # 2. active=0 → push_score == 0 (or skip if no inactive rows exist)
    inactive_pushes = df.loc[df["active"] == 0, "push_score"]
    if len(inactive_pushes):
        assert (inactive_pushes == 0.0).all()
        print(f"  [2] active=0 rows have push_score==0 OK ({len(inactive_pushes)} rows)")
    else:
        print("  [2] SKIP: no inactive rows in DB to verify")

    # 3. batch == individual (default missing=None: raises on unknown key)
    sample = stats.get_overstock_items(top_k=5)
    batch = stats.get_push_scores(sample)
    for key in sample:
        assert abs(batch[key] - stats.get_push_score(*key)) < 1e-9
    # Also verify missing=<float> path
    fake = [(-1, "ZZ")]
    assert stats.get_push_scores(fake, missing=0.0) == {(-1, "ZZ"): 0.0}
    try:
        stats.get_push_scores(fake)
    except KeyError:
        raised = True
    else:
        raised = False
    assert raised, "get_push_scores(missing=None) should raise on unknown key"
    print("  [3] get_push_scores batch/missing semantics OK")

    # 4. groupby total matches raw total
    by_color = stats.get_stock_stats(by=["color"])
    assert int(by_color["stock_count"].sum()) == int(df["stock_count"].sum())
    print(
        f"  [4] sum by color = {int(by_color['stock_count'].sum())} == "
        f"raw total {int(df['stock_count'].sum())} OK"
    )

    # 5. overstock #1 >= median
    first = stats.get_overstock_items(top_k=1)[0]
    assert stats.get_push_score(*first) >= df["push_score"].median()
    print("  [5] top overstock push_score >= median OK")


def main() -> int:
    try:
        stats = StockStats()
    except FileNotFoundError as exc:
        print(f"Missing file: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(stats.df)} (item, size) rows from {stats.db_path}")
    print(f"weights={stats.weights}  refs={stats.refs}")

    _print_overstock(stats, n=20)
    _print_top_performers(stats, n=10)
    _print_pressure(stats, top_n=3)

    try:
        _self_checks(stats)
    except AssertionError as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 2

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
