from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "clothing.db"
DEFAULT_CONFIG_PATH = BASE_DIR / "stock_config.json"

EVENT_SEED = "seed"
EVENT_SALE = "sale"


def _load_config(path: Path) -> dict:
    with path.open() as fp:
        return json.load(fp)


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc).replace(microsecond=0)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _ensure_schema(cursor: sqlite3.Cursor) -> None:
    cursor.execute("PRAGMA table_info(items)")
    cols = {row[1] for row in cursor.fetchall()}
    if "created_at" not in cols:
        raise RuntimeError(
            "items.created_at missing. Run migrate_stock_schema.py first."
        )
    for tbl in ("item_stock", "stock_events"):
        cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (tbl,)
        )
        if cursor.fetchone() is None:
            raise RuntimeError(
                f"{tbl} missing. Run migrate_stock_schema.py first."
            )


def _existing_data_counts(cursor: sqlite3.Cursor) -> tuple[int, int, int]:
    cursor.execute("SELECT COUNT(*) FROM item_stock")
    n_stock = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM stock_events")
    n_events = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM items WHERE created_at IS NOT NULL")
    n_items_with_created = cursor.fetchone()[0]
    return n_stock, n_events, n_items_with_created


def _wipe(cursor: sqlite3.Cursor) -> None:
    cursor.execute("DELETE FROM stock_events")
    cursor.execute("DELETE FROM item_stock")
    cursor.execute("UPDATE items SET created_at = NULL")
    # Reset autoincrement on stock_events
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='stock_events'")


def _sample_created_at(
    rng: np.random.Generator, n: int, now: datetime, cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Sample age_days per item via bucket weights, then convert to ISO timestamps."""
    age_max = cfg["age_days_max"]
    bucket_w = np.array(cfg["age_bucket_weights"], dtype=float)
    bucket_w /= bucket_w.sum()
    # buckets: [0, age_max/3), [age_max/3, 2*age_max/3), [2*age_max/3, age_max]
    edges = np.linspace(0, age_max, len(bucket_w) + 1)
    bucket_idx = rng.choice(len(bucket_w), size=n, p=bucket_w)
    lows = edges[bucket_idx]
    highs = edges[bucket_idx + 1]
    age_days = rng.uniform(lows, highs)
    created_at = [
        _iso(now - timedelta(days=float(d), seconds=float(rng.uniform(0, 86400))))
        for d in age_days
    ]
    return age_days, np.array(created_at)


def _sample_stock_and_sales(
    rng: np.random.Generator,
    item_ids: np.ndarray,
    age_days: np.ndarray,
    cfg: dict,
) -> dict[str, np.ndarray]:
    """Generate per-(item, size) stock + total_sold + last_sold_at offsets.

    Returns a dict of flat numpy arrays of length n_items * n_sizes.
    """
    sizes = cfg_sizes = cfg["__sizes__"]
    size_mult_map = cfg["__size_mult__"]
    n_items = len(item_ids)
    n_sizes = len(sizes)

    sizes_arr = np.tile(np.array(sizes), n_items)
    size_mult = np.tile(
        np.array([size_mult_map[s] for s in sizes], dtype=float), n_items
    )
    item_id_arr = np.repeat(item_ids, n_sizes)
    age_arr = np.repeat(age_days, n_sizes)

    # ---- stock_count ----
    mu = cfg["base_stock_lognormal"]["mu"]
    sigma = cfg["base_stock_lognormal"]["sigma"]
    base_stock = np.clip(rng.lognormal(mu, sigma, size=item_id_arr.shape), *cfg["stock_clip"])

    age_decay = np.maximum(0.05, 1.0 - age_arr / cfg["age_days_max"])
    noise = rng.lognormal(0.0, cfg["noise_lognormal_sigma"], size=item_id_arr.shape)

    # Fat-tail overstock: 8% of rows multiplied by Uniform(2, 4)
    fat_mask = rng.random(item_id_arr.shape) < cfg["fat_tail_overstock_rate"]
    lo, hi = cfg["fat_tail_overstock_mult_range"]
    fat_mult = np.where(fat_mask, rng.uniform(lo, hi, size=item_id_arr.shape), 1.0)

    stock_count = np.clip(
        np.round(base_stock * size_mult * age_decay * noise * fat_mult),
        *cfg["stock_count_clip"],
    ).astype(np.int64)

    # ---- expected_velocity (units/day) ----
    v_lo, v_hi = cfg["velocity_uniform_range"]
    velocity = rng.uniform(v_lo, v_hi, size=item_id_arr.shape) * size_mult

    # Underperformer fat tail
    under_mask = rng.random(item_id_arr.shape) < cfg["underperformer_rate"]
    u_lo, u_hi = cfg["underperformer_mult_range"]
    under_mult = np.where(under_mask, rng.uniform(u_lo, u_hi, size=item_id_arr.shape), 1.0)
    velocity = velocity * under_mult

    # ---- total_sold ----
    sold_noise_lo, sold_noise_hi = cfg["total_sold_noise_range"]
    sold_noise = rng.uniform(sold_noise_lo, sold_noise_hi, size=item_id_arr.shape)
    total_sold = np.clip(
        np.round(velocity * age_arr * sold_noise),
        *cfg["total_sold_clip"],
    ).astype(np.int64)

    # ---- last_sold_at offsets (days ago from now) ----
    # gap_days ~ Exp(scale=30 / max(velocity, 0.01)); slow movers -> larger gaps
    eff_velocity = np.maximum(velocity, 0.01)
    gap_scale = cfg["gap_days_exp_scale"] / eff_velocity
    gap_days = rng.exponential(gap_scale)

    # Clip gap to [0, age_days] so last_sold_at >= created_at
    gap_days = np.minimum(gap_days, age_arr)

    # Force never-sold rate + items with total_sold == 0
    never_mask = (rng.random(item_id_arr.shape) < cfg["never_sold_rate"]) | (total_sold == 0)

    return {
        "item_id": item_id_arr,
        "size": sizes_arr,
        "stock_count": stock_count,
        "total_sold": total_sold,
        "gap_days": gap_days,
        "never_sold": never_mask,
        "age_days": age_arr,
        "fat_overstock": fat_mask,
        "underperformer": under_mask,
        "velocity": velocity,
    }


def _build_last_sold_at(
    now: datetime, gap_days: np.ndarray, never_mask: np.ndarray
) -> list[str | None]:
    out: list[str | None] = []
    for g, never in zip(gap_days, never_mask):
        if never:
            out.append(None)
        else:
            out.append(_iso(now - timedelta(days=float(g))))
    return out


def _build_events(
    now: datetime,
    flat_item_id: np.ndarray,
    flat_size: np.ndarray,
    stock_count: np.ndarray,
    total_sold: np.ndarray,
    created_at_iso: np.ndarray,
    age_days_per_item: np.ndarray,
    n_sizes: int,
) -> list[tuple]:
    """One arrival event per (item, size); one aggregated sale event if total_sold>0.

    Inputs `flat_*`, `stock_count`, `total_sold` are length n_items * n_sizes.
    `created_at_iso` / `age_days_per_item` are per-item; repeated here.
    """
    events: list[tuple] = []
    created_arr = np.repeat(created_at_iso, n_sizes)
    age_arr = np.repeat(age_days_per_item, n_sizes)

    for i in range(len(flat_item_id)):
        it = int(flat_item_id[i])
        sz = str(flat_size[i])
        arrival_delta = int(stock_count[i] + total_sold[i])
        events.append((it, sz, arrival_delta, EVENT_SEED, str(created_arr[i])))
        if total_sold[i] > 0:
            midpoint = now - timedelta(days=float(age_arr[i]) / 2.0)
            events.append((it, sz, -int(total_sold[i]), EVENT_SALE, _iso(midpoint)))
    return events


def seed(
    db_path: Path,
    config_path: Path,
    *,
    seed: int,
    force: bool,
    no_events: bool,
    no_checks: bool,
) -> int:
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    full_cfg = _load_config(config_path)
    cfg = dict(full_cfg["seeder"])
    cfg["__sizes__"] = full_cfg["sizes"]
    cfg["__size_mult__"] = full_cfg["size_multipliers"]

    rng = np.random.default_rng(seed)
    now = _utc_now()

    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")

        _ensure_schema(cursor)

        n_stock, n_events, n_items_with_created = _existing_data_counts(cursor)
        if (n_stock or n_events or n_items_with_created) and not force:
            print(
                f"Existing data present "
                f"(item_stock={n_stock}, stock_events={n_events}, items.created_at set on {n_items_with_created}). "
                f"Re-run with --force to wipe and reseed.",
                file=sys.stderr,
            )
            return 3

        cursor.execute("SELECT id FROM items ORDER BY id")
        item_ids = np.array([row[0] for row in cursor.fetchall()], dtype=np.int64)
        n_items = len(item_ids)
        if n_items == 0:
            print("No rows in items; nothing to seed.", file=sys.stderr)
            return 1

        n_sizes = len(cfg["__sizes__"])
        print(f"Seeding {n_items} items × {n_sizes} sizes = {n_items * n_sizes} stock rows")
        print(f"seed={seed}, now={_iso(now)}")

        try:
            cursor.execute("BEGIN")

            if force:
                _wipe(cursor)

            # 1. items.created_at
            age_days_items, created_iso = _sample_created_at(rng, n_items, now, cfg)
            cursor.executemany(
                "UPDATE items SET created_at = ? WHERE id = ?",
                [(c, int(i)) for c, i in zip(created_iso, item_ids)],
            )

            # 2. item_stock rows
            data = _sample_stock_and_sales(rng, item_ids, age_days_items, cfg)
            last_sold_at = _build_last_sold_at(now, data["gap_days"], data["never_sold"])

            rows = [
                (int(iid), sz, int(sc), int(ts), lsa, 1)
                for iid, sz, sc, ts, lsa in zip(
                    data["item_id"],
                    data["size"],
                    data["stock_count"],
                    data["total_sold"],
                    last_sold_at,
                )
            ]
            cursor.executemany(
                "INSERT INTO item_stock "
                "(item_id, size, stock_count, total_sold, last_sold_at, active) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            print(f"item_stock: {len(rows)} rows")

            # 3. stock_events (optional)
            if not no_events:
                events = _build_events(
                    now,
                    data["item_id"],
                    data["size"],
                    data["stock_count"],
                    data["total_sold"],
                    created_iso,
                    age_days_items,
                    n_sizes,
                )
                cursor.executemany(
                    "INSERT INTO stock_events (item_id, size, delta, reason, ts) "
                    "VALUES (?, ?, ?, ?, ?)",
                    events,
                )
                print(f"stock_events: {len(events)} rows")
            else:
                print("stock_events: skipped (--no-events)")

            connection.commit()
        except sqlite3.Error:
            connection.rollback()
            raise

        if not no_checks:
            _run_smoke_checks(connection, full_cfg)

    finally:
        connection.close()

    print("Seeding complete.")
    return 0


def _run_smoke_checks(connection: sqlite3.Connection, full_cfg: dict) -> None:
    print("\n--- smoke checks ---")
    cursor = connection.cursor()

    refs = full_cfg["push_score"]["refs"]
    w = full_cfg["push_score"]["weights"]
    age_max = full_cfg["seeder"]["age_days_max"]

    # 1. Stock decreases across age quartiles
    cursor.execute(
        """
        WITH s AS (
          SELECT
            i.id,
            CAST((julianday('now') - julianday(i.created_at)) AS REAL) AS age_days,
            (SELECT AVG(stock_count) FROM item_stock WHERE item_id = i.id) AS mean_stock
          FROM items i
          WHERE i.created_at IS NOT NULL
        ),
        q AS (
          SELECT id, age_days, mean_stock,
                 NTILE(4) OVER (ORDER BY age_days) AS quartile
          FROM s
        )
        SELECT quartile, AVG(mean_stock), AVG(age_days)
        FROM q GROUP BY quartile ORDER BY quartile
        """
    )
    q_rows = cursor.fetchall()
    stocks = [r[1] for r in q_rows]
    print("[1] mean stock per age quartile (Q1→Q4, expect decreasing):")
    for q, ms, ad in q_rows:
        print(f"    Q{q}: age~{ad:6.1f}d  mean_stock={ms:7.2f}")
    assert all(stocks[i] > stocks[i + 1] for i in range(len(stocks) - 1)), \
        "FAIL: mean stock not strictly decreasing"

    # 2. days_since_last_sale increases across age quartiles
    cursor.execute(
        """
        WITH s AS (
          SELECT
            i.id,
            CAST((julianday('now') - julianday(i.created_at)) AS REAL) AS age_days,
            (SELECT AVG(julianday('now') - julianday(last_sold_at))
               FROM item_stock WHERE item_id = i.id AND last_sold_at IS NOT NULL) AS mean_gap
          FROM items i
          WHERE i.created_at IS NOT NULL
        ),
        q AS (
          SELECT id, age_days, mean_gap,
                 NTILE(4) OVER (ORDER BY age_days) AS quartile
          FROM s
          WHERE mean_gap IS NOT NULL
        )
        SELECT quartile, AVG(mean_gap)
        FROM q GROUP BY quartile ORDER BY quartile
        """
    )
    gaps = [r[1] for r in cursor.fetchall()]
    print(f"[2] mean days_since_last_sale per age quartile: {[round(g, 1) for g in gaps]}")
    assert all(gaps[i] < gaps[i + 1] for i in range(len(gaps) - 1)), \
        "FAIL: mean gap not strictly increasing"

    # 3. total_sold consistent with stock_events sales
    cursor.execute("SELECT COALESCE(SUM(total_sold), 0) FROM item_stock")
    sum_total_sold = cursor.fetchone()[0]
    cursor.execute(
        "SELECT COALESCE(SUM(-delta), 0) FROM stock_events WHERE reason = 'sale'"
    )
    sum_sale_events = cursor.fetchone()[0]
    print(f"[3] sum(total_sold)={sum_total_sold} vs sum(-sale.delta)={sum_sale_events}")
    if sum_sale_events > 0:
        assert sum_total_sold == sum_sale_events, "FAIL: aggregate mismatch"

    # 4. Top-50 push_score over-represents oldest quartile
    cursor.execute(
        f"""
        WITH joined AS (
          SELECT
            i.id, s.size, s.stock_count, s.total_sold, s.last_sold_at,
            CAST((julianday('now') - julianday(i.created_at)) AS REAL) AS age_days
          FROM items i JOIN item_stock s ON s.item_id = i.id
          WHERE i.created_at IS NOT NULL
        ),
        scored AS (
          SELECT *,
            MIN(stock_count * 1.0 / {refs['STOCK_REF']}, 1.0) AS stock_score,
            MIN(age_days / {refs['AGE_REF']}, 1.0) AS age_score,
            MIN(
              CASE
                WHEN last_sold_at IS NULL THEN age_days
                ELSE julianday('now') - julianday(last_sold_at)
              END / {refs['AGE_REF']}, 1.0
            ) AS stag_score,
            (1.0 - MIN(
              (total_sold * 1.0 / MAX(age_days, 1)) / {refs['VELOCITY_REF']}, 1.0
            )) AS perf_score
          FROM joined
        )
        SELECT id, size, age_days,
               {w['w_stock']}*stock_score + {w['w_age']}*age_score
               + {w['w_stag']}*stag_score + {w['w_perf']}*perf_score AS push_score
        FROM scored
        ORDER BY push_score DESC
        LIMIT 50
        """
    )
    top50 = cursor.fetchall()
    age_q4_threshold = 3 * age_max / 4
    oldest_share = sum(1 for r in top50 if r[2] >= age_q4_threshold) / max(len(top50), 1)
    print(f"[4] top-50 push_score: {oldest_share*100:.0f}% are in oldest age quartile (>{age_q4_threshold:.0f}d)")
    assert oldest_share >= 0.50, \
        f"FAIL: top-50 push not over-represented in oldest quartile ({oldest_share:.2f})"

    print("--- all smoke checks passed ---")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Seed item_stock + stock_events + items.created_at with synthetic data."
    )
    p.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="Wipe existing stock/events/created_at and reseed.")
    p.add_argument("--no-events", action="store_true",
                   help="Skip writing stock_events (faster, but breaks audit log).")
    p.add_argument("--no-checks", action="store_true",
                   help="Skip post-seed smoke assertions.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        return seed(
            args.db,
            args.config,
            seed=args.seed,
            force=args.force,
            no_events=args.no_events,
            no_checks=args.no_checks,
        )
    except (sqlite3.Error, RuntimeError, AssertionError) as exc:
        print(f"Seeding failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
