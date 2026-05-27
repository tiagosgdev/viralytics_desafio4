from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DEFAULT_DB_PATH = REPO_ROOT / "LNIAGIA" / "DB" / "SQLLite" / "clothing.db"

EVENT_SALE = "sale"
EVENT_RESTOCK = "restock"


class StockError(Exception):
    """Base class for stock mutation failures."""


class StockNotFound(StockError):
    """(item_id, size) pair has no row in item_stock."""


class StockUnavailable(StockError):
    """Row exists but cannot fulfill the mutation (inactive / insufficient stock)."""


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a writer connection with WAL + FK pragmas + autocommit (we manage txns)."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _fetch_stock_row(
    cursor: sqlite3.Cursor, item_id: int, size: str
) -> tuple[int, int]:
    cursor.execute(
        "SELECT stock_count, active FROM item_stock WHERE item_id = ? AND size = ?",
        (item_id, size),
    )
    row = cursor.fetchone()
    if row is None:
        raise StockNotFound(f"no item_stock row for item_id={item_id}, size={size!r}")
    return int(row[0]), int(row[1])


def sell(
    conn: sqlite3.Connection, item_id: int, size: str, qty: int = 1
) -> dict:
    """Record a sale: decrement stock_count, increment total_sold, append sale event.

    Raises:
      ValueError if qty <= 0.
      StockNotFound if the (item_id, size) row doesn't exist.
      StockUnavailable if active=0 or stock_count < qty.

    Returns dict with new stock_count, total_sold, last_sold_at.
    """
    if qty <= 0:
        raise ValueError(f"qty must be > 0, got {qty}")

    cursor = conn.cursor()
    cursor.execute("BEGIN IMMEDIATE")
    try:
        stock, active = _fetch_stock_row(cursor, item_id, size)
        if not active:
            raise StockUnavailable(
                f"item_id={item_id} size={size!r} is discontinued (active=0)"
            )
        if stock < qty:
            raise StockUnavailable(
                f"insufficient stock for item_id={item_id} size={size!r}: "
                f"have {stock}, need {qty}"
            )

        ts = _utc_now_iso()
        cursor.execute(
            "UPDATE item_stock SET stock_count = stock_count - ?, "
            "total_sold = total_sold + ?, last_sold_at = ? "
            "WHERE item_id = ? AND size = ?",
            (qty, qty, ts, item_id, size),
        )
        cursor.execute(
            "INSERT INTO stock_events (item_id, size, delta, reason, ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (item_id, size, -qty, EVENT_SALE, ts),
        )
        cursor.execute("COMMIT")
    except Exception:
        cursor.execute("ROLLBACK")
        raise

    return {
        "item_id": item_id,
        "size": size,
        "qty_sold": qty,
        "stock_count": stock - qty,
        "last_sold_at": ts,
    }


def restock(
    conn: sqlite3.Connection, item_id: int, size: str, qty: int
) -> dict:
    """Add stock for an existing (item_id, size) row. Does NOT touch total_sold.

    Raises:
      ValueError if qty <= 0.
      StockNotFound if the (item_id, size) row doesn't exist.

    Note: restocking a row with active=0 is allowed — it just adds stock without
    flipping active. Reactivation belongs to a separate Phase-2 helper.
    """
    if qty <= 0:
        raise ValueError(f"qty must be > 0, got {qty}")

    cursor = conn.cursor()
    cursor.execute("BEGIN IMMEDIATE")
    try:
        stock, _active = _fetch_stock_row(cursor, item_id, size)

        ts = _utc_now_iso()
        cursor.execute(
            "UPDATE item_stock SET stock_count = stock_count + ? "
            "WHERE item_id = ? AND size = ?",
            (qty, item_id, size),
        )
        cursor.execute(
            "INSERT INTO stock_events (item_id, size, delta, reason, ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (item_id, size, qty, EVENT_RESTOCK, ts),
        )
        cursor.execute("COMMIT")
    except Exception:
        cursor.execute("ROLLBACK")
        raise

    return {
        "item_id": item_id,
        "size": size,
        "qty_added": qty,
        "stock_count": stock + qty,
        "ts": ts,
    }


# ─── CLI ────────────────────────────────────────────────────────────────

def _cli_sell(args: argparse.Namespace) -> int:
    conn = get_connection(args.db)
    try:
        result = sell(conn, args.item_id, args.size, args.qty)
    finally:
        conn.close()
    print(result)
    return 0


def _cli_restock(args: argparse.Namespace) -> int:
    conn = get_connection(args.db)
    try:
        result = restock(conn, args.item_id, args.size, args.qty)
    finally:
        conn.close()
    print(result)
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stock mutation helpers (sell / restock).")
    p.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    sub = p.add_subparsers(dest="cmd", required=True)

    sell_p = sub.add_parser("sell", help="Record a sale.")
    sell_p.add_argument("--item-id", type=int, required=True)
    sell_p.add_argument("--size", type=str, required=True)
    sell_p.add_argument("--qty", type=int, default=1)
    sell_p.set_defaults(func=_cli_sell)

    restock_p = sub.add_parser("restock", help="Add stock to an existing (item, size).")
    restock_p.add_argument("--item-id", type=int, required=True)
    restock_p.add_argument("--size", type=str, required=True)
    restock_p.add_argument("--qty", type=int, required=True)
    restock_p.set_defaults(func=_cli_restock)

    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        return args.func(args)
    except (StockError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except sqlite3.Error as exc:
        print(f"DB error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
