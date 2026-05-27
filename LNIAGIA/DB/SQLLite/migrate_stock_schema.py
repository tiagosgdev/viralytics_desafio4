from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "clothing.db"

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_identifier(identifier: str, *, label: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError(f"Invalid {label}: {identifier}")
    return f'"{identifier}"'


def _normalize(name: str) -> str:
    return name.strip().lower()


def _table_columns(cursor: sqlite3.Cursor, table_name: str) -> set[str]:
    table_sql = _quote_identifier(table_name, label="table name")
    cursor.execute(f"PRAGMA table_info({table_sql})")
    return {_normalize(row[1]) for row in cursor.fetchall()}


def _table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def _backup_db(db_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.bak-{timestamp}")
    # Checkpoint WAL into the main file so the backup is self-contained
    # (no -wal / -shm sidecars needed). No-op on rollback-journal mode.
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shutil.copy2(db_path, backup_path)
    return backup_path


def _ensure_items_created_at(cursor: sqlite3.Cursor) -> bool:
    existing = _table_columns(cursor, "items")
    if "created_at" in existing:
        return False
    cursor.execute('ALTER TABLE "items" ADD COLUMN "created_at" TEXT')
    return True


def _create_item_stock(cursor: sqlite3.Cursor) -> bool:
    created = not _table_exists(cursor, "item_stock")
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS item_stock (
            item_id       INTEGER NOT NULL,
            size          TEXT    NOT NULL,
            stock_count   INTEGER NOT NULL,
            total_sold    INTEGER NOT NULL DEFAULT 0,
            last_sold_at  TEXT,
            active        INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (item_id, size),
            FOREIGN KEY (item_id) REFERENCES items(id)
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_item_stock_size ON item_stock(size)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_item_stock_total_sold ON item_stock(total_sold)"
    )
    return created


def _create_stock_events(cursor: sqlite3.Cursor) -> bool:
    created = not _table_exists(cursor, "stock_events")
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_events (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id   INTEGER NOT NULL,
            size      TEXT    NOT NULL,
            delta     INTEGER NOT NULL,
            reason    TEXT    NOT NULL,
            ts        TEXT    NOT NULL,
            FOREIGN KEY (item_id) REFERENCES items(id)
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_stock_events_item ON stock_events(item_id, size)"
    )
    return created


def migrate(db_path: Path, *, backup: bool = True) -> int:
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    if backup:
        backup_path = _backup_db(db_path)
        print(f"Backup written: {backup_path}")
    else:
        print("Backup skipped (--no-backup)")

    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()

        # WAL + FK pragmas first; WAL is persistent on disk, FK is per-session
        # (declared FKs on the new tables only bite once seeders/mutations run).
        cursor.execute("PRAGMA journal_mode=WAL")
        journal_mode = cursor.fetchone()[0]
        print(f"journal_mode = {journal_mode}")
        cursor.execute("PRAGMA foreign_keys=ON")

        # Python's sqlite3 auto-commits DDL by default; wrap in an explicit
        # transaction so a mid-flight failure rolls back cleanly.
        try:
            cursor.execute("BEGIN")

            added_created_at = _ensure_items_created_at(cursor)
            print(
                "items.created_at: added"
                if added_created_at
                else "items.created_at: already present"
            )

            created_item_stock = _create_item_stock(cursor)
            print(
                "item_stock: created"
                if created_item_stock
                else "item_stock: already present"
            )

            created_stock_events = _create_stock_events(cursor)
            print(
                "stock_events: created"
                if created_stock_events
                else "stock_events: already present"
            )

            connection.commit()
        except sqlite3.Error:
            connection.rollback()
            raise
    finally:
        connection.close()

    print("Migration complete.")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add stock-tracking schema to clothing.db (items.created_at, item_stock, stock_events)."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to clothing.db (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the timestamped .bak copy before migrating.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        return migrate(args.db, backup=not args.no_backup)
    except sqlite3.Error as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
