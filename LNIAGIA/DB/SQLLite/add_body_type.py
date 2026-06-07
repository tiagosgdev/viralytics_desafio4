"""
Add and backfill the `body_type` column on the clothing dataset.

`body_type` records which body shape(s) a garment is designed to flatter. Values
are gender-specific (see models.BODY_TYPES_BY_GENDER) and chosen from each item's
cut via models.generate_body_types (Constraint #7).

This script:
  1. Adds the `body_type` column to clothing.db -> items (if missing).
  2. Backfills every existing row with logic-driven body type(s).
  3. Backfills the JSON data source(s) in DataSources/ so future DB rebuilds
     (DBManager "recreate") keep the field. Originals are copied to *.bak first.

Assignment is seeded per item id (random.Random(id)), so it is reproducible and
the DB row and JSON item with the same id receive identical body types. Re-running
the script is therefore idempotent.

Usage:
    python add_body_type.py             # backfill DB + JSON
    python add_body_type.py --dry-run   # report only, write nothing
    python add_body_type.py --no-json   # backfill the DB only
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
import sys
from collections import Counter
from pathlib import Path

# Support both `python -m LNIAGIA.DB.SQLLite.add_body_type` and direct execution.
try:
    from LNIAGIA.DB.models import generate_body_types
except ModuleNotFoundError:
    db_dir = Path(__file__).resolve().parent.parent
    if str(db_dir) not in sys.path:
        sys.path.insert(0, str(db_dir))
    from models import generate_body_types

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "clothing.db"
DATA_SOURCES_PATH = BASE_DIR / "DataSources"

# Item fields read by generate_body_types — fetched from the DB row.
RELEVANT_FIELDS = ("gender", "fit", "dress_style", "leg_style", "waist_style", "type")


def _body_type_for(item: dict) -> str:
    """Logic-driven body type(s) for an item, seeded by its id for reproducibility."""
    item_id = item.get("id")
    seed = int(item_id) if item_id is not None else None
    return generate_body_types(item, rng=random.Random(seed))


def _print_distribution(values) -> None:
    """Print how many shapes items flatter, and per-shape frequency."""
    count_per_item = Counter(len(v.split(", ")) if v else 0 for v in values)
    shape_freq = Counter(s for v in values if v for s in v.split(", "))
    total = len(values)

    print("  Shapes per item:")
    for n in sorted(count_per_item):
        c = count_per_item[n]
        print(f"    {n} shape(s): {c} item(s) ({c / total:.1%})")

    print("  Per-shape frequency:")
    for shape, c in shape_freq.most_common():
        print(f"    {shape:<18} {c} item(s)")


# ------------------ DATABASE ------------------

def backfill_db(db_path: Path, dry_run: bool) -> None:
    if not db_path.is_file():
        print(f"  Database not found, skipping: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA table_info(items)")
        columns = [row[1] for row in cursor.fetchall()]
        if not columns:
            print("  No 'items' table found, skipping DB.")
            return

        if "body_type" not in columns:
            if dry_run:
                print("  Dry run: would add column 'body_type' to items.")
            else:
                cursor.execute('ALTER TABLE items ADD COLUMN body_type TEXT')
                print("  Added column 'body_type' to items.")
        else:
            print("  Column 'body_type' already present (will refresh values).")

        select_cols = ", ".join(["id", *RELEVANT_FIELDS])
        cursor.execute(f"SELECT {select_cols} FROM items")
        rows = [dict(row) for row in cursor.fetchall()]

        updates = [(_body_type_for(item), item["id"]) for item in rows]

        if dry_run:
            print(f"  Dry run: would set body_type on {len(updates)} row(s).")
        else:
            cursor.executemany(
                "UPDATE items SET body_type = ? WHERE id = ?", updates
            )
            conn.commit()
            print(f"  Updated body_type on {len(updates)} row(s).")

        _print_distribution([bt for bt, _ in updates])
    finally:
        conn.close()


# ------------------ JSON DATA SOURCES ------------------

def backfill_json(data_sources_path: Path, dry_run: bool) -> None:
    json_files = sorted(data_sources_path.glob("*.json"))  # top-level only, skips old/
    if not json_files:
        print(f"  No JSON files found in {data_sources_path}")
        return

    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as handle:
            items = json.load(handle)

        if not isinstance(items, list) or not items:
            print(f"  Skipping (not a non-empty list): {json_file.name}")
            continue

        body_types = [_body_type_for(item) for item in items]

        if dry_run:
            print(f"  Dry run: would backfill {len(items)} item(s) in {json_file.name}")
            _print_distribution(body_types)
            continue

        # One-time pristine backup, so re-runs never clobber the original.
        backup = json_file.with_suffix(json_file.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(json_file, backup)
            print(f"  Backup written: {backup.name}")

        for item, body_type in zip(items, body_types):
            item["body_type"] = body_type

        with json_file.open("w", encoding="utf-8") as handle:
            json.dump(items, handle, indent=2, ensure_ascii=False)

        print(f"  Backfilled {len(items)} item(s) in {json_file.name}")
        _print_distribution(body_types)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add/backfill the body_type field.")
    parser.add_argument("--db-path", default=str(DB_PATH), help="Path to clothing.db")
    parser.add_argument(
        "--data-sources", default=str(DATA_SOURCES_PATH),
        help="Folder with the JSON data source(s)",
    )
    parser.add_argument("--no-json", action="store_true", help="Backfill the DB only")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without writing anything",
    )
    args = parser.parse_args()

    print("Backfilling DATABASE:")
    backfill_db(Path(args.db_path), args.dry_run)

    if not args.no_json:
        print("\nBackfilling JSON DATA SOURCES:")
        backfill_json(Path(args.data_sources), args.dry_run)

    print("\nDone." + (" (dry run — nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
