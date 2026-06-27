#!/usr/bin/env python3
"""
repair_catalog_coherence.py — in-place coherence repair for clothing.db `items`.

Workstream A (catalog coherence). The catalog assigned the type-independent
attributes (age_group/style and the downstream pattern/material/season/occasion
chain) incoherently, so a persona's multi-attribute goal conjunction matched
only a handful of rows and the simulated-shopper review capped at ~2-2.5.

This script repairs every row IN PLACE (keeps all rows; no drop/rebuild/reseed):

  PART 1 — coherent re-derivation
    For each row, seed the module-level RNG with the item id (deterministic +
    idempotent), then re-derive the incoherent chain conditioned on the FIXED
    attributes, mirroring DataGenerator.generate_item():
        age_group <- type        season   <- type
        material  <- season       style    <- type + age
        pattern   <- style (+age re-roll)  occasion <- type (+age re-roll)
        fit ; type-specific extras ; body_type LAST (depends on cut/fit/gender)
    Held FIXED (never written): id, type, price, created_at, gender, color, brand
    (and the derived image_url / short_description, which stay valid because
    color+type are fixed).

  PART 2 — targeted top-up
    Guarantees each persona's in-band conjunction has >= target (~70) rows by
    forcing the persona's required style (+ occasion) onto eligible rows whose
    FIXED attrs (type/color/gender) are already in-band, then re-deriving the
    dependent fields. office_daniel is the one that actually needs this
    (2 in-band on the 10k DB; ~169 eligible).

Every written value is asserted to be a member of its enum before the UPDATE.
The whole repair runs in ONE transaction. Re-running yields an identical DB.

Usage:
    python3 LNIAGIA/DB/SQLLite/repair_catalog_coherence.py            # repair default DB
    python3 LNIAGIA/DB/SQLLite/repair_catalog_coherence.py --dry-run  # report only
    python3 LNIAGIA/DB/SQLLite/repair_catalog_coherence.py --db PATH
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from pathlib import Path

# ─── locate models.py (LNIAGIA/DB) the way the sibling scripts do ───
_DB_DIR = Path(__file__).resolve().parent.parent  # .../LNIAGIA/DB
if str(_DB_DIR) not in sys.path:
    sys.path.insert(0, str(_DB_DIR))

import models as M  # noqa: E402

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "clothing.db"

# Columns held fixed (never written by the repair).
FIXED_COLUMNS = ("id", "type", "price", "created_at", "gender", "color", "brand")

# Multi-valued (comma-joined) columns -> their enum, for validation.
MULTI_VALUE_ENUMS = {"age_group": M.AGE_GROUP, "body_type": M.BODY_TYPE}

# Single-valued re-derived columns -> their enum, for validation.
SINGLE_VALUE_ENUMS = {
    "style": M.STYLE,
    "pattern": M.PATTERN,
    "material": M.MATERIAL,
    "fit": M.FIT,
    "season": M.SEASON,
    "occasion": M.OCCASION,
    "neckline": M.NECKLINE,
    "collar": M.COLLAR,
    "sleeve_style": M.SLEEVE_STYLE,
    "hem_style": M.HEM_STYLE,
    "closure": M.CLOSURE,
    "hood": M.HOOD,
    "insulation": M.INSULATION,
    "waterproof": M.WATERPROOF,
    "outwear_pockets": M.OUTWEAR_POCKETS,
    "waist": M.WAIST,
    "waist_style": M.WAIST_STYLE,
    "rise": M.RISE,
    "length": M.LENGTH,
    "leg_style": M.LEG_STYLE,
    "bottom_pockets": M.BOTTOM_POCKETS,
    "dress_style": M.DRESS_STYLE,
}

# ─── persona goal-bands (verified against customers.json + models enums) ───
# occasions=None means the persona has no occasion filter (casual_sofia).
PERSONA_BANDS = {
    "party_maya": {
        "types": ("short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"),
        "colors": ("red", "orange", "pink", "coral", "multicolor"),
        "styles": ("elegant", "streetwear", "vintage"),
        "occasions": ("party", "date night", "wedding"),
        "genders": ("female", "unisex"),
        "target": 70,
    },
    "office_daniel": {
        "types": ("long_sleeve_top",),
        "colors": ("navy", "gray", "white", "beige", "black"),
        "styles": ("minimalist", "smart casual"),
        "occasions": ("work",),
        "genders": ("male", "unisex"),
        "target": 70,
    },
    "casual_sofia": {
        "types": ("trousers", "skirt", "shorts"),
        "colors": ("blue", "black", "olive", "gray", "beige"),
        "styles": ("casual", "minimalist"),
        "occasions": None,
        "genders": ("female", "unisex"),
        "target": 70,
    },
}


# ─── coherent re-derivation (mirrors DataGenerator.generate_item) ──────────

def rederive_item(row: dict) -> dict:
    """Return {column: new_value} for every re-derived column of one row.

    Seeds the module RNG with the item id first, so the output is fully
    deterministic and idempotent. Only re-derives columns that are NOT fixed;
    the FIXED attrs (type/gender/color/brand/price/created_at) are read from
    `row` and used to condition the chain.
    """
    item_id = row["id"]
    item_type = row["type"]
    random.seed(item_id)

    age_group = M.generate_age_groups(item_type)
    season = M.get_weighted_season_for_type(item_type)
    material = M.get_weighted_material_for_season(season)
    style = M.get_weighted_style_for_type(item_type, age_group)

    pattern = M.get_weighted_pattern_for_style(style)
    for _ in range(10):
        if M.filter_by_age_appropriateness("pattern", pattern, age_group):
            break
        pattern = M.get_weighted_pattern_for_style(style)

    occasion = M.get_valid_occasion_for_type(item_type)
    for _ in range(10):
        if M.filter_by_age_appropriateness("occasion", occasion, age_group):
            break
        occasion = M.get_valid_occasion_for_type(item_type)

    fit = random.choice(M.FIT)

    out = {
        "style": style,
        "pattern": pattern,
        "material": material,
        "fit": fit,
        "age_group": age_group,
        "season": season,
        "occasion": occasion,
    }

    # Type-specific extra fields (same rules as the generator).
    for field in M.TYPE_FIELDS.get(item_type, ()):
        if field == "neckline":
            valid = [n for n in M.EXTRA_FIELD_VALUES[field]
                     if M.filter_by_age_appropriateness("neckline", n, age_group)]
            out[field] = random.choice(valid) if valid else random.choice(M.EXTRA_FIELD_VALUES[field])
        elif field == "insulation":
            out[field] = M.get_valid_insulation_for_season(season)
        else:
            out[field] = random.choice(M.EXTRA_FIELD_VALUES[field])

    # body_type LAST — depends on gender (fixed) + fit / cut fields (just set).
    item_for_body = {"gender": row["gender"], "fit": fit}
    for cut in ("dress_style", "leg_style", "waist_style"):
        if cut in out:
            item_for_body[cut] = out[cut]
    out["body_type"] = M.generate_body_types(item_for_body, rng=random)

    return out


# ─── top-up: force a persona band onto eligible rows ───────────────────────

def force_persona_band(row: dict, derived: dict, band: dict, persona: str) -> dict:
    """Given an already-re-derived row, force the persona's style (+occasion)
    and re-derive the dependent fields (pattern, body_type). Returns the
    updated `derived` dict. Deterministic (re-seeds with item id)."""
    item_id = row["id"]
    item_type = row["type"]
    random.seed(item_id)

    age_group = derived["age_group"]

    # Force style: persona styles that are age-valid.
    valid_styles = [s for s in band["styles"]
                    if M.filter_by_age_appropriateness("style", s, age_group)]
    if not valid_styles:
        valid_styles = list(band["styles"])
    style = random.choice(valid_styles)
    derived["style"] = style

    # Force occasion (if the persona has one): persona occasions valid for the
    # type and the age group.
    if band["occasions"] is not None:
        type_valid = []
        for occ in band["occasions"]:
            constrained = occ in M.OCCASION_TYPE_CONSTRAINTS
            if (not constrained or item_type in M.OCCASION_TYPE_CONSTRAINTS[occ]) \
               and M.filter_by_age_appropriateness("occasion", occ, age_group):
                type_valid.append(occ)
        if not type_valid:
            raise RuntimeError(
                f"persona {persona}: no type-valid occasion in {band['occasions']} "
                f"for type {item_type} (id={item_id})")
        derived["occasion"] = random.choice(type_valid)

    # Re-derive pattern from the forced style (age re-roll preserved).
    pattern = M.get_weighted_pattern_for_style(style)
    for _ in range(10):
        if M.filter_by_age_appropriateness("pattern", pattern, age_group):
            break
        pattern = M.get_weighted_pattern_for_style(style)
    derived["pattern"] = pattern

    # body_type LAST (cut fields unchanged by the force; fit unchanged).
    item_for_body = {"gender": row["gender"], "fit": derived["fit"]}
    for cut in ("dress_style", "leg_style", "waist_style"):
        if cut in derived:
            item_for_body[cut] = derived[cut]
    derived["body_type"] = M.generate_body_types(item_for_body, rng=random)
    return derived


def row_in_band(row: dict, derived: dict, band: dict) -> bool:
    if row["type"] not in band["types"]:
        return False
    if row["color"] not in band["colors"]:
        return False
    if row["gender"] not in band["genders"]:
        return False
    if derived["style"] not in band["styles"]:
        return False
    if band["occasions"] is not None and derived["occasion"] not in band["occasions"]:
        return False
    return True


def row_eligible(row: dict, band: dict) -> bool:
    """Fixed attrs (type/color/gender) already in-band -> can be forced."""
    return (row["type"] in band["types"]
            and row["color"] in band["colors"]
            and row["gender"] in band["genders"])


# ─── validation ────────────────────────────────────────────────────────────

def assert_enums(item_id: int, derived: dict) -> None:
    for col, val in derived.items():
        if col in MULTI_VALUE_ENUMS:
            allowed = set(MULTI_VALUE_ENUMS[col])
            parts = [p.strip() for p in (val or "").split(",") if p.strip()]
            for p in parts:
                if p not in allowed:
                    raise ValueError(f"id={item_id}: {col}={p!r} not in enum")
        elif col in SINGLE_VALUE_ENUMS:
            if val not in set(SINGLE_VALUE_ENUMS[col]):
                raise ValueError(f"id={item_id}: {col}={val!r} not in enum")
        else:
            raise ValueError(f"id={item_id}: unexpected re-derived column {col!r}")


# ─── reporting helpers ─────────────────────────────────────────────────────

def _dress_baby_senior_primary(rows: list[dict], get) -> int:
    n = 0
    for r in rows:
        if "dress" in r["type"]:
            primary = (get(r)["age_group"] or "").split(",")[0].strip()
            if primary in ("baby", "senior"):
                n += 1
    return n


def _persona_count(rows: list[dict], get, band: dict) -> int:
    return sum(1 for r in rows if row_in_band(r, get(r), band))


# ─── main ──────────────────────────────────────────────────────────────────

def repair(db_path: Path, dry_run: bool = False) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM items ORDER BY id")]
    except sqlite3.OperationalError as exc:
        print(f"ERROR reading items: {exc}", file=sys.stderr)
        conn.close()
        return 1

    print(f"Loaded {len(rows)} items from {db_path}")

    # ── BEFORE snapshot (from the live DB columns) ──
    before_get = lambda r: r  # noqa: E731  rows already hold the current values
    before_baby_senior = _dress_baby_senior_primary(rows, before_get)
    before_persona = {p: _persona_count(rows, before_get, b)
                      for p, b in PERSONA_BANDS.items()}

    # ── PART 1: coherent re-derivation ──
    derived_by_id: dict[int, dict] = {}
    for r in rows:
        d = rederive_item(r)
        assert_enums(r["id"], d)
        derived_by_id[r["id"]] = d

    # ── PART 2: targeted top-up ──
    topup_applied: dict[str, int] = {}
    for persona, band in PERSONA_BANDS.items():
        get = lambda r: derived_by_id[r["id"]]  # noqa: E731
        in_band_now = _persona_count(rows, get, band)
        target = band["target"]
        if in_band_now >= target:
            topup_applied[persona] = 0
            continue
        shortfall = target - in_band_now
        # Eligible rows not already in-band, deterministic order by id.
        candidates = [r for r in rows
                      if row_eligible(r, band)
                      and not row_in_band(r, derived_by_id[r["id"]], band)]
        candidates.sort(key=lambda r: r["id"])
        forced = 0
        for r in candidates:
            if forced >= shortfall:
                break
            d = force_persona_band(r, derived_by_id[r["id"]], band, persona)
            assert_enums(r["id"], d)
            derived_by_id[r["id"]] = d
            forced += 1
        topup_applied[persona] = forced

    # ── AFTER snapshot (from re-derived values) ──
    after_get = lambda r: derived_by_id[r["id"]]  # noqa: E731
    after_baby_senior = _dress_baby_senior_primary(rows, after_get)
    after_persona = {p: _persona_count(rows, after_get, b)
                     for p, b in PERSONA_BANDS.items()}

    # ── report ──
    print("\n── coherence report (before → after) ──")
    print(f"  baby/senior-PRIMARY dresses : {before_baby_senior:>5} → {after_baby_senior}")
    for p in PERSONA_BANDS:
        print(f"  {p:<14} in-band       : {before_persona[p]:>5} → {after_persona[p]}"
              f"   (top-up forced {topup_applied[p]})")

    if dry_run:
        print("\n[dry-run] no rows written.")
        conn.close()
        return 0

    # ── single-transaction write ──
    cur = conn.cursor()
    cur.execute("BEGIN")
    try:
        for r in rows:
            d = derived_by_id[r["id"]]
            cols = list(d.keys())
            assert not (set(cols) & set(FIXED_COLUMNS)), \
                f"repair tried to write fixed columns: {set(cols) & set(FIXED_COLUMNS)} (id={r['id']})"
            assigns = ", ".join(f"{c}=?" for c in cols)
            params = [d[c] for c in cols] + [r["id"]]
            cur.execute(f"UPDATE items SET {assigns} WHERE id=?", params)
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise

    # checkpoint WAL so the .db file is self-contained for backups/hashing
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    print(f"\nWrote {len(rows)} rows (1 transaction). Repair complete.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="In-place catalog coherence repair.")
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH), help="path to clothing.db")
    ap.add_argument("--dry-run", action="store_true", help="report only, no writes")
    args = ap.parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1
    return repair(db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
