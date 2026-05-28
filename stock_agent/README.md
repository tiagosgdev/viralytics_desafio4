# StockAgent — Phase 1 Pipeline

Schema + data + stats layer for the StockAgent (see `desafio4/stock_agent_plan.md`).
No SPADE wrapping yet — that's Phase 2.

All paths below are relative to `viralytics_desafio3/` (the cloned repo root).

---

## Files

| File | Role |
|---|---|
| `stock_agent/migrate_stock_schema.py` | Adds `items.created_at`, creates `item_stock` + `stock_events` tables. Idempotent. |
| `stock_agent/seed_stock.py` | Populates the new tables with synthetic stock + sales history. Deterministic with `--seed`. |
| `stock_agent/stock_mutations.py` | `sell()` and `restock()` helpers + CLI. |
| `stock_agent/stock_stats.py` | Read-only stats API + push-score ranking + CLI smoke. |
| `stock_agent/stock_agent.py` | `StockAgent` class: candidate retrieval (tier-based attribute relaxation) + LLM-driven top-10 picker + interactive REPL. |
| `stock_agent/stock_config.json` | Push-score weights + reference constants. Edit here, not in code. |

---

## Prerequisites

- Python 3.10+
- `pandas`, `numpy`, `tqdm`, `ollama` (already in root `requirements.txt`)
- `sqlite3` CLI for ad-hoc inspection (optional)
- **For `stock_agent.py` only:** Ollama daemon running locally + the model pulled:
  ```
  ollama pull qwen2.5:7b-instruct-q3_K_M
  ollama serve   # in another terminal if not already running
  ```

From repo root:
```
pip install -r requirements.txt
```

---

## End-to-end: first-time setup

Run the three scripts in order. Each is idempotent; safe to re-run.

```bash
cd viralytics_desafio3

# 1. Schema migration. Backs up clothing.db automatically.
python3 stock_agent/migrate_stock_schema.py

# 2. Seed synthetic stock + sales. Refuses if data exists; use --force to wipe.
python3 stock_agent/seed_stock.py

# 3. Smoke-test the stats API (also a sanity check that everything loaded).
python3 stock_agent/stock_stats.py
```

Expected after step 1: backup file `clothing.db.bak-<timestamp>` appears next to `clothing.db`.
Expected after step 2: ~60K `item_stock` rows + ~119K `stock_events` rows.
Expected after step 3: top-20 overstock list + top-10 performers + 5 self-checks passing.

---

## Each script in detail

### 1. `migrate_stock_schema.py`

```
python3 stock_agent/migrate_stock_schema.py [--db PATH] [--no-backup]
```

| Flag | Meaning |
|---|---|
| `--db PATH` | Override default DB path. |
| `--no-backup` | Skip the timestamped `.bak` copy. Useful for dev re-runs. |

Adds (idempotent):
- Column `items.created_at TEXT`.
- Table `item_stock(item_id, size, stock_count, total_sold, last_sold_at, active)`.
- Table `stock_events(id, item_id, size, delta, reason, ts)`.
- Indexes on size, total_sold, (item_id, size).
- Enables WAL mode persistently.

Exit codes: `0` success, `1` DB missing, `2` SQL error.

### 2. `seed_stock.py`

```
python3 stock_agent/seed_stock.py [--db PATH] [--config PATH] [--seed N] [--force] [--no-events] [--no-checks]
```

| Flag | Meaning |
|---|---|
| `--seed N` | RNG seed (default 42). |
| `--force` | Wipe `item_stock` + `stock_events` + `items.created_at` and reseed. |
| `--no-events` | Skip writing the synthetic event log (faster, breaks audit). |
| `--no-checks` | Skip post-seed smoke assertions. |

Distributions live in `stock_config.json` under the `seeder` key.

Runs five smoke assertions after seeding (monotone stock decay, monotone gap-days growth, total_sold ↔ event-log consistency, top-50 push items skewed to oldest quartile).

Exit codes: `0` success, `1` DB missing, `2` runtime/SQL error, `3` refused (data exists, no `--force`).

### 3. `stock_mutations.py`

Two commands. Both transactional (atomic update + event log append).

```bash
# Record a sale
python3 stock_agent/stock_mutations.py sell --item-id 1 --size M --qty 1

# Add stock
python3 stock_agent/stock_mutations.py restock --item-id 1 --size M --qty 10
```

Errors (exit 2):
- `qty <= 0`
- `(item_id, size)` not found
- Selling an inactive row (`active=0`)
- Selling more than `stock_count`

Restocking an inactive row is allowed but does NOT reactivate it.

Programmatic use:
```python
from LNIAGIA.DB.SQLLite.stock_mutations import get_connection, sell, restock

conn = get_connection()
result = sell(conn, item_id=1, size="M", qty=2)
# {'item_id': 1, 'size': 'M', 'qty_sold': 2, 'stock_count': 12, 'last_sold_at': '...'}
conn.close()
```

### 4. `stock_stats.py`

Read-only. Loads once, queries from cache.

```python
import sys; sys.path.insert(0, "LNIAGIA/DB/SQLLite")
from stock_stats import StockStats, default_stats

stats = StockStats()                       # or default_stats() for memoized singleton

# Aggregated stats
stats.get_stock_stats(by=["color"])        # DataFrame
stats.get_stock_stats(by=["color", "size"])

# Ranking
stats.get_overstock_items(top_k=50)        # [(item_id, size), ...]
stats.get_top_performers(top_k=20)

# Per-row push scores
stats.get_push_score(1, "M")               # raises KeyError if missing
stats.get_push_scores([(1, "M"), (2, "L")])              # raises on unknown
stats.get_push_scores([(1, "M"), (-1, "ZZ")], missing=0) # default for unknown

# Negotiation signal
stats.get_attribute_pressure()             # {color: {red: 0.27, ...}, type: {...}, fit: {...}, size: {...}}

# Refresh after mutations
stats.reload()
```

Stand-alone smoke:
```
python3 stock_agent/stock_stats.py
```

Prints top-20 overstock, top-10 performers, attribute pressure, runs 5 self-checks. Exit `0` on success, `2` on failed assertion.

### 5. `stock_agent.py` — interactive StockAgent

Standalone agent surface on top of `stock_stats`. Three responsibilities:

1. **Retrieve 40 candidates** from a structured query via tier-based attribute relaxation (matches all params first → matches all-except-1 → ... until 40 collected, tiebreaker `item_id ASC`).
2. **Rate** each candidate by its push_score (private rating, not a vote — "vote" is reserved for Phase 2 inter-agent negotiation).
3. **Pick top 10** via an Ollama LLM "stock manager" persona that weighs old stock, current real-world season, push_score, and runout risk.

Prereq: Ollama daemon up + `qwen2.5:7b-instruct-q3_K_M` pulled (see Prerequisites).

Programmatic:
```python
import sys; sys.path.insert(0, "LNIAGIA/DB/SQLLite")
from stock_agent import StockAgent

agent = StockAgent()
forty   = agent.get_candidates({"color": "red", "size": "M"}, n=40)
ratings = agent.rate(forty)                  # {(item_id, size): push_score}
top10   = agent.pick_top(forty, k=10)        # LLM picks
```

REPL:
```
python3 stock_agent/stock_agent.py
```

| Command | Action |
|---|---|
| `query color=red type=trousers size=M` | Set the structured query (any subset of the keys below). |
| `candidates [n]` | Fetch + show the 40 (or `n`) candidates with `match_count`. |
| `rate` | Show push_score for each cached candidate. |
| `pick [k]` | Have the LLM pick top-`k` (default 10) from cached candidates. |
| `state` | Print current query + cached count. |
| `help` | Command reference. |
| `exit` / `quit` | Leave REPL. |

If Ollama is unreachable, `pick` raises `RuntimeError` with the cause; REPL keeps running so you can retry once the daemon is up.

#### Allowed query keys

Pulled from `LNIAGIA/DB/models.py`. Unknown keys raise `ValueError`.

**Equality keys** (each contributes to `match_count`; tier-based relaxation drops them one at a time):

| Key | Allowed values |
|---|---|
| `color` | `black, white, gray, navy, blue, red, green, yellow, orange, pink, purple, brown, beige, cream, burgundy, olive, teal, coral, multicolor` |
| `type` | `short_sleeve_top, long_sleeve_top, long_sleeve_outwear, vest, shorts, trousers, skirt, short_sleeve_dress, long_sleeve_dress, vest_dress, sling_dress` (underscores required) |
| `fit` | `slim fit, regular, relaxed, oversized, tailored, loose, fitted, athletic, baggy, cropped` (spaces in `slim fit` need shell quoting) |
| `size` | `XS, S, M, L, XL, XXL` (uppercase) |
| `style` | `casual, formal, smart casual, sporty, bohemian, minimalist, streetwear, vintage, elegant, preppy` |
| `pattern` | `plain, striped, checkered, plaid, floral, polka dot, geometric, abstract, animal print, camouflage, tie-dye, graphic, embroidered` |
| `material` | `cotton, polyester, linen, silk, wool, denim, leather, suede, velvet, satin, chiffon, fleece, cashmere, nylon, rayon, spandex, organic cotton` |
| `gender` | `male, female, unisex` |
| `season` | `spring, summer, autumn, winter, all-season` |
| `occasion` | `everyday, work, party, wedding, beach, sport, date night, travel, lounge, formal event` |
| `brand` | Free-text. ~80 brands across budget/mid/premium/luxury/ultra_luxury tiers (see `BRAND_TIERS` in `models.py`). Examples: `Zara, H&M, Uniqlo, Levi's, Ralph Lauren, Gucci`. |
| `age_group` | **Substring match (case-insensitive).** Stored as comma-separated string like `"adult, young adult"`. Query `age_group=adult` matches both `"adult"` and `"adult, young adult"`. Valid tokens: `baby, child, teenager, young adult, adult, senior`. |

**Range keys** (hard filter — do NOT contribute to `match_count`; rows out of range are dropped before scoring):

| Key | Type | Effect |
|---|---|---|
| `price_min` | float (EUR) | drop rows where `price < price_min` |
| `price_max` | float (EUR) | drop rows where `price > price_max` |

**Voting axes stay narrower.** `get_attribute_pressure()` and `get_stock_stats(by=...)` still operate only over the 4 PIVOT_KEYS (`color, type, fit, size`) — the extended keys exist purely for the retrieval phase.

#### Example sessions

**A. Full 4-attribute query — should yield many `match_count=4` items**

```
stock> query color=red type=trousers fit=slim\ fit size=M
query set: {'color': 'red', 'type': 'trousers', 'fit': 'slim fit', 'size': 'M'}
stock> candidates
40 candidates (sorted by match_count DESC, item_id ASC):
  ( 1760, M  ) match=4  stock= 12 sold=1084 age= 931.2d  color=red type=trousers fit=slim fit
  ...
stock> pick 10
[~90s] Top 10: ...
```

Note: shell escaping for the space — `slim\ fit` or quote the whole token: `query "fit=slim fit"`.

**A2. Extended query — narrow with material + brand + price cap**

```
stock> query color=red type=trousers material=denim brand=Levi's price_max=80
stock> candidates
40 candidates (sorted by match_count DESC, item_id ASC):
  ( 4321, M  ) match=4  stock= 18 sold= 110 ... color=red type=trousers material=denim brand=Levi's
  ...
stock> pick 10
[~90s] Top 10: ...
```

Notes:
- `match_count` ranges 0–4 here (color/type/material/brand). `price_max=80` is a hard filter, not in `match_count`.
- `Levi's` apostrophe + spaces in brand names: shell-quote — `query "brand=Levi's"`.

**B. Partial query — only color + type**

```
stock> query color=navy type=long_sleeve_outwear
stock> candidates 40
40 candidates (max match_count=2):
  ...
stock> rate
rate() — push_scores ...
```

**C. Single attribute — just size**

```
stock> query size=XXL
stock> candidates
```

Returns all in-stock XXL items, ranked by `match_count=1` then `item_id ASC`. Useful for size-clearance demos — XXL skews toward overstock per the seeder distribution.

**D. Demo a sale + reload cycle (mixed REPL + Python)**

Run agent REPL in one terminal:
```
stock> query color=red size=M
stock> candidates
stock> pick 5
[note the #1 item, e.g. (5425, M)]
```

Then in another terminal:
```bash
python3 stock_agent/stock_mutations.py sell --item-id 5425 --size M --qty 50
```

Back in REPL — agent caches the stats frame, so to see the effect:
```
stock> exit
$ python3 stock_agent/stock_agent.py    # restart (or call .stats.reload() programmatically)
stock> query color=red size=M
stock> candidates
[item 5425's stock_count is now 50 lower]
```

**E. Programmatic use (no REPL)**

```python
import sys
sys.path.insert(0, "LNIAGIA/DB/SQLLite")
from stock_agent import StockAgent

agent = StockAgent()

# 1. Retrieve candidates
forty = agent.get_candidates({"color": "blue", "type": "skirt", "size": "S"}, n=40)

# 2. See own rating (push_score per candidate)
ratings = agent.rate(forty)
top_by_push = sorted(ratings.items(), key=lambda kv: -kv[1])[:5]
print("Top 5 by push_score:", top_by_push)

# 3. Ask LLM to pick top 10
picks = agent.pick_top(forty, k=10)
print("LLM picks:", picks)

# Optional: refresh agent after external mutations
agent.stats.reload()
```

**F. Edge cases**

```
stock> query                           # empty query — error
error: query must contain at least one of color/type/fit/size

stock> query brand=Zara                # unknown key
error: unknown query keys: ['brand']; allowed: ('color', 'type', 'fit', 'size')

stock> candidates                      # no query set
set a query first (e.g. query color=red size=M)

stock> rate                            # no candidates cached
no cached candidates; run `candidates` first

stock> query color=reed                # typo'd value — quiet, max match_count=0
stock> candidates
40 candidates (sorted by match_count DESC, item_id ASC):
  (    1, XS ) match=0  ...            # nothing matched, just first 40 by id
```

---

## Push-score formula (`stock_config.json`)

```
push_score = w_stock * stock_score
           + w_age   * age_score
           + w_stag  * stagnation_score
           + w_perf  * (1 - velocity_score)
```

Defaults:
- Weights: `w_stock=0.3, w_age=0.2, w_stag=0.2, w_perf=0.3`
- Refs: `STOCK_REF=200, AGE_REF=1080, VELOCITY_REF=0.5`

All four sub-scores clamped to `[0, 1]` with fixed refs, so `push_score ∈ [0, Σw]` and is stable across turns (does NOT min-max normalize over the current catalogue).

`active=0` rows have `push_score=0` by construction.

---

## Ad-hoc DB inspection

```bash
DB=LNIAGIA/DB/SQLLite/clothing.db

# Row counts
sqlite3 "$DB" "SELECT 'items', COUNT(*) FROM items
   UNION ALL SELECT 'item_stock', COUNT(*) FROM item_stock
   UNION ALL SELECT 'stock_events', COUNT(*) FROM stock_events;"

# All sizes for one item
sqlite3 "$DB" "SELECT size, stock_count, total_sold, last_sold_at
   FROM item_stock WHERE item_id=1
   ORDER BY CASE size
     WHEN 'XS' THEN 1 WHEN 'S' THEN 2 WHEN 'M' THEN 3
     WHEN 'L' THEN 4 WHEN 'XL' THEN 5 WHEN 'XXL' THEN 6 END;"

# Recent events
sqlite3 "$DB" "SELECT id, item_id, size, delta, reason, ts
   FROM stock_events ORDER BY id DESC LIMIT 10;"

# Verify denormalized total_sold matches event log
sqlite3 "$DB" "SELECT
   (SELECT SUM(total_sold) FROM item_stock) AS denorm,
   (SELECT SUM(-delta) FROM stock_events WHERE reason='sale') AS log;"
```

---

## Re-running / resetting

| Goal | Command |
|---|---|
| Reseed with a new RNG state | `python3 stock_agent/seed_stock.py --force --seed 123` |
| Restore pre-seed state | `cp clothing.db.bak-preseed-<ts> clothing.db` (find with `ls *.bak*`) |
| Restore pre-migration state | `cp clothing.db.bak-<ts> clothing.db` (the original migration backup) |
| Reset to empty stock without re-migrating | `python3 stock_agent/seed_stock.py --force` (wipes + reseeds) |

Backups are NOT auto-pruned. Delete old `*.bak*` files manually when satisfied.

---

## Integration smoke (mutation → reload → score change)

Quick end-to-end check that the layers actually talk to each other:

```bash
python3 - <<'EOF'
import sys; sys.path.insert(0, "LNIAGIA/DB/SQLLite")
from stock_stats import StockStats
from stock_mutations import get_connection, sell

stats = StockStats()
before = stats.get_push_score(1, "M")

conn = get_connection()
sell(conn, 1, "M", 1)
conn.close()

stats.reload()
after = stats.get_push_score(1, "M")
print(f"push_score(1,M): {before:.4f} -> {after:.4f}  (delta={after-before:+.4f})")
EOF
```

`after` should be slightly different from `before` — stock down by 1, total_sold up by 1, last_sold_at updated. Sign of the delta depends on which sub-score dominates for that row.

---

## What's NOT here (Phase 2 / later)

- SPADE wrapping of `StockAgent` (process + XMPP messaging)
- `negotiate()` — inter-agent comparison/convergence on a final shown list (needs other agents to exist)
- `vote()` — Phase 2 voting protocol (distinct from `rate()`, which is StockAgent's private rating)
- `manual_remove()` / `discontinue()` / `reactivate()` helpers
- Auto sales simulator (sells drain on a clock)
- RL trainer that updates `stock_config.json` weights or the LLM prompt
- Qdrant integration — kept untouched on purpose
- pytest suite (smoke `__main__` + REPL is the only test layer for now)
- Notebook with sanity plots (`stock_agent_plan.md` §5 step 8) — deferred to Demo Day prep
- Type-specific narrow fields as query keys (`neckline`, `collar`, `sleeve_style`, `hem_style`, `closure`, `hood`, `insulation`, `waterproof`, `waist`, `rise`, `length`, `leg_style`, `dress_style`, pocket variants) — only valid per item type, so left out of the retrieval API. Customers can drill via dialogue (LLM picker has them in the candidate context if relevant)
