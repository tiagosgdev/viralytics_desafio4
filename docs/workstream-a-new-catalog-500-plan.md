# Workstream A — Build a NEW, coherent ~500-item catalog (full regenerate)

> **PLAN ONLY — nothing here has been implemented.** No `.py` was changed, no DB was
> mutated, no experiment was run. This document was produced by a plan-author agent in an
> agent pipeline (read/inspect the repo → write this plan). The next step is a *separate*
> implementer session that follows §10 top-to-bottom.
>
> **Scope of this plan:** Workstream A of `docs/experiment-improvement-plan.md`, via the
> approach the user chose: **regenerate a brand-new, semantically coherent catalog of
> N≈500 items** (Option **i** — "fix the generator + full regenerate"), *not* the in-place
> `UPDATE items` repair (Option ii) and *not* the `repair_catalog_coherence.py` script the
> doc originally recommended.
>
> **Hygiene for the implementer (not this plan):** branch `rl-learning-curve` only, never
> push to `main`; commit author `tiagosgdev` with **no Claude co-author trailer**; do **not**
> `pip install -r requirements.txt` (it pins `numpy<2` and breaks the env — system Python
> already has numpy/pandas/tqdm).

---

## 1. Approach & rationale

### 1.1 What we build
A fresh `clothing.db` whose `items` table has **~500 rows** that are *semantically coherent*
— attributes are conditioned on `type`/`style` so that "a baby formal sling dress" basically
never gets generated, and a persona's 3–4-attribute goal conjunction matches **many** rows
instead of a handful. The **structured attribute columns** and every enum value-set are
**unchanged**; only the *joint distribution* of values changes (plus a deliberate quota layer
— see §4). One caveat on the schema, made precise below: the rebuild pipeline preserves the
structured attribute columns + `created_at`, but it does **not** create the two derived
columns (`image_url`, `short_description`) — those only reappear if
`update_images_and_description.py` is run (see §1.3 / §5).

This is a **pipeline**, not a single script. A coherent DB requires, in order:
1. `LNIAGIA/DB/SQLLite/DataGenerator.py` → writes `DataSources/<timestamp>.json` (N items).
2. `LNIAGIA/DB/SQLLite/DBManager.py::populate_db(json, recreate=True)` → DROPs+recreates
   `items` + `users`, inserts the JSON, creates 10 random users.
3. `stock_agent/migrate_stock_schema.py` → adds `items.created_at`, creates `item_stock` +
   `stock_events` (and writes a `.bak-<ts>` backup automatically).
4. `stock_agent/seed_stock.py --force` → seeds `created_at`, `item_stock`
   (N × 6 sizes rows), and `stock_events`.

### 1.2 Why a full regenerate (Option i) rather than in-place repair (Option ii)
- The user explicitly wants a **smaller, more realistic catalog (~500 items)**. In-place
  `UPDATE items` keeps all 10,000 rows; it can fix coherence but cannot *shrink* the catalog
  or re-roll prices/brands to a tighter realistic set. Shrinking ⇒ regenerate.
- Regenerating also lets us fold the coherence fix into `models.py`/`DataGenerator.py`
  permanently, so every *future* regenerate is coherent too (the doc's stated long-term goal).
- A 500-row catalog is small enough that a deliberate **stratified/quota** generation pass
  (§4) is cheap and lets us *guarantee* persona-band coverage — something a random in-place
  repair cannot promise.

### 1.3 Blast radius (be explicit — this is destructive)
Running the pipeline with `recreate=True` + `seed_stock --force` **destroys and rebuilds**:
- **`items`** — dropped and recreated from the new JSON. The current 10,000 rows are gone.
  `id` values restart at 1..N (new ids; old ids meaningless afterwards).
- **`users`** — dropped and recreated (10 fresh random users). The current 10 users are gone.
- **`item_stock`** (currently 60,000 rows) — wiped by `seed_stock --force`, rebuilt as
  N×6 ≈ **3,000 rows**.
- **`stock_events`** (currently 119,391 rows) — wiped + rebuilt (~thousands of rows).
- **`items.created_at`** — re-sampled by the seeder.

**Stales (NOT auto-updated):**
- **`image_url`** + **`short_description`** columns: in the *new* catalog these columns are
  **not created at all** (absent, not NULL). `populate_db(recreate=True)` builds `items` from
  `DBManager._all_columns()` (= `models.GLOBAL_FIELDS` + `TYPE_FIELDS`), which does **not**
  include either column; only `update_images_and_description.py` adds them back via
  `ALTER TABLE ... ADD COLUMN`. The SPADE experiment never reads them (confirmed: the
  scored query in `stock_agent/stock_stats.py` selects only structured columns), so the
  experiment is fine without them. The **separate chat-search app**
  (`LNIAGIA/search_app.py` / `src/api/main.py`) *does* use them and will be broken (or error
  on the missing columns) until they are regenerated. See §5.
- **Qdrant vector store** (`LNIAGIA/DB/vector/qdrant_storage/`): embeddings keyed to old
  item ids/attributes become stale/orphaned. Again only the chat-search app cares. See §5.

> **Metric epoch:** rebuilding the catalog **resets the primary 1–5 review metric**. Runs
> after this change are **NOT comparable** to historical experiments #5–#12 (nor to the
> post-`af3b41d`/`bf26765` epoch). Record this as a metric-epoch marker (Workstream D2 in the
> improvement plan).

### 1.4 Verified baseline (why this matters) — measured on the current 10k DB
| Metric (current 10,000-row catalog) | Value |
|---|---|
| `items` / `item_stock` / `stock_events` / `users` | 10,000 / 60,000 / 119,391 / 10 |
| dress rows by **primary** age_group (≈uniform!) | young adult 328, **senior 323**, child 306, **baby 295**, teenager 293, adult 278 |
| baby-**primary** dresses (`type LIKE '%dress%' AND age_group LIKE 'baby%'`) | **581** |
| baby-tagged dresses, **any group** (`type LIKE '%dress%' AND age_group LIKE '%baby%'`) | **907** |
| **party_maya** goal conjunction (dress ∧ vivid colour ∧ party-occasion ∧ elegant/streetwear/vintage ∧ female/unisex) | **97 rows** |
| **office_daniel** goal conjunction (long_sleeve_top ∧ neutral colour ∧ minimalist/smart-casual ∧ work ∧ male/unisex) | **2 rows** ⚠️ |
| **casual_sofia** goal conjunction (trousers/skirt/shorts ∧ classic colour ∧ casual/minimalist ∧ female/unisex) | **143 rows** |

> **Baby-dress predicate note (re-verified on the live 10k DB):** the "581" figure counts
> dresses whose **primary** age_group is baby (`age_group LIKE 'baby%'`). Counting dresses
> with baby in **any** group position (`age_group LIKE '%baby%'`) gives **907** — the larger
> number a reviewer measuring "any-group" will see. Both are real; they answer different
> questions (primary-vs-any). The age_group column is comma-joined, so the two predicates
> diverge by design. The three persona conjunction counts (97 / 2 / 143) were re-run with the
> exact predicates in §8(b) and reproduce on the current DB.

The age_group distribution being ~flat across dresses *is* the incoherence the doc describes.
And note **office_daniel already matches only 2 rows in 10k**. At N=500 (1/20th the rows),
naive coherent generation would give office_daniel **~0 matching items** — which would
*defeat the entire purpose*. This is the headline risk and the reason §4 (quota generation)
is mandatory, not optional.

---

## 2. Generator coherence changes (`models.py` + `DataGenerator.py`)

Today (verified in `DataGenerator.generate_item`), four fields bypass type/style conditioning:
- `age_group = generate_age_groups()` — **no `type` arg**; draws a uniform primary group
  (`random.randint(0, len-1)`) → flat age distribution, hence baby dresses.
- `style = random.choice(valid_styles)` — uniform over age-allowed styles; **`STYLE_WEIGHTS`
  is defined (models.py L834) but never used.**
- `color = random.choice(STATIC_GLOBAL_FIELDS["color"])` — uniform.
- `fit = random.choice(STATIC_GLOBAL_FIELDS["fit"])` — uniform; **no `FIT_WEIGHTS` exists.**

The other chain fields already condition on type/season/style/age:
`get_weighted_season_for_type` → `get_weighted_material_for_season` →
`get_weighted_pattern_for_style` → `get_valid_occasion_for_type`, and
`generate_body_types(item)` reads the cut fields. Keep that structure; sharpen it.

### 2.1 `age_group` — the worst offender (REQUIRED)
Author a new affinity table and give `generate_age_groups()` a `type` parameter.

```python
# models.py — new table near AGE_GROUP_WEIGHTS (~L819).
# Primary-age affinity per type. Keys MUST be exact TYPE values; values MUST be
# exact AGE_GROUP values. Weights are relative (they get normalized at draw time).
TYPE_AGE_GROUP_AFFINITY = {
    # dresses: overwhelmingly young-adult/adult, a little teen; ~never baby/child/senior
    "short_sleeve_dress": {"young adult": 5, "adult": 4, "teenager": 1.5, "senior": 0.4},
    "long_sleeve_dress":  {"young adult": 5, "adult": 4, "teenager": 1.5, "senior": 0.4},
    "vest_dress":         {"young adult": 5, "adult": 4, "teenager": 1.5, "senior": 0.4},
    "sling_dress":        {"young adult": 6, "adult": 3, "teenager": 1.5},  # party-ish, no senior
    # smart/work tops & outwear: adult-centric
    "long_sleeve_top":    {"adult": 5, "young adult": 4, "teenager": 1.5, "senior": 1.0},
    "long_sleeve_outwear":{"adult": 5, "young adult": 4, "senior": 1.5, "teenager": 1.0},
    "trousers":           {"adult": 5, "young adult": 4, "senior": 1.5, "teenager": 1.0},
    # broadly-aged casual basics: keep a real spread (incl. kids) so not everything is adult
    "short_sleeve_top":   {"young adult": 4, "adult": 4, "teenager": 2, "child": 1.5, "senior": 1, "baby": 0.5},
    "vest":               {"young adult": 4, "adult": 3, "teenager": 2, "child": 1.5, "senior": 1},
    "shorts":             {"young adult": 4, "adult": 3, "teenager": 2, "child": 2, "senior": 0.8, "baby": 0.5},
    "skirt":              {"young adult": 5, "adult": 3, "teenager": 2, "child": 1, "senior": 0.8},
}
DEFAULT_AGE_AFFINITY = {a: AGE_GROUP_WEIGHTS[a] for a in AGE_GROUP}  # fallback = global marginal
```

Rewrite `generate_age_groups(item_type=None)`:
- Pick the **primary** group by `random.choices(list(affinity), weights=...)` from
  `TYPE_AGE_GROUP_AFFINITY.get(item_type, DEFAULT_AGE_AFFINITY)` instead of `randint`.
- **Keep** the existing multi-group count logic (50% single / 30% two / …) and the
  adjacency/contiguity logic — only the *primary index* selection changes. (Implementation
  note: today the primary is an *index* into `AGE_GROUP`; switch to choosing the primary
  *value*, then map back to its index, then reuse the unchanged adjacency code.)
- **Back-compat:** default `item_type=None` → use `DEFAULT_AGE_AFFINITY` so any other caller
  keeps working. (Grep shows the only generator caller is `DataGenerator.generate_item`;
  pass `item_type` there.)

> **Adjacency caveat to honor:** because adjacent groups are added around the primary, a
> `young adult` primary can still pull in `teenager`/`adult` (good) but the contiguous logic
> means a primary near an end can include one neighbour. That is *fine* and even desirable for
> spread — what we are killing is `baby`/`senior` as the **primary** on dresses.

### 2.2 `style` — wire in `STYLE_WEIGHTS` + light type conditioning (REQUIRED)
- At minimum: replace `random.choice(valid_styles)` with a **weighted** choice using
  `STYLE_WEIGHTS` restricted to the age-valid styles (renormalize over survivors).
- Recommended (cheap, big coherence win): add a small per-type **boost** so smart types lean
  smart and party/dress types lean elegant/streetwear:

```python
# models.py
STYLE_TYPE_BOOST = {
    "long_sleeve_top":     {"minimalist": 2.5, "smart casual": 2.5, "formal": 1.5, "casual": 1.2},
    "long_sleeve_outwear": {"smart casual": 2, "minimalist": 2, "casual": 1.5, "sporty": 1.2},
    "trousers":            {"smart casual": 2, "minimalist": 1.8, "casual": 1.8, "formal": 1.5},
    "skirt":               {"casual": 1.8, "minimalist": 1.8, "elegant": 1.5, "vintage": 1.3},
    "shorts":              {"casual": 2.5, "sporty": 2, "streetwear": 1.5},
    "short_sleeve_dress":  {"elegant": 2.5, "streetwear": 1.8, "vintage": 1.5, "casual": 1.2},
    "long_sleeve_dress":   {"elegant": 2.5, "vintage": 1.8, "formal": 1.5},
    "vest_dress":          {"elegant": 2, "streetwear": 1.8, "casual": 1.5},
    "sling_dress":         {"elegant": 2.5, "streetwear": 2, "vintage": 1.5},
}
# effective weight(style) = STYLE_WEIGHTS[style] * STYLE_TYPE_BOOST.get(type,{}).get(style,1.0)
```
Add a helper `get_weighted_style_for_type(item_type, age_groups_str)` in `models.py` that:
1. builds `valid = [s for s in STYLE if filter_by_age_appropriateness("style", s, age)]`,
2. weights each by `STYLE_WEIGHTS[s] * boost`, 3. `random.choices`. Call it from the generator.

> **Order dependency:** `style` must be chosen **after** `age_group` (age filters styles) and
> **before** `pattern` (pattern conditions on style). Keep that order (it already holds).

### 2.3 `color` — leave it alone (RECOMMENDED), with a caveat
- The doc says do **not** touch color: a red dress is perfectly coherent, color independence
  is *not* the coherence problem, and color is the key for `image_url` (see §5 — holding color
  to the existing enum keeps `image_url` regenerable without an image pipeline change).
- Keep `color = random.choice(STATIC_GLOBAL_FIELDS["color"])` (uniform over all 19 colors).
  Uniform-19 actually *helps* persona coverage: each persona names ~5 colours, so ~26% of
  items already fall in-palette by color alone.
- **If** the implementer insists on reweighting color, they MUST first extend `COLOR_WEIGHTS`
  to all **19** colors — it currently lists only **15**, omitting `burgundy/olive/teal/coral`,
  and **`olive` is a casual_sofia persona colour** (it would be ~5× suppressed via the `0.01`
  default in `weighted_choice`). Default recommendation: **don't reweight color.**

### 2.4 `fit` — optional (leave uniform OR small affinity)
There is **no `FIT_WEIGHTS`** and no type/style→fit table. `fit` feeds `body_type` (via
`FIT_BODYTYPE_AFFINITY`) but no persona filters on `fit` directly. Cheapest correct choice:
**leave `fit` uniform.** If desired, add a tiny `STYLE_FIT_AFFINITY` (sporty→athletic/relaxed,
elegant→fitted/tailored, minimalist→regular/slim fit) and weight the pick. **Mark optional.**

### 2.5 Sharpen existing weighted helpers (REQUIRED, but gently)
The three weighted helpers use a flat **3×** preferred-vs-other ratio:
- `get_weighted_pattern_for_style` (models.py L1208)
- `get_weighted_material_for_season` (L1183)
- `get_weighted_season_for_type` already uses per-type weights (sharper) — leave as is.

Raise pattern/material preferred ratio from `3.0` to **~6.0** so conjunctions concentrate.
**Do NOT over-concentrate** (do not go to 1.0 vs 0 / hard filter) — Workstream B's `veto_batch`
needs spread within the band or it starves (see §9). 6× keeps ~85–90% of mass on preferred
values while leaving a real tail.

### 2.6 `DataGenerator.py` edits
- Add module constant `SEED` and **seed the module-level `random`** at the top of `main()`
  (and in any new quota driver) so the JSON is reproducible (see §7). The whole chain uses the
  *module-level* `random`, so one `random.seed(SEED)` makes the entire dataset deterministic.
- Change `N = 10000` → `N = 500` (or read from CLI/env — see §11 open question on exact N).
- In `generate_item()`:
  - `age_group = generate_age_groups(item_type)` (pass type).
  - replace the `style` block with `style = get_weighted_style_for_type(item_type, age_group)`.
  - leave `color`/`fit` as `random.choice` (per §2.3/§2.4) unless implementer opts in.
  - everything else (gender constraint, season/material/pattern/occasion chain, type-specific
    fields, `generate_body_types`) stays — it already conditions correctly.

> **Do NOT change** the schema, the enum tuples (`TYPE/COLOR/STYLE/...`), `GLOBAL_FIELDS`,
> `TYPE_FIELDS`, `EXTRA_FIELD_VALUES`, brand/price tables, or `generate_body_types`. Coherence
> comes from *distribution* changes only. Every generated value MUST stay ∈ its enum (§8
> asserts this).

---

## 3. Persona goal-bands — verified against `customers.json` AND the enums

Retrieval semantics (verified in `stock_agent/stock_agent.py::_field_match`): the OR-band and
candidate filters use **exact equality (`isin`) for every key except `age_group`**, which uses
case-insensitive substring (fine for the comma-joined column). `body_type` is **stripped**
before the stock query (`multi_agent/retrieval.py::_prune_to_stock_keys`), so it does NOT gate
retrieval — but the reviewer LLM still *reads* it, so we keep it coherent. **All persona
filter values below were confirmed present in the `models.py` enums.**

| Persona | type(s) | colors | style | occasion | gender | body_type (reviewer-visible) |
|---|---|---|---|---|---|---|
| **party_maya** | `short_sleeve_dress` (+ other `*dress*` ok) | red, orange, pink, coral, multicolor | elegant, streetwear, vintage | party, **date night**, wedding | female (→ female/unisex) | hourglass |
| **office_daniel** | `long_sleeve_top` | navy, gray, white, beige, black | minimalist, smart casual *(=clean/smart-casual)* | work | male (→ male/unisex) | trapezoid |
| **casual_sofia** | trousers, skirt, shorts | blue, black, olive, gray, beige | casual, minimalist | *(no occasion filter in goal; "everyday")* | female (→ female/unisex) | pear |

Enum cross-checks (all PASS):
- `short_sleeve_dress`, `long_sleeve_top`, `trousers`, `skirt`, `shorts` ∈ `TYPE`. ✔
- `red/orange/pink/coral/multicolor`, `navy/gray/white/beige/black`, `blue/olive` ∈ `COLOR`. ✔
  (`olive` IS in `COLOR` — the only risk is reweighting color, see §2.3.)
- `elegant/streetwear/vintage`, `minimalist/smart casual`, `casual` ∈ `STYLE`. ✔ (note the enum
  spelling is **`smart casual`** with a space — match exactly.)
- `party/date night/wedding`, `work` ∈ `OCCASION`. ✔ (enum is **`date night`**, not "date".)
- `hourglass`/`pear` ∈ `FEMALE_BODY_TYPES`; `trapezoid` ∈ `MALE_BODY_TYPES`. ✔ — so the
  gender constraint is already consistent (maya/sofia female-typed → female body types;
  daniel male → trapezoid). `generate_body_types` will produce these for the right genders.

> **Gotcha for quotas:** `short_sleeve_dress`/`long_sleeve_dress`/`vest_dress`/`sling_dress`
> are gender-constrained to **female/unisex** (`GENDER_CONSTRAINTS_BY_TYPE`). Good for maya.
> `long_sleeve_top` allows all genders — quota items for daniel must be forced to `male`
> (or unisex) so the soft-gender include keeps them.

---

## 4. THE KEY RISK — guaranteeing persona-band coverage at N=500 (quota generation)

**Problem.** Random coherent generation at N=500 will, by the §1.4 math, leave
office_daniel's band near-empty (~0–2 items) and the others thin. That defeats the purpose.

**Solution — stratified / quota generation.** Generate the catalog in two layers:
1. **Quota layer (guaranteed):** explicitly generate a fixed number of items *inside each
   persona's goal-band*, by constraining `type`/`color`/`style`/`occasion`/`gender` to the
   persona's allowed sets and letting the rest of the coherent chain fill in the remaining
   attributes. This guarantees coverage regardless of the random draw.
2. **Background layer (distractors / spread):** generate the remaining items with the normal
   coherent generator (no persona constraint). These provide off-band distractors so
   retrieval/veto has spread and the recommender still has to *choose* (not trivially win).

### 4.1 Concrete target counts (N = 500)
| Bucket | Target items | How |
|---|---|---|
| party_maya in-band | **70** | type ∈ {short_sleeve_dress, long_sleeve_dress, vest_dress, sling_dress}; color ∈ {red,orange,pink,coral,multicolor}; style ∈ {elegant,streetwear,vintage}; occasion ∈ {party,date night,wedding}; gender ∈ {female,unisex}; age primary young adult/adult |
| office_daniel in-band | **70** | type = long_sleeve_top; color ∈ {navy,gray,white,beige,black}; style ∈ {minimalist,smart casual}; occasion = work; gender ∈ {male,unisex}; age adult/young adult |
| casual_sofia in-band | **70** | type ∈ {trousers,skirt,shorts}; color ∈ {blue,black,olive,gray,beige}; style ∈ {casual,minimalist}; gender ∈ {female,unisex}; age young adult/adult |
| Background (coherent, unconstrained) | **~290** | normal `generate_item()` |
| **Total** | **500** | |

Rationale for ~70 each: retrieval keeps only the **best-stocked size per item** and requires
`stock_count>0`, and `seed_stock` marks ~5% never-sold and clips some sizes to 0 stock. ~70
in-band items comfortably yields **dozens of in-stock distinct candidates per persona** even
after stock attrition — an order of magnitude above today's office_daniel=2. It also leaves
~290 background items so the OR-band (color OR type OR price) is still wide and veto has spread.
These counts are a **starting point**; §8 validation re-counts post-build and §11 lets the user
tune N / quota sizes.

### 4.2 How to implement the quota layer (no schema/enum changes)
Add a `generate_constrained_item(constraints: dict)` to `DataGenerator.py` that mirrors
`generate_item()` but, where a field is constrained, draws from the constrained subset and
otherwise calls the same coherent helpers. Critically, **still run the coherent chain** for
unconstrained fields and **recompute `body_type` last** so quota items stay internally
coherent. Pseudo:

```python
def generate_constrained_item(c):
    item_type = random.choice(c.get("type", STATIC_GLOBAL_FIELDS["type"]))
    genders = [g for g in get_valid_genders_for_type(item_type) if g in c.get("gender", GENDER)] \
              or list(get_valid_genders_for_type(item_type))
    gender = random.choice(genders)
    age_group = generate_age_groups(item_type)                      # coherent, type-aware
    season   = get_weighted_season_for_type(item_type)
    material = get_weighted_material_for_season(season)
    style    = random.choice(c["style"]) if "style" in c else get_weighted_style_for_type(item_type, age_group)
    pattern  = get_weighted_pattern_for_style(style)               # + age re-roll loop as today
    occasion = random.choice(c["occasion"]) if "occasion" in c else get_valid_occasion_for_type(item_type)
    color    = random.choice(c["color"]) if "color" in c else random.choice(STATIC_GLOBAL_FIELDS["color"])
    fit      = random.choice(STATIC_GLOBAL_FIELDS["fit"])
    # ... assemble, brand←type, price←type+brand, type-specific extras, body_type LAST ...
```
Then a `generate_dataset_with_quotas(n)` driver: emit the three quota buckets, then fill the
rest with `generate_item()`, assign ids 1..n, **shuffle once** (so ids aren't bucket-ordered),
and write JSON. Seed `random` first (§7).

> **Keep occasion age-valid:** the quota for daniel forces `occasion="work"`, which is
> age-inappropriate for `baby`/`child` — but daniel quota ages are adult/young-adult, so the
> existing `AGE_INAPPROPRIATE_OCCASIONS` filter won't fire. Still, assert validity in §8.

---

## 5. Derived-artifact scope decision

**Recommendation: experiment-only scope.** Do **not** regenerate images, descriptions, or
Qdrant as part of this milestone. Reasons:
- The SPADE experiment path reads only structured columns (verified: `_SCORED_QUERY` in
  `stock_agent/stock_stats.py` selects `color,type,fit,season,style,pattern,material,gender,
  age_group,occasion,brand,price` + stock fields — never `image_url`/`short_description`).
- `image_url`/`short_description`/Qdrant feed the **separate** chat-search app
  (`LNIAGIA/search_app.py`, `src/api/main.py`), which is out of scope for the experiment.

**Document the staleness:** after rebuild, `image_url`/`short_description` are **not created
(absent columns, not NULL)** and Qdrant is orphaned; the chat-search app will be broken until
refreshed. Put a one-line note in the experiment README / metric-epoch marker.

**If full repo consistency is later required** (chat-search app must work), the *additional*
steps are:
1. `python LNIAGIA/DB/SQLLite/update_images_and_description.py` — would regenerate BOTH
   `image_url` (color+type → CSV-mapped Drive URL) and `short_description` (and `ALTER TABLE
   ADD COLUMN`s them back, since the rebuild pipeline doesn't create them). It maps via
   `COLOR_TO_CSV_COLOR_MAP` (all 19 colors covered) keyed on `color`+`type`; because we
   **hold `color` to the existing enum** (§2.3), every `image_url` would stay valid — the
   color-fixed shortcut holds.
   > ⚠️ **BLOCKED — source CSV is missing.** This script requires a source CSV at
   > `DataSources/clothing_full (1).csv` (default), with fallbacks
   > `Images/clothing_full.csv` and `DataSources/clothing_full.csv`. **None of these exist in
   > the repo** (verified by glob — `DataSources/` itself is absent; the only `.csv` present is
   > `models/weights/yolov8n_fashion/results.csv`, unrelated). With no CSV the script raises
   > `FileNotFoundError` and the image-regeneration path **cannot run**. Therefore the
   > experiment-only scope is the default and the full-consistency option is **currently not
   > executable for images** until that CSV is supplied. (The `short_description` half is
   > generated from structured attributes, but the script as written loads the CSV first, so it
   > also fails before reaching descriptions.)
2. Re-embed Qdrant via `LNIAGIA/DB/vector/VectorDBManager.py` (+ `description_generator.py`).
   Heavier; only needed for vector search. Out of scope here.

---

## 6. Backups & safety

Before ANY destructive step:
1. **Checkpoint WAL + copy the DB.** WAL sidecars (`clothing.db-wal`, `clothing.db-shm`) hold
   uncommitted pages; copy them too, or checkpoint first. The repo already has a safe pattern
   (`migrate_stock_schema.py::_backup_db`): `PRAGMA wal_checkpoint(TRUNCATE)` then `copy2`.
   Manual equivalent (PowerShell), do this FIRST:
   ```powershell
   $ts = Get-Date -Format "yyyyMMdd_HHmmss"
   python -c "import sqlite3; c=sqlite3.connect(r'LNIAGIA/DB/SQLLite/clothing.db'); c.execute('PRAGMA wal_checkpoint(TRUNCATE)'); c.close()"
   Copy-Item "LNIAGIA/DB/SQLLite/clothing.db" "LNIAGIA/DB/SQLLite/clothing.db.bak-$ts"
   ```
   (`migrate_stock_schema.py` ALSO auto-writes its own `.bak-<ts>` when it runs in step 3 — so
   you get a second safety net, but make the manual backup anyway, before `populate_db` drops
   the table.)
2. Optionally keep the current 10k DB permanently as `clothing.db.10k-backup` (see §11).
3. **Rollback:** stop any running experiment/app, delete the new `clothing.db`
   (+ `-wal`/`-shm`), restore the `.bak-<ts>` copy back to `clothing.db`. The DB is the only
   mutated artifact (the generator only *adds* a new JSON under `DataSources/`).

> The new catalog **REPLACES** `LNIAGIA/DB/SQLLite/clothing.db` in place. Many modules hardcode
> that exact path (`stock_agent/stock_stats.py`, `stock_agent/seed_stock.py`,
> `multi_agent/agents/body_agent.py`, `src/api/main.py`, …). A new-path catalog would require
> editing all of them — not worth it. Replace in place (after backup). See §11.

---

## 7. Reproducibility / idempotence

- **One seed governs the catalog.** The generator chain uses the *module-level* `random`, so
  `random.seed(SEED)` once at the start of the dataset driver makes the entire JSON
  deterministic. Pick a fixed `SEED` (e.g. 1337) and record it.
- **The JSON artifact is the source of truth.** `DataSources/<timestamp>.json` is the exact
  catalog; `populate_db` is a pure load. Re-running `populate_db` on the *same* JSON yields an
  identical `items`/`users`-insert order (users are random per run — see caveat below).
- **`seed_stock.py` is separately seeded** (`--seed 42` default, numpy RNG). Same `--seed` ⇒
  identical stock. So `(catalog SEED, stock --seed)` fully pins the experiment-relevant DB.
- **Caveats:**
  - `DBManager.create_random_users` uses the *unseeded* module `random` at populate time, so
    the 10 `users` rows differ run-to-run. Users are irrelevant to the experiment; ignore, OR
    seed `random` right before calling `populate_db` if byte-identical users are wanted.
  - `populate_db` is **interactive** in its menu; drive it programmatically (§10) to avoid
    `input()`.

- **Idempotence / how to regenerate.** To regenerate the catalog from scratch, re-run **all**
  rebuild phases in order (generator → `populate_db(recreate=True)` → `migrate_stock_schema.py`
  → `seed_stock.py`). There is no partial-rebuild shortcut: `populate_db(recreate=True)` drops
  and recreates `items`/`users` (so `created_at` and the derived columns are gone again until
  the later phases run), and **`seed_stock.py` requires `--force` on every re-run** — without
  it the seeder refuses to overwrite existing stock data.
- **`created_at` is empty until `seed_stock` runs.** `migrate_stock_schema.py` only *adds* the
  `created_at` column (all NULL); `seed_stock.py` is what populates it. Consequence:
  `stock_agent/stock_stats.py` filters on `WHERE created_at IS NOT NULL`, so if it (or any
  validation that scopes on `created_at`) is run **between** `populate_db` and `seed_stock` it
  returns an **empty frame** — not a bug, just an ordering artifact. **Run validation only
  after the full pipeline completes** (i.e. after `seed_stock --force`).

---

## 8. Validation (Windows-friendly — Python only; `sqlite3` CLI is NOT installed)

> Confirmed: `shutil.which("sqlite3")` is `None` on this machine. Use `python -c "import
> sqlite3 ..."`. The doc's `sqlite3 ...` one-liners must be translated to Python.

Run these after the full pipeline. Save as a scratch snippet or paste inline.

**(a) Coherence — age_group is now type-correlated:**
```python
python -c "import sqlite3; c=sqlite3.connect(r'LNIAGIA/DB/SQLLite/clothing.db'); cur=c.cursor(); \
print('dress age_group:', cur.execute(\"SELECT age_group,COUNT(*) FROM items WHERE type LIKE '%dress%' GROUP BY 1 ORDER BY 2 DESC\").fetchall()); \
print('baby/senior dresses (expect ~0):', cur.execute(\"SELECT COUNT(*) FROM items WHERE type LIKE '%dress%' AND (age_group LIKE 'baby%' OR age_group LIKE 'senior%')\").fetchone()[0])"
```
Expect: dress primaries dominated by young adult/adult; baby+senior-primary dresses ≈ 0.

**(b) Persona-band conjunction counts (must be materially > §1.4 baseline, esp. daniel):**
```python
python -c "import sqlite3; c=sqlite3.connect(r'LNIAGIA/DB/SQLLite/clothing.db'); cur=c.cursor(); q=cur.execute; \
print('party_maya', q(\"SELECT COUNT(*) FROM items WHERE type LIKE '%dress%' AND color IN('red','orange','pink','coral','multicolor') AND occasion IN('party','date night','wedding') AND style IN('elegant','streetwear','vintage') AND gender IN('female','unisex')\").fetchone()[0]); \
print('office_daniel', q(\"SELECT COUNT(*) FROM items WHERE type='long_sleeve_top' AND color IN('navy','gray','white','beige','black') AND style IN('minimalist','smart casual') AND occasion='work' AND gender IN('male','unisex')\").fetchone()[0]); \
print('casual_sofia', q(\"SELECT COUNT(*) FROM items WHERE type IN('trousers','skirt','shorts') AND color IN('blue','black','olive','gray','beige') AND style IN('casual','minimalist') AND gender IN('female','unisex')\").fetchone()[0])"
```
Acceptance: each ≥ ~40 (target ~70 minus stock attrition); **office_daniel must be ≥ 40**
(vs 2 baseline). If daniel is low, raise its quota in §4.1 and regenerate.

**(c) In-stock distinct-item coverage (closer to what retrieval actually sees):** repeat (b)
but JOIN `item_stock` with `stock_count>0 AND active=1` and `COUNT(DISTINCT i.id)` — confirms
the band survives stock attrition.

**(d) Enum-membership assertion (no value drifted outside its enum):**
```python
python -c "
import sqlite3, sys; sys.path.insert(0, r'LNIAGIA/DB');
import models as M
c=sqlite3.connect(r'LNIAGIA/DB/SQLLite/clothing.db'); cur=c.cursor()
checks={'type':M.TYPE,'color':M.COLOR,'style':M.STYLE,'pattern':M.PATTERN,'material':M.MATERIAL,'fit':M.FIT,'gender':M.GENDER,'season':M.SEASON,'occasion':M.OCCASION}
bad=0
for col,allowed in checks.items():
    vals=[r[0] for r in cur.execute(f'SELECT DISTINCT {col} FROM items').fetchall()]
    extra=[v for v in vals if v not in set(allowed)]
    if extra: bad+=1; print('DRIFT', col, extra)
# multi-valued columns: split on comma
for col,allowed in (('age_group',M.AGE_GROUP),('body_type',M.BODY_TYPE)):
    vals=set()
    for (s,) in cur.execute(f'SELECT DISTINCT {col} FROM items').fetchall():
        for p in (s or '').split(','): 
            p=p.strip()
            if p: vals.add(p)
    extra=[v for v in vals if v not in set(allowed)]
    if extra: bad+=1; print('DRIFT', col, extra)
print('ENUM DRIFT OK' if bad==0 else f'{bad} columns drifted')"
```

**(e) Stock pipeline self-check:**
```bash
python stock_agent/stock_stats.py        # must run without raising
python stock_agent/seed_stock.py --force # its own smoke checks must pass (see WARNING below)
```
> ⚠️ **N=500 caveat for seed_stock smoke checks:** `seed_stock._run_smoke_checks` asserts
> **strictly** decreasing mean stock and **strictly** increasing sale-gap across the 4 age
> quartiles. With only ~3,000 stock rows (vs 60,000) the law-of-large-numbers smoothing is
> weaker and those *strict* monotonic asserts could occasionally fail on an unlucky seed. If
> that happens it is a **statistical flake, not a coherence bug**: re-run with a different
> `--seed`, or run `seed_stock --force --no-checks` and verify monotonicity manually as a
> trend. Do NOT weaken the asserts. Flag to the user (this is a known N-shrink side effect).

**(f) FK integrity (no orphan stock rows):**
```python
python -c "import sqlite3; c=sqlite3.connect(r'LNIAGIA/DB/SQLLite/clothing.db'); \
print('orphans', c.execute('SELECT COUNT(*) FROM item_stock s LEFT JOIN items i ON i.id=s.item_id WHERE i.id IS NULL').fetchone()[0]); \
print('items', c.execute('SELECT COUNT(*) FROM items').fetchone()[0], 'stock_rows', c.execute('SELECT COUNT(*) FROM item_stock').fetchone()[0])"
```
Expect orphans=0; stock_rows ≈ items×6.

**(g) End-to-end review-lift smoke (USER RUNS — needs XMPP + Ollama + GPU):**
```bash
docker compose up -d xmpp
EXPERIMENT_MODE=ofat EXPERIMENT_REPEATS=1 python -m multi_agent.experiments.run_experiment
python -c "import sqlite3; c=sqlite3.connect(r'multi_agent/experiments/results.db'); \
print(c.execute('SELECT customer_id, AVG(final_review), COUNT(*) FROM episodes WHERE final_review IS NOT NULL GROUP BY 1').fetchall())"
```
Expect mean review trending **above ~2–2.5**. This is the real acceptance signal but it is the
**user's to run** (the agent pipeline cannot start XMPP/Ollama/GPU). Treat (a)–(f) as the
agent-checkable gate; (g) as the user-confirmed outcome.

---

## 9. Risks / edge-cases

- **Quota too small ⇒ purpose defeated.** office_daniel is the canary (baseline 2/10k). Always
  re-run §8(b)/(c) after build; if any persona's in-stock distinct count is <~40, raise that
  bucket's quota and regenerate. This is the single most important check.
- **Over-concentration ⇄ Workstream B starvation.** If pattern/material ratios or style boosts
  are pushed too hard (or to hard filters), the relevance band narrows and `veto_batch` can't
  accumulate survivors. Keep ratios ≤ ~6× and keep ≥2–3 live values per conditioned field.
  Background layer (~290 unconstrained coherent items) deliberately preserves spread.
- **Enum drift** silently breaks `get_candidates`/`get_random_batch` (`isin`/substring filters
  match nothing). §8(d) asserts membership — run it every time. Spelling traps: `smart casual`
  (space), `date night` (space), `all-season` (hyphen), `organic cotton` (space).
- **Gender/type coherence for quotas:** dresses are female/unisex-only (enforced by
  `get_valid_genders_for_type`); daniel's `long_sleeve_top` must be forced male/unisex so the
  soft-gender include keeps it.
- **seed_stock strict-monotonic smoke flake at N=500** — see §8(e) WARNING. Statistical, not a
  bug.
- **Derived staleness** (`image_url`/`short_description` columns absent — not created by the
  rebuild pipeline — and Qdrant orphaned) — accepted for experiment-only scope; document for
  the chat-search app (§5).
- **Metric epoch reset** — post-change runs are NOT comparable to experiments #5–#12; add the
  epoch marker (Workstream D2).
- **`populate_db` interactivity** — never let it block on `input()`; drive it programmatically
  (§10). It also re-rolls 10 random `users` each run (harmless; §7).
- **WAL sidecars on backup/rollback** — checkpoint or copy `-wal`/`-shm` too (§6), else a
  "restore" can silently keep new data.
- **N as a magic number** — `DataGenerator.N` and the quota counts are hardcoded; if the user
  picks a different N, quotas must scale (keep in-band buckets ≥ ~40 each). See §11.

---

## 10. Step-by-step task list (implementer follows top→bottom)

> All commands assume CWD = repo root. Windows/PowerShell shown; the Bash tool also works.

**Phase 0 — branch & backup**
1. Confirm on branch `rl-learning-curve` (`git status`). Do NOT touch `main`.
2. Make the manual DB backup (§6 step 1). Optionally also copy to `clothing.db.10k-backup`.

**Phase 1 — generator coherence (`LNIAGIA/DB/models.py`)**
3. Add `TYPE_AGE_GROUP_AFFINITY` + `DEFAULT_AGE_AFFINITY` near `AGE_GROUP_WEIGHTS` (§2.1).
4. Rewrite `generate_age_groups(item_type=None)` to draw the primary group from the affinity
   table; keep the multi-group/adjacency logic (§2.1).
5. Add `STYLE_TYPE_BOOST` + `get_weighted_style_for_type(item_type, age_groups_str)` using
   `STYLE_WEIGHTS × boost` over age-valid styles (§2.2).
6. Bump preferred ratio in `get_weighted_pattern_for_style` and
   `get_weighted_material_for_season` from `3.0` → `6.0` (§2.5). Leave color/fit/season as is.
7. (Optional) add `STYLE_FIT_AFFINITY` + weighted fit (§2.4) — skip unless time permits.

**Phase 2 — generator + quotas (`LNIAGIA/DB/SQLLite/DataGenerator.py`)**
8. Add `SEED` constant; `random.seed(SEED)` at start of the dataset driver (§7). Set `N = 500`
   (or wire to CLI/env — §11).
9. In `generate_item()`: `age_group = generate_age_groups(item_type)`; replace style pick with
   `get_weighted_style_for_type(...)` (§2.6). Leave color/fit as `random.choice`.
10. Add `generate_constrained_item(constraints)` (§4.2) and
    `generate_dataset_with_quotas(n)` that emits the 3 persona buckets (70/70/70) + ~290
    background items, assigns ids 1..n, shuffles once. Point `main()` at it.
11. Run the generator to produce the JSON:
    ```powershell
    python -m LNIAGIA.DB.SQLLite.DataGenerator   # or: python "LNIAGIA/DB/SQLLite/DataGenerator.py"
    ```
    Note the new `DataSources/<timestamp>.json` filename.

**Phase 3 — load the new catalog (non-interactive drive of `populate_db`)**
12. Drive `populate_db(json, recreate=True)` programmatically (avoids the `input()` menu):
    ```powershell
    python -c "import sys; sys.path.insert(0, r'LNIAGIA/DB/SQLLite'); sys.path.insert(0, r'LNIAGIA/DB'); \
    import glob, os; from DBManager import populate_db; \
    f=max(glob.glob(r'LNIAGIA/DB/SQLLite/DataSources/*.json'), key=os.path.getmtime); \
    print('loading', f); populate_db(f, recreate=True)"
    ```
    Verify: `items` ≈ 500 rows, `users` = 10, and **30 columns at this point**
    (`id` + 29 structured columns from `_all_columns()`). The pipeline does **not** create
    `image_url`/`short_description`/`created_at` here. After Phase 4's
    `migrate_stock_schema.py` adds `created_at`, the count becomes **31**. (The live 10k DB has
    33 only because `update_images_and_description.py` ran historically and added the two
    derived columns — that script is out of scope here; see §5.) Re-count after migrate:
    ```powershell
    python -c "import sqlite3; c=sqlite3.connect(r'LNIAGIA/DB/SQLLite/clothing.db'); print('items cols:', len(c.execute('PRAGMA table_info(items)').fetchall()))"
    ```
    Expect **30** right after this step, **31** after Phase 4 migrate.

**Phase 4 — stock pipeline**
13. Migrate schema (auto-writes a `.bak`): `python stock_agent/migrate_stock_schema.py`
14. Seed stock + events (force-wipe old): `python stock_agent/seed_stock.py --force`
    - If strict-monotonic smoke asserts flake (§8(e) WARNING): re-run with `--seed 7`
      (or another), or `--force --no-checks` + manual trend check. Do NOT weaken asserts.

**Phase 5 — validate (agent-checkable)**
15. Run §8 (a) coherence, (b) persona conjunctions, (c) in-stock coverage, (d) enum drift,
    (f) FK integrity, and `python stock_agent/stock_stats.py`. All must pass; daniel ≥ ~40.
16. If a persona band is thin → raise its quota (§4.1), re-run Phase 2→5.

**Phase 6 — docs & handoff (no commit unless user asks)**
17. Add a metric-epoch note (experiment README) + the derived-artifact staleness note (§5).
18. Show the diff to the user before committing. If approved: commit on `rl-learning-curve`,
    author `tiagosgdev`, **no Claude co-author trailer**. Do NOT push to `main`.
19. Hand off §8(g) end-to-end review-lift smoke to the user (needs XMPP+Ollama+GPU).

---

## 11. Open questions / decisions for the user

1. **Exact N.** Plan assumes **N=500** with quotas 70/70/70 + ~290 background. Confirm, or
   pick another N (quotas must scale so each in-band bucket stays ≥ ~40 in-stock items).
2. **Replace in place vs new path.** Plan **replaces** `LNIAGIA/DB/SQLLite/clothing.db` in
   place (many modules hardcode that path). Acceptable, or do you want a config-driven path
   (larger change touching `stock_stats.py`, `seed_stock.py`, `body_agent.py`, `src/api/main.py`)?
3. **Keep the 10k DB as a permanent backup?** (e.g. `clothing.db.10k-backup`) — recommended so
   you can A/B old-vs-new, but it's extra disk. Default: yes, keep it.
4. **Derived-artifact scope.** Plan recommends **experiment-only** (skip image/description/
   Qdrant regen; chat-search app goes stale). Note the full-consistency path is **currently not
   executable for images**: `update_images_and_description.py` needs a source CSV
   (`DataSources/clothing_full (1).csv` or the fallbacks) that is **not present in the repo**
   (§5), so it would fail with `FileNotFoundError`. To enable full repo consistency the user
   must first supply that CSV; only then can `image_url`/`short_description` be regenerated
   (color is held fixed so `image_url` stays valid). Qdrant re-embed remains optional and
   independent.
5. **Color & fit.** Plan **leaves color uniform** (don't reweight; protects `olive` +
   `image_url`) and **leaves fit uniform** (no FIT_WEIGHTS). OK, or do you want the optional
   `STYLE_FIT_AFFINITY`?
6. **Concentration strength.** Plan uses ~6× preferred ratios + moderate style boosts to avoid
   starving Workstream B's `veto_batch`. If you are *not* running `veto_batch`, you could push
   sharper. Confirm which `SELECTION_MODE` the next runs use.
7. **`seed_stock` strict-monotonic asserts at N=500.** Accept "re-roll seed on flake" as the
   policy, or should the implementer add an N-aware tolerance (a code change to `seed_stock`)?
