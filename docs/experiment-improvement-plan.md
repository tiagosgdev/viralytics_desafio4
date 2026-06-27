# Experiment-Harness Improvement Plan

> **Status:** PLAN ONLY — nothing here is implemented yet. Two people can take one
> workstream each (they touch disjoint files). Each workstream is self-contained so a
> fresh Claude Code session can implement it from this doc alone.
>
> Produced by a plan agent + reviewed by a second agent; the review's corrections are
> baked in (look for **[review]** notes where the first-pass plan was wrong).

---

## 0. How to use this document

- **Split:** Workstream **A** (catalog) and **B** (selection knobs) are fully independent
  (disjoint files, no merge conflicts). Assign one person each. **C** is docs-only. **D**
  are optional suggestions.
- **To brief Claude on a part:** point it at this file and say e.g. *"Implement Workstream B
  from `docs/experiment-improvement-plan.md`. Read the referenced files first, follow the
  acceptance criteria, don't touch other workstreams."* The Context (§1) + Background (§2)
  give it everything it needs.
- **Workflow:** plan → implement (fresh subagent) → review (fresh subagent) → show before
  commit. Branch `rl-learning-curve` only, **never push to main**. Commits authored
  `tiagosgdev`, **no Claude co-author trailer**. Do **not** `pip install -r requirements.txt`
  (it pins numpy<2 and breaks the env); the system python already has what's needed.

---

## 1. Project context

- **Repo:** `/home/tsg/Projects/viralytics_desafio4`, branch `rl-learning-curve`.
- **System:** a SPADE multi-agent clothing recommender + an experiment harness that drives it
  with a simulated LLM shopper. Five scorer agents (`body`, `clothing`, `colour`, `stock`,
  `rl`) propose item rankings; the orchestrator aggregates them (Borda, or a random-batch +
  weighted-veto loop) into a final top-10; a simulated shopper (qwen2.5:14b via Ollama) holds a
  multi-turn chat and gives a closing **1–5 review** that is the primary metric.
- **Run prereqs:** `docker compose up -d xmpp` (broker); Ollama up with `qwen2.5:14b-instruct`
  + `qwen2.5:7b-instruct-q3_K_M`. GPU box.
- **Experiment modes** (`EXPERIMENT_MODE`): `ofat` (default) | `full` (factorial grid) |
  `curve` (RL-learning-from-satisfaction). `SELECTION_MODE`: `borda` (default) | `veto_batch`.

### What's already been done this session (do not redo)
- Fixed a veto_batch reward-drop bug (commit `2fac08e`).
- Recalibrated the shopper review rubric to give partial credit (commit `af3b41d`).
- Broadened the 3 personas in `customers.json` to well-stocked, **filterable** attribute sets
  (commit `bf26765`).
- **Metric epoch warning:** those two commits (and Workstream A below) each **reset the primary
  review metric**. Runs after them are **not comparable** to historical experiments #5–#12.

---

## 2. Background — why these changes (findings established this session)

1. **Catalog incoherence is the root cause of low reviews.** The catalog
   (`LNIAGIA/DB/SQLLite/clothing.db`, table `items`, 10k rows) has several attributes assigned
   **independently of item type**, producing nonsensical items (e.g. ~900 of ~3,600 "dresses"
   tagged `age_group=baby`; "metallic/graphic" is ~5% of dresses; a single item can be a "baby
   travel camouflage organic-cotton streetwear dress"). Because attributes are independent,
   any 3–4-attribute conjunction matches only ~tens of 10k rows, so the recommender can't
   satisfy a specific shopper goal → reviews cap at ~2–2.5. **[review]** The generator
   (`LNIAGIA/DB/models.py`) *already* conditions season/material/pattern/occasion/body_type on
   type — only **`age_group`, `style`, `color`, `fit`** bypass it. → Workstream A.
2. **The veto selection's consideration set is small.** In `veto_batch`, agents draw a broad
   *random* 40-item batch, veto eliminates many (often ~30/40), the loop **stops as soon as
   ≥10 survive** (`orchestrator.py:414`), then Borda-ranks the pool → top 10. Good items are
   sparse in a small random pool. Enlarging the veto-passed pool before ranking is the most
   promising selection-side lever. → Workstream B.
3. **RL learns from the wrong signal in normal runs.** In `ofat`/`full` the 1–5 review is
   recorded as the eval metric but **not** fed to RL; RL learns from the **pass-rate** reward
   (did its picks reach the final top-10 — i.e. conformity to the committee). Only `curve` mode
   feeds the review, and that signal is degenerate (reviews pile at 2; reward `(r-3)/2` makes
   ~99% of it ≤0 → flat learning). → Workstreams C & D.
4. **The episode loop is persona-major** (all of persona A, then B, then C), which confounds any
   early-vs-late / time-ordered analysis with persona identity. → Workstream D.

---

## 3. Workstream A — Catalog coherence  (Owner: ____________)

**Files:** `LNIAGIA/DB/models.py`, `LNIAGIA/DB/SQLLite/DataGenerator.py`, and a **new**
`LNIAGIA/DB/SQLLite/repair_catalog_coherence.py`. Isolated to `LNIAGIA/DB/` — no overlap with B.

### Goal
Make `items` semantically coherent (attributes conditioned on type/style) so persona-shaped
conjunctions match many more rows and reviews rise above the ~2–2.5 ceiling — **without**
changing the table/columns, `item_stock`/FK, `id`/`price`/`created_at`, or breaking the
retrieval filters and RL feature extractor.

### Scope decision — what actually needs to change **[review-corrected]**
- The **SPADE experiment only reads the structured columns** of `items` (filters use
  `color/type/style/occasion/pattern/...`; the reviewer/agents see those attributes). So for the
  experiment objective, **only the structured columns matter.**
- `short_description`, `image_url`, and the **Qdrant** vector embeddings are derived
  (description ← many attrs; image ← color+type) and feed a **separate** chat-search app
  (`LNIAGIA/search_app.py`), **not** the experiment path. They go stale if you mutate the
  attrs they encode. **Decide explicitly:**
  - **Minimum (experiment-only):** repair structured columns; leave description/image/Qdrant
    stale; document the search-app inconsistency. Lowest risk/effort.
  - **Full repo consistency:** after repairing attrs, re-run
    `LNIAGIA/DB/SQLLite/update_images_and_description.py` (regenerates `short_description` +
    `image_url`) and re-embed Qdrant (`LNIAGIA/DB/vector/`). Needed only if the search app must
    stay correct. **Tip:** if you **hold `color` fixed** (color independence isn't the coherence
    problem — a red dress is fine), `image_url` (color+type-keyed) stays valid and you avoid the
    image regen; only `short_description`+Qdrant would need refresh.

### Approach options
- **(i) Fix the generator + full regenerate** (`DataGenerator.main()` → `populate_db(recreate=True)`):
  cleanest long-term, but `recreate=True` **drops `items`** (loses `created_at`, `body_type`,
  `users`), forces re-migrate (`migrate_stock_schema.py`) + re-seed of 60k `item_stock` rows
  (re-rolls stock + push_scores → also resets the stock-side metric), and stales Qdrant. Highest
  blast radius.
- **(ii) In-place idempotent repair script (RECOMMENDED for the experiment):** re-derive the
  incoherent chain per row, holding `id/type/price/created_at/gender` fixed; `UPDATE items …`.
  Keeps ids, `item_stock`, stock scores intact. **[review]** Must also handle the derived
  `short_description`/`image_url`/Qdrant per the scope decision above — they are **not** free.
- **(iii) Import real product data:** max realism, heavy value→enum mapping + id realignment risk.
  Not for this milestone.

**Recommended:** **(ii)**, experiment-only scope first; also fold the same conditioning back into
`models.py` so a future (i) is coherent too.

### Steps (Option ii)
1. **`models.py` — make the four bypassing fields coherent:**
   - **`age_group` (the worst offender):** **[review]** `AGE_GROUP_WEIGHTS` is a *global marginal*
     (already used for *user* profiles), not a type→age table — applying it alone reduces baby
     overall but won't stop baby-*dresses*. To truly fix coherence, **author a small
     `TYPE_AGE_GROUP_AFFINITY` table** (dresses/outwear/work items → adult/young-adult; only
     genuinely kid-plausible types may take baby/child) and rewrite `generate_age_groups()` to
     take `type` and draw the primary group from that table (keep the existing adjacency / multi-
     group logic).
   - **`style`:** wire the **defined-but-unused `STYLE_WEIGHTS`** into selection; optionally
     condition lightly on type.
   - **`color`:** **leave it alone** for the experiment (not a coherence problem, and changing it
     cascades to `image_url`). **[review]** If you *do* reweight color, note `COLOR_WEIGHTS` only
     covers 15 of 19 colors — `burgundy/olive/teal/coral` would be ~5× suppressed, and **olive is
     a casual_sofia persona colour** — so extend the table to all 19 first.
   - **`fit`:** **[review]** there is **no `FIT_WEIGHTS`** and no type/style→fit table. Either
     leave `fit` uniform (acceptable) or author a small affinity table (sporty→athletic/relaxed,
     elegant→fitted/tailored). Mark as optional.
   - Sharpen the existing weighted helpers (`get_weighted_pattern_for_style`,
     `get_weighted_material_for_season`, `get_weighted_season_for_type`) from the current **3×**
     to ~6–8× so conjunctions concentrate (don't over-concentrate — see risks).
2. **`repair_catalog_coherence.py`** (new): for each row, **[review] `random.seed(item_id)`
   before the row** (the chain helpers use the module-level `random`, not a passed rng) for
   idempotence. Hold `type/id/price/created_at/gender(/color)` fixed; re-derive in dependency
   order: `age_group←type → season←type → material←season → style → pattern←style+age →
   occasion←type+age → fit`. **[review] Recompute `body_type` LAST** via
   `generate_body_types(item, rng)` from the updated `fit`/`gender`/cut fields (it depends on
   them). **Assert every written value ∈ its enum** before the `UPDATE`. One transaction;
   re-runnable; print a before/after coherence report.
3. **Back up `clothing.db`** (and `qdrant_storage/` if doing full consistency) before running.
4. (Full-consistency scope only) run `update_images_and_description.py` + Qdrant re-embed.

### Risks / edge-cases
- **Dependent-chain consistency:** changing `style`/`season` without re-deriving
  `pattern`/`material`/`body_type` re-introduces incoherence — re-derive the **whole** chain.
- **Enum drift:** any value outside the `models.py` enums silently breaks `get_candidates` /
  `get_random_batch` filters — assert membership.
- **Over-concentration ⇄ Workstream B:** if conditioning is too sharp the relevance band
  collapses and `veto_batch` starves (see B). Keep multi-value spread.
- **Derived-artifact staleness** (`short_description`/`image_url`/Qdrant) per scope decision.
- **Metric reset:** repairing the catalog resets the review metric again (epoch marker).

### Acceptance criteria
- Row count unchanged; `item_stock` FK intact; `id/gender/price/created_at` byte-identical.
- `age_group` is type-correlated: ≈0 `baby`/`senior` dresses; tops not bulk-tagged baby.
- Persona-band conjunctions (see queries) match materially more rows than the ~tens baseline.
- Every mutated field ∈ its enum; `python stock_agent/stock_stats.py` self-checks pass; a
  `veto_batch` smoke round returns non-empty.
- **Idempotent:** running the repair twice yields an identical DB.
- A baseline smoke grid shows mean review rising above ~2–2.5.

### Validation commands
```bash
# coherence before/after
sqlite3 LNIAGIA/DB/SQLLite/clothing.db "SELECT age_group,COUNT(*) FROM items WHERE type LIKE '%dress%' GROUP BY 1 ORDER BY 2 DESC;"
sqlite3 LNIAGIA/DB/SQLLite/clothing.db "SELECT COUNT(*) FROM items WHERE type LIKE '%dress%' AND age_group LIKE 'baby%';"   # expect ~0
# persona-band conjunction count (party_maya)
sqlite3 LNIAGIA/DB/SQLLite/clothing.db "SELECT COUNT(*) FROM items WHERE type LIKE '%dress%' AND color IN('red','orange','pink','coral','multicolor') AND occasion IN('party','date night','wedding') AND style IN('elegant','streetwear','vintage');"
python stock_agent/stock_stats.py     # must pass self-checks
# end-to-end review lift
EXPERIMENT_MODE=ofat EXPERIMENT_REPEATS=1 python -m multi_agent.experiments.run_experiment
sqlite3 multi_agent/experiments/results.db "SELECT customer_id,AVG(final_review),COUNT(*) FROM episodes WHERE final_review IS NOT NULL GROUP BY 1;"
```

---

## 4. Workstream B — Selection knob `SURVIVOR_TARGET`  (Owner: ____________)

**Files:** `multi_agent/config.py`, `multi_agent/agents/orchestrator.py`. Isolated — no overlap with A.

### Goal
Let `veto_batch` accumulate a **larger veto-passed survivor pool** before the final Borda ranking
(which still returns 10), so the top-10 is chosen from more acceptable candidates. **Default
behavior byte-identical when the knob is unset.**

### Steps
1. **`config.py`** (beside `MAX_BATCHES`/`BATCH_SIZE`, ~L59):
   ```python
   SURVIVOR_TARGET = int(os.environ.get("SURVIVOR_TARGET", str(TOP_K)))  # default == TOP_K == today
   ```
2. **`orchestrator.py`:** import `SURVIVOR_TARGET`; change the stop at **L414**
   `if len(distinct_survivors) >= TOP_K:` → `>= SURVIVOR_TARGET:`. **Leave the final
   `borda_aggregate(..., k=TOP_K)` (L428) and fallback-fill threshold on `TOP_K`** — the returned
   list stays 10; the knob only controls how big the pool grows first.
3. **Logging:** add a round-end summary — `survivors_reached / SURVIVOR_TARGET`,
   `batches_used / MAX_BATCHES`, and stop-reason (`target-met` | `MAX_BATCHES` | `band-exhausted`);
   emit a `WARNING` when `MAX_BATCHES` is hit with pool `< SURVIVOR_TARGET`.
4. **Run config (env only, no default change):**
   `SELECTION_MODE=veto_batch SURVIVOR_TARGET=40 BATCH_SIZE=80 MAX_BATCHES=10`.
   Sizing: survivors/batch ≈ `BATCH_SIZE × (1 − veto_elim_rate)`; at ~0.75 elimination a 40-item
   batch yields ~10 survivors ⇒ ~4 batches to reach 40, so give `MAX_BATCHES` headroom (8–12).
   Bigger `BATCH_SIZE` reaches the target in fewer batches but drains the band faster.

### Risks / edge-cases  **[review-augmented]**
- **veto_batch only** — inert in `borda`; the run **must** set `SELECTION_MODE=veto_batch`.
- **Band exhaustion → starvation:** `get_random_batch` draws an OR-band (color OR type OR price;
  body_type is stripped in `retrieval.py`) ≈ 1,200–1,800 items for the personas — **ample for
  target 40**. Still, add a guard: assert band size ≥ ~3×`SURVIVOR_TARGET` before a run; the
  round-end WARNING (above) surfaces any starvation. Note Workstream A may thin rare-colour bands.
- **Pass-rate reward bias [review]:** `settle_round` scores pass-rate against the final 10; a
  40→10 funnel lowers per-item pass probability and nudges the RL pass-rate reward negative.
  Consider re-tuning `PASSRATE_ANCHOR` (config) if RL behavior matters for the run.
- **Runtime [review]:** more batches = more CFP broadcasts + the ~48s Ollama weight path per round;
  a full grid can blow up wall-clock. Budget it / accept per the user's "don't worry about
  diminishing returns."
- **Last-batch weights:** the pool is ranked with the last batch's agent weights (pre-existing).

### Acceptance criteria
- Unset knob → byte-identical round (same ≥10 stop, identical final 10 on a fixed seed).
- `SURVIVOR_TARGET=40` → logs show the pool growing toward 40, round-end summary prints
  reached/target + stop-reason, **final list is exactly 10**.
- Band exhaustion → graceful Borda over the partial pool (no exception), WARNING emitted.

### Validation commands
```bash
# default-identical smoke
SELECTION_MODE=veto_batch python -m multi_agent.experiments.run_experiment    # observe ">=10 survivors" stop
# enlarged-pool smoke
SELECTION_MODE=veto_batch SURVIVOR_TARGET=40 BATCH_SIZE=80 MAX_BATCHES=10 \
  EXPERIMENT_MODE=ofat EXPERIMENT_REPEATS=1 python -m multi_agent.experiments.run_experiment
# confirm final list size is 10
sqlite3 multi_agent/experiments/results.db "SELECT turn_id,COUNT(*) FROM turn_items GROUP BY 1 ORDER BY turn_id DESC LIMIT 5;"
```

---

## 5. Workstream C — Keep RL-sim (curve) and agents-sim (grid) SEPARATE  (docs only)

**No code.** Add to the experiment README/report:

- **Grid (`ofat`/`full`)** measures *agent/personality impact* with the system held fixed; the
  1–5 review is the eval metric, **not** fed to RL.
- **Curve (`curve`)** measures *RL learning* over sequential baseline episodes with factors fixed,
  feeding each review into PPO (`feed_review=True`, `defer_consumption=True`).
- **Do not add a freeze flag and do not merge curve into grid** — feeding the review during a grid
  lets the policy chase satisfaction while personalities also vary (confounds agent effect with RL
  adaptation); varying personalities during a curve confounds learning-over-time with factor
  change. Two separate experiments, each holding the other's variable fixed, is correct.
- **Known caveat to document:** even with `feed_review=False`, a grid run still drifts RL — the
  orchestrator notifies the RL agent each round → `settle_round` applies `passrate_reward` →
  auto-consume every `PPO_ROLLOUT_ROUNDS=8` → `rl_policy.learn()` → **overwrites `rl_ppo.pt`**.
  **[review]** Operational mitigation (env only, no flag): point `RL_CHECKPOINT_PATH` at a throwaway
  file and/or set `RL_FRESH_START=1` so grid runs don't disturb the production/curve checkpoint.

---

## 6. Workstream D — Suggestions (optional)

1. **Interleave personas (de-confound time-order) — LOW effort, HIGH value.** The loop
   (`run_experiment.py:370-395`) is persona-major, confounding any early-vs-late delta with persona
   identity. Add `EXPERIMENT_ORDER=interleave` that round-robins personas. **[review]** Requires
   moving combo apply/restore (`_apply_combo`/`_restore_strategies`) from per-combo to
   **per-episode**, keeping the `try/finally` restore at the new granularity so a mid-episode
   exception can't leak a combo. Touches `run_experiment.py` only — coordinate with the curve owner.
2. **Document the metric reset — TRIVIAL.** Commits `af3b41d`, `bf26765`, and the Workstream-A
   repair each reset the review metric; add a "metric epoch" marker so post-change runs aren't
   compared to #5–#12.

---

## 7. Cross-cutting

- **Independence:** A ⊂ `LNIAGIA/DB/`; B ⊂ `multi_agent/config.py` + `orchestrator.py`. No overlap.
  D1 (interleave) touches `run_experiment.py` only.
- **Sequencing:** **validate Workstream A first** (coherence + review lift) before any large grid
  or curve run — otherwise B/D results sit on the degenerate metric.
- **Hygiene:** branch `rl-learning-curve` only; author `tiagosgdev`, no Claude co-author trailer;
  no `pip install -r`; XMPP + Ollama up.

## 8. Key files (map)
| Area | File |
|---|---|
| Catalog gen + weight tables | `LNIAGIA/DB/models.py` (weight tables ~L819-863; `generate_age_groups` ~L937; chain helpers) |
| Item gen chain | `LNIAGIA/DB/SQLLite/DataGenerator.py` (~L53-161) |
| Derived artifacts | `LNIAGIA/DB/SQLLite/update_images_and_description.py`; `LNIAGIA/DB/vector/` (Qdrant) |
| Catalog load (experiment) | `stock_agent/stock_stats.py`; `stock_agent/stock_agent.py:188` (OR-band) |
| Selection loop | `multi_agent/agents/orchestrator.py` (stop L414; Borda L428) |
| Config knobs | `multi_agent/config.py` (L37-59 batch/TOP_K; L104 RL_REWARD_MODE) |
| RL | `multi_agent/rl/store.py` (`rating_reward` L84; `settle_round` L149); `multi_agent/rl/policy.py` (advantage norm L210-211) |
| Harness | `multi_agent/experiments/run_experiment.py` (feed_review L317-327; persona-major loop L370-395) |
| Personas / shopper | `multi_agent/experiments/customers.json`; `multi_agent/experiments/shopper.py` (`final_review`) |
