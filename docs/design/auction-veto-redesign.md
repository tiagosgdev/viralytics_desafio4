# Auction redesign: random-batch + veto, iterate to fill 10

> Status: PLAN (not yet implemented). Goal: make agent personalities actually
> change the recommendations, using deterministic (non-LLM) agents.

## Context — why

Experiment #7 (full factorial grid, 729 episodes — see `docs/experiments/exp_7/`)
showed agent personalities barely move the review score. Detailed review found the
root cause:

- The slate is hard-filtered to the scan (e.g. `color=red, type=short_sleeve_dress`),
  so it's **uniform on color and type** and varied only on secondary attributes.
- The scorer agents are **single-attribute specialists**: `colour` reads only color,
  `clothing` only the filter axes. On a hard-filtered slate they bid **perfectly flat**
  (measured within-top10 bid stdev: colour 0.0, clothing 0.0 for party_maya; clothing
  ~0.01 across *all* personas/turns). Their personalities are then no-ops.
- Only `body` (0.25) and `stock` (0.18) genuinely differentiate — hence `body=lenient`
  was the only marginal effect that surfaced.
- Worse, `aggregator.py:59-63` turns a flat bid into an **arbitrary item-id-ordered
  ramp** (ties broken by `item_key` string), so the two inert agents inject ~51% of
  the weight as noise that pins the same ~7 "winners" every round.

LLM-style holistic weighted scoring was rejected: the agents are hard-coded, so
nuanced "prioritize X but also weigh Y" judgment would be brittle hand-tuning.

## The new mechanism (hybrid random batch + veto, iterate to 10)

Per João Serra's proposal, adapted to a **broad random sample** (option 3):

1. **Retrieval goes broad + random.** Sample N=40 **random** in-stock items from the
   *broad relevance band* = items matching **at least one** signal — any one scanned
   attribute (color / type / body_type) OR any conversation-requested feature
   (decision #2). This band holds ~1,300+ items (vs 45 for the old `color AND type`), so
   random sampling genuinely varies the slate AND every agent's dimension varies → no
   agent is inert.
2. **Agents score + veto (membership).** Each agent keeps scoring only its own dimension
   (no holistic rewrite), and **vetoes** items below a personality-set `veto_threshold`.
   Personality = veto strictness.
3. **Weighted veto eliminates (decision #1 = B).** For each item, `reject_mass` = Σ of the
   weights of the agents that vetoed it; **eliminate if `reject_mass ≥ τ`**. Surviving
   items go into a running pool (keep each agent's raw score for them). Relevance is left
   entirely to the agents' votes (decision #3 = votes) — no separate soft scan-preference.
4. **Iterate to fill 10.** If the pool has < 10 survivors, draw another random 40
   (excluding already-seen) and repeat, up to `MAX_BATCHES`.
5. **Tie-aware Borda orders the survivors (ranking).** Once the pool has ≥ 10 (or batches
   exhausted), run **tie-aware weighted Borda over the accumulated survivor pool** (using
   each agent's stored raw scores) → ordered top-10. If still < 10 after `MAX_BATCHES`,
   best-effort fill from the least-vetoed / highest-Borda items seen.

> **Veto vs Borda:** the weighted veto decides *which* items survive (membership); Borda is
> NOT removed — it runs *after*, on the survivors, to decide the *order* and final 10. Same
> Borda mechanism as today, just over the post-veto pool instead of the raw 40.

Why this fixes it: a diverse slate makes every agent's bid vary, and the **veto makes
each agent's personality directly change which items survive** → swapping personalities
changes the output, and the top-10 is naturally varied (not 10 near-identical reds).

## Veto: derive from the existing score (minimal, no agent rewrite)

Each scorer already returns `{item_key: score in [0,1]}` (`score_fn` in
`multi_agent/strategies/*`). Add one per-strategy param **`veto_threshold`**: the agent
vetoes any item whose score `< veto_threshold`. This needs **no new scoring logic** —
just a threshold + emitting the veto set. Personality maps cleanly, e.g. colour:
- `purist` → high threshold (veto anything not exact/compatible)
- `harmonizer` → medium (allow the complementary palette)
- `adventurous` → low/zero (veto only clashes)

PROPOSE carries the veto set: extend `make_propose` (`multi_agent/messages.py:82`) and
`parse` to include `vetoes: [item_key, ...]` alongside `scores`.

## The two veto-combination options (DECISION NEEDED)

### Option A — Hard blackball (any agent vetoes → out)
An item is eliminated if **any single agent** vetoes it. Survivors satisfy *every*
agent's minimum bar.
- **Pros:** dead simple; every survivor is acceptable to all agents; strong, legible
  personality effect (a strict agent visibly thins the field).
- **Cons:** one picky agent can eliminate almost everything → many iterations / slow;
  ignores agent weight (a 0.13-weight agent vetoes as hard as a 0.26 one); brittle to a
  single threshold.

### Option B — Weighted collective veto (eliminate if reject-mass ≥ τ)
Each vetoing agent adds **its weight** to a `reject_mass`; eliminate the item if
`reject_mass ≥ τ` (e.g. τ=0.5).
- **Pros:** respects agent weights (important agents' vetoes count more); robust to one
  picky agent; τ is a single tunable knob; degrades gracefully.
- **Cons:** more moving parts; τ needs tuning; a high-weight agent can still dominate
  (often desirable).

> **Decision: B (weighted collective).** Eliminate an item when the combined weight of the
> agents vetoing it reaches τ. A (blackball) is the τ→0+ special case, so a `VETO_MODE` flag
> can still expose it for A/B comparison, but B is the chosen default.

## Files to change

- **`multi_agent/retrieval.py`** (`get_candidates`, line 63) + **`stock_agent/stock_agent.py`**
  (`get_candidates`, line 118): add a broad-random sampling path (random N within a broad
  band; e.g. `ORDER BY RANDOM()` over a relaxed filter, or `random.sample` over the broad
  set). Gate behind a `SELECTION_MODE` config flag so the legacy exact-filter path stays
  for comparison. **NB — today the query returns items `match_count desc, item_id asc`, so
  the same scan yields the *identical* 40 every round.** Random sampling fixes this *only if
  the broad band holds ≫ 40 items* — sampling 40 of ~45 still returns ~the same set
  (decision #2 governs this). This is a *separate* id-ordering issue from the aggregation
  one below; both must be fixed (see "Two id-ordering bugs").
- **`multi_agent/strategies/*.py`** (colour/body/clothing/stock): add a `veto_threshold`
  param to each strategy's params (no change to the score functions themselves).
- **`multi_agent/messages.py`** (`make_propose` line 82, `parse` line 130): carry `vetoes`.
- **`multi_agent/agents/*_agent.py`** (4 scorers, e.g. `colour_agent.py:53-64`): after
  scoring, compute `vetoes = [k for k,v in scores.items() if v < veto_threshold]` and pass
  to `make_propose`.
- **`multi_agent/agents/orchestrator.py`** (round loop, lines 126-185): replace the
  single retrieve→CFP→Borda with the **batch loop**: retrieve random 40 → CFP → collect
  scores+vetoes → eliminate (Option A/B) → accumulate survivors → repeat until 10 or
  `MAX_BATCHES`, then best-effort fill from highest-weighted non-eliminated items.
- **`multi_agent/aggregator.py`**: add a `select_with_veto(proposals, vetoes, weights, k)`
  helper (reuse the weighted ranking; apply elimination first). **Included in this work:**
  the **tie-aware Borda** fix — see the appendix below.

### Two id-ordering bugs (both fixed here, different layers)
1. **Retrieval order** — query returns `id`-sorted, deterministic → same 40 every round.
   Fix: **random sampling** in retrieval (effective only with a broad band, decision #2).
2. **Aggregation tie-break** — Borda breaks score-ties by `item_key`, so even a *different*
   random batch is internally ranked by item-id among ties. Fix: **tie-aware Borda**.
   Random sampling does NOT fix this layer; the two fixes are complementary and both ship.
- **`multi_agent/config.py`**: add `SELECTION_MODE` (`borda` | `veto_batch`), `VETO_MODE`
  (`blackball` | `weighted`), `VETO_TAU`, `MAX_BATCHES`, `BATCH_SIZE`.

## Verification

- **Unit:** extend `tests/test_strategies.py` (veto threshold → veto set), add
  `tests/test_aggregator` cases for `select_with_veto` under both modes, and an
  orchestrator batch-loop test (mock agents, assert it fills 10 and terminates at
  `MAX_BATCHES`).
- **Diversity metric:** re-run the controlled check (party_maya, turn 0): bid stdev per
  agent should be > 0 for all agents on a random slate, and the distinct-items / Jaccard
  across personality combos should rise sharply from the exp-7 baseline (30 sets / 0.70).
- **End-to-end:** `EXPERIMENT_MODE=full EXPERIMENT_REPEATS=3` re-run on GPU, then
  `python -m multi_agent.experiments.report`. Success = per-agent marginal spread now
  exceeds noise (Δ ≫ 0.04 SE), i.e. personalities separate.

## Resolved decisions
1. **Veto combination → B (weighted collective).** Eliminate if Σ(weights of vetoing
   agents) ≥ τ. A (blackball) kept reachable as the τ→0+ special case via `VETO_MODE`.
2. **Broad band width → match ≥ 1 signal.** Band = items matching at least one of the
   scanned attributes (color/type/body_type) OR any conversation-requested feature.
   ~1,300+ items, so random sampling actually varies the slate.
3. **Relevance floor → votes only.** No separate soft scan-preference in ranking;
   relevance is entirely the agents' scores + vetoes.

## Committed scope
- **Tie-aware Borda** fix (appendix below) — ranks the post-veto survivors.
- **Random sampling** in retrieval replaces the deterministic id-ordered query.

---

## Appendix — the tie-aware Borda fix (in detail)

### What Borda count is
Each agent is a "voter" that ranks the N candidate items. Borda gives the top-ranked
item N points, the next N−1, … down to 1 for last. Each agent's points are scaled by its
weight and summed across agents; highest total wins. Today's code
(`multi_agent/aggregator.py:59-63`):

```python
ranked = sorted(all_items, key=lambda ik: (scores.get(ik, 0.0), ik), reverse=True)
for rank_0, item_key in enumerate(ranked):
    borda_pts = n - rank_0                 # N for rank 1, … 1 for rank N
    composite[item_key] += weight * borda_pts
```

### The bug: ties become an arbitrary strict order
Borda assumes a strict ranking, but agents routinely produce **ties** — equal scores for
many items. Example: the `colour` agent on an all-red slate scores all 40 items `1.0`.
They are *tied for first*. But the sort key is `(score, item_key)`, so the tie is broken
by the **item_key string**, and the loop then hands out a strict ramp `40, 39, 38, …` as
if the agent had a real preference order.

That order is arbitrary — it's just `item_id` text sorting (`"8626" > "777" > "5994"`
because `'8' > '7' > '5'`). So an agent that is genuinely **indifferent** is forced to
"prefer" high-item-id garments, and its weight (~0.255 for colour) multiplies that
arbitrary ramp into the result. Because `colour` and `clothing` are both flat and both
sort by the same `item_key`, they emit the **same** arbitrary ramp — together ~51% of the
weight — which pins the same ~7 "winners" every round regardless of personality. This is
a measured cause of the exp-7 non-differentiation, not a hypothetical.

### The fix: average (fractional) Borda points for tied items
Standard handling of ties in rank scoring: every item in a tied group receives the
**average** of the Borda points the group spans.

- **Fully flat agent** (all 40 tied at the same score): the group spans points 40…1,
  average `(40+1)/2 = 20.5`. Every item gets **20.5** → a constant added to all items →
  it does **not** change the relative ranking → the indifferent agent contributes
  **zero differentiation** (truly neutral), instead of arbitrary noise.
- **Partially tied agent** (e.g. 30 items at 1.0, 10 at 0.2): the 30 share the top ranks
  → each gets avg of points 40…11 = **25.5**; the 10 share the bottom → each gets avg of
  10…1 = **5.5**. The two groups are still separated (real signal preserved), but within
  each group every item is equal (no fake order). Only genuine preference moves points.

Net effect: flat/indifferent bids stop injecting item-id noise, so the agents that
actually have a preference decide the ranking — and personality differences surface.

### Why it still matters under the new veto design
Even with a broad random slate, an agent can still tie many items (e.g. `colour` scoring
every red item in a batch `1.0`). Tie-aware Borda keeps those ties neutral so they don't
re-introduce the same arbitrary-order artifact when ranking the survivors. It's small
(one function change) and complementary to the veto mechanism.
