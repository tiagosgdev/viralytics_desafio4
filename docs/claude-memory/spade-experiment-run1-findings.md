---
name: spade-experiment-run1-findings
description: Results + diagnosis of the FIRST live Part E OFAT experiment run (2026-06-24). Recommender bugs found; shopper-LLM verdict.
metadata: 
  node_type: memory
  type: project
  originSessionId: 6e3f69f2-a5aa-46d7-8a01-54bdd89b1fd1
---

First live end-to-end run of the Part E harness (`multi_agent/experiments/`), 2026-06-24, experiment_id=1 in `multi_agent/experiments/results.db`. 27 episodes (9 OFAT combos × 3 customers × 1 repeat), ~70 min on the M1 Pro 16GB, qwen2.5:7b-instruct-q3_K_M for both FeatureWeightAgent and the LLM shopper.

## Harness is VALID
- 27/27 completed, 0 abandoned, no NULL/fallback ratings (no LLM/parse crashes).
- Weight knob works (baseline vs body=lenient differ: stock 0.09→0.24). NOTE: personality strategies change how an agent SCORES, not the top-level weights — so some combos show identical turn-1 weights; that's expected, not a bug.
- Shopper-as-JUDGE is trustworthy: every review reason correctly names the real mismatch.

## Results dominated by RECOMMENDER bugs, not personalities (fix in this order)
1. **Size duplication kills diversity (biggest).** Top-N treats each `(item_id,size)` as a separate result → party_maya final top-10 = only 4 unique garments (item 63 as S/XS/M/L, item 38 as S/XL/XS/XXL). Shopper rightly complains "not diverse" nearly every episode. FIX: dedup top-N by item_id (collapse sizes). Cheapest + biggest expected win.
2. **Final-turn drift.** Reviews score ONLY the final turn. party_maya was red 469 / white 33 across all turns (color IS honored), but baseline's LAST turn surfaced white → rated 1. Rankings degrade off-goal by turn 5-6; shopper never satisfied → 22/27 episodes hit the 6-turn MAX_TURNS cap. Investigate why late turns drift (accumulated-intent wiring or RL).
3. **Soft attrs ignored.** Matches color/type/fit but not style/occasion/price: casual_sofia wanted cheap everyday casual, got $605 preppy date-night trousers (right color/type/fit, all else wrong).

## Better LLM for the shopper? — mixed, and order matters
- As a JUDGE: no, reviews are calibrated/discriminating in content.
- As a STEERER: yes, real weakness — turn messages HALLUCINATE ("I love the checkered red dress!" when items are white). 7b-q3 doesn't reliably perceive item attrs, so it praises non-existent items and steers vaguely.
- BUT a better shopper WON'T move the scores — low ratings are genuine recommender misses. Fix recommender first; upgrade shopper for realism only.

## FIXES APPLIED (2026-06-24, after run 1 — uncommitted on spade-dynamic-weights-experiments)
- **#1 diversity (recommender).** Root cause was deeper than aggregation: the 40-candidate
  POOL itself was only ~7 distinct items (each ×~6 sizes), because stock pool is at
  (item_id,size) grain. Two-layer fix: (a) `multi_agent/retrieval.py` now overfetches
  `n*_SIZE_OVERFETCH` (8) rows and collapses to n DISTINCT items, keeping the best-stocked
  size per item, preserving match-count order; (b) `multi_agent/aggregator.py` `borda_aggregate`
  dedups the top-k by item_id (best-scoring size) as a safety net. Verified live: a round now
  returns 10 distinct items (was 3-4), all colour/type-correct.
- **#2 drift (HARNESS bug, not recommender).** Turns 0-4 were all on-goal (red); the final
  turn collapsed to white because the harness sent only the LATEST shopper message as
  user_answer, dropping the colour anchor when a later turn only said "defined waist". The real
  frontend ACCUMULATES (commit cb6bc95, `accumulatedUserIntent()` = last 6 user turns joined
  '. '). Fixed `multi_agent/experiments/run_experiment.py` to accumulate likewise (shopper still
  sees its individual lines, clean prompt). Verified live: colour/type anchor now survives refining turns.
- **#3 price/budget — FIXED (2026-06-24).** Budget now parsed from chat and steers retrieval.
  `feature_weighting._extract_price_range()` = deterministic parser (numeric phrases + vague
  cheap<$50<medium<$150<expensive via PRICE_CHEAP_MAX/PRICE_EXPENSIVE_MIN; handles negated floors
  "nothing over $40" = ceiling). analyze_intent emits price_min/max into `filters`; retrieval
  already forwards the whole filters dict. `stock_agent.get_candidates` makes price a SOFT feature
  (in-budget = +1 match_count, like an include key) NOT a hard filter — so the pool always
  backfills to n (tiered relaxation), never dead-ends on a tight budget. Per user's spec:
  "always need 40, follow same rule as other features." c) no price weight/scoring emphasis —
  price only gates retrieval. Verified live: "nothing over $40"→$17-38; "premium over $200"→
  $208-1153. Budget fully honoured when >=n in-budget items exist, else best-effort.

## Stats caveat
Per-combo means 1.33–2.33 over n=3 (3 customers × repeats=1), scores compressed to 1/2/3 (no 4-5). The "winner" (clothing=weighted_axes 2.33) is noise. Can't discriminate personalities until bugs fixed AND repeats raised (→3) and/or a stronger shopper model.

See [[spade-experiments-state]], [[recommender-architecture]].
