# Experiment #8 — `veto_batch` full factorial grid — ANALYSIS

_Run 2026-06-25 18:03 → 21:51 (3h47m, avg 18.7s/episode). 729 episodes
(81 combos × 3 personas × 3 repeats). Shopper `qwen2.5:14b-instruct`, agents
`qwen2.5:7b-instruct-q3_K_M`, both on GPU. **0 weight-timeouts, 0 errors,
0 abandoned.** git `7870a40`._

This is the **veto_batch arm** of the borda-vs-veto_batch A/B. The borda arm is
experiment #9 (running); the head-to-head comparison lives in
[`../comparison_veto_vs_borda.md`](../comparison_veto_vs_borda.md) once #9 lands.
Baseline for context is **exp #7** (the previous full grid, which used the legacy
borda + hard color/type retrieval filter).

---

## TL;DR

The redesign did **exactly what it was built to do**, and exposed the cost of
doing it:

1. **The "only ~16 items ever surface" ceiling is gone.** Distinct items
   surfaced across the whole run went **1,112 → 8,978 (8×)**. In the controlled
   cell (party_maya, repeat0, turn0) distinct items went **16 → 724** and mean
   pairwise Jaccard across the 81 combos collapsed **0.70 → 0.002**.
2. **Personalities now move the outcome.** Colour strategy was pure noise in
   exp #7 (1.81/1.79/1.76); here it's a clean monotonic gradient
   **purist 1.35 > harmonizer 1.29 > adventurous 1.19 (Δ −0.16)**.
3. **The veto is load-bearing** — it eliminates ~47% of every batch
   (~18.8 of 40), and 942 rounds needed a 2nd+ batch. It is *not* a no-op.
4. **But review fell: 1.79 → 1.28.** Broad random retrieval surfaces off-theme
   items that the personas penalise. This is the **variety↔relevance trade-off**,
   and the colour gradient is the smoking gun (the *more on-theme* the colour
   personality, the *better* the review).

**Verdict:** veto_batch fixes the diversity problem outright but pays for it in
relevance. Review is the wrong yardstick for "did the redesign work" (it did);
the lever to recover relevance is **τ** (and/or a soft relevance prior in
retrieval), not the veto rule itself.

---

## 1. Review scores

| metric | exp #7 (borda+hard filter) | exp #8 (veto_batch) |
|--------|:--:|:--:|
| overall mean review | **1.79** (σ=0.61) | **1.28** (σ=0.46) |
| range | 1–4 (six 4s) | 1–3 (zero 4s, four 3s) |
| distribution | mostly 2s | 532× **1**, 193× 2, 4× 3 |

Review **dropped ~0.5**. The distribution shifted hard toward 1: two-thirds of
episodes scored the floor. Note this was **predicted** — party_maya and
casual_sofia have goals that penalise off-colour/off-theme items, and broad
retrieval surfaces exactly those.

### Per persona
| persona | mean | n | range |
|---------|:--:|:--:|:--:|
| office_daniel | **1.51** | 243 | 1–3 |
| party_maya | 1.19 | 243 | 1–3 |
| casual_sofia | 1.13 | 243 | 1–2 |

Persona spread **widened to 0.38** (exp #7 was 0.25). office_daniel (good mood,
generous) tolerates the broader slate; casual_sofia (capped at 2) punishes it.

### Per agent (marginal, n=243/cell — the reliable view)
| agent | best → worst | spread | read |
|-------|--------------|:--:|------|
| **colour** | purist 1.35 → harmonizer 1.29 → adventurous 1.19 | **0.16** | **REAL & monotonic** — stricter is better |
| body | lenient 1.30 → strict 1.28 → flattering_only 1.24 | 0.06 | borderline noise |
| clothing | strict_type 1.28 → weighted_axes 1.28 → match_count 1.27 | 0.01 | noise |
| stock | bestsellers 1.29 → overstock 1.28 → push 1.26 | 0.03 | noise |

**This is the headline change from exp #7.** There, *every* agent was noise.
Here **colour clearly separates** (0.16, ~4σ on SE≈0.03–0.04) and it does so in
the intuitive direction: a stricter colour personality keeps the slate on-theme
and scores better. Body/clothing/stock are still ≈noise — consistent with these
personas steering primarily on colour.

---

## 2. Diversity (the metric the redesign actually targeted)

Computed identically for both experiments from `turn_items` (see
`scratch_diversity.py`). Controlled cell = customer fixed, repeat0, turn0, so
the only thing varying across the 81 rows is the agent-personality combo.

| measure | exp #7 | exp #8 | change |
|---------|:--:|:--:|:--:|
| TOTAL distinct items surfaced (all turns) | 1,112 | **8,978** | **8.1×** |
| party_maya — distinct top-10 **sets** / 81 | 30 | **81** | all unique |
| party_maya — distinct items | 16 | **724** | 45× |
| party_maya — mean pairwise **Jaccard** | 0.701 | **0.002** | ~0 overlap |
| office_daniel — distinct items / Jaccard | 16 / 0.63 | 724 / 0.002 | — |
| casual_sofia — distinct items / Jaccard | 16 / 0.65 | 734 / 0.001 | — |

The monochrome 16-red-dresses pool is **gone**. The candidate space the agents
work over is now the broad catalogue, not a hard-filtered sliver.

> ⚠️ **Honest caveat on interpreting this.** Under veto_batch each round draws
> *random* batches, so two combos with identical input get different items partly
> by luck of the draw. The 0.002 Jaccard is therefore **dominated by draw
> randomness, not by personality** — you can no longer read the controlled-cell
> diversity as "personalities pick different items." That conclusion now comes
> from the **review marginals** instead (the colour gradient in §1), which *is* a
> genuine personality effect. The diversity number proves the *pool* is broad;
> the review marginal proves *personality matters*. Two different claims, each
> with its own evidence.

---

## 3. Is the veto doing real work? (load-bearing check)

From the exp #8 log, aggregated over **5,484 batches** in **4,342 rounds**:

| stat | value |
|------|:--:|
| mean items vetoed-out per batch | **18.81 / 40 (~47%)** |
| mean items surviving per batch | 21.19 / 40 |
| rounds that needed a 2nd+ batch | **942** (~22%) |
| rounds that hit best-effort backfill | **4** (~0.1%) |
| mean batches per round | 1.26 |

**The veto is decisively load-bearing** — this refutes the earlier worry that
"never dropping below 10" might mean the veto does nothing. It eliminates ~half
of every batch via the weighted-coalition rule (τ=0.5), and in ~1 round in 5 the
post-veto pool fell short of 10 and a second batch was drawn. Backfill almost
never fires, so the survivor pool genuinely supplies the top-10. The "≥10
survive in one batch" pattern just means *enough* survive, not that *nothing* is
cut.

---

## 4. What this means / next levers

- **Did the redesign succeed?** Yes, on both stated goals: broke the retrieval
  ceiling (8× more items) and made personalities matter (colour gradient). Both
  were the explicit failures of exp #7.
- **Why did review fall?** Pure variety↔relevance trade-off. τ=0.5 + broad band
  is tuned hard toward variety; off-theme items reach the slate and the personas
  dock them. The colour gradient (purist best) confirms relevance is what's lost.
- **Lever to recover relevance (follow-up, not done):** lower **τ toward
  0.2–0.3** so a smaller / single strong coalition can eliminate off-theme items
  — trades some variety back for on-theme relevance. (Do **not** switch to
  unconditional single-agent veto / blackball: with clothing vetoing ~38/40, one
  strict agent would dictate the slate and collapse diversity back toward the
  exp #7 failure.)
- **Confound to keep in mind for the A/B:** the borda arm (#9) uses the *narrow*
  hard-filtered retrieval, veto_batch uses *broad random*. So #8-vs-#9 tests
  "narrow+borda" vs "broad+veto+borda" — the veto's own marginal contribution is
  entangled with the retrieval change. The comparison doc will flag this.

See [`README.md`](README.md) (auto-report), [`by_agent.md`](by_agent.md),
[`by_persona.md`](by_persona.md), [`combos.md`](combos.md).
