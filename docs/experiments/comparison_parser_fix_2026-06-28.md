# Parser fix (style/occasion → include) — borda vs veto vs RL curve

_Overnight run 2026-06-28. Three sims on `rl-learning-curve` HEAD + one uncommitted
change: `analyze_intent` now extracts **style + occasion** into `include` (was
color/type/body_type only). Full factorial grid (81 combos × 3 personas × 3 repeats
= 729 episodes) for borda and veto; 300-episode learning curve for RL.
`qwen2.5:14b-instruct` shopper. All runs: 0 errors, 0 abandoned, RL isolated to
scratch checkpoints (prod `rl_ppo.pt` untouched)._

> **The change in one sentence.** Every persona goal hinges on style + occasion, but
> the intent parser dropped those two axes, so the recommender could not tell an
> in-band item from an off-band one — capping reviews near 2.0. Wiring the two axes
> into `include` (a prompt-only change; the whole downstream was already built for
> it) is the first thing to move the metric off that ceiling.

> **METRIC EPOCH RESET.** The parser change resets the primary metric — exps #20–22
> are **not** comparable to historical exps #5–#17. The only valid baseline is
> exp #15 (same personas + rubric, *without* the fix).

---

## The result in one table

| metric | exp #15 (no fix) | #20 borda + fix | #21 veto + fix |
|--------|:--:|:--:|:--:|
| **overall mean review** | 1.99 | **2.56** | 2.11 |
| Δ vs #15 | — | **+0.57** | +0.12 |
| review dist (1/2/3/4/5) | —/—/118/5/0 | 10/364/**295**/**60**/0 | 75/505/142/7/0 |
| 3-rate | 16% | **40%** | 19% |
| 4-rate | 0.7% | **8.2%** | 1.0% |
| 5-rate | 0% | 0% | 0% |

### Per persona
| persona | #15 | #20 borda | #21 veto | best Δ |
|---------|:--:|:--:|:--:|:--:|
| office_daniel | 2.22 | **2.91** | 2.41 | **+0.69** |
| party_maya | 1.85 | **2.48** | 1.96 | **+0.63** |
| casual_sofia | 1.90 | **2.28** | 1.96 | **+0.38** |

---

## Finding 1 — the fix works, and works through scoring

Every persona improved, and the **mass moved up the scale**: under borda, 3s went
16%→40% and 4s went ~0.7%→8.2% (60 fours vs **5 in all of exp #15**). Mechanism,
verified directly before launch: the clothing agent's `score_match_count = hits /
n_axes` now divides over up to 5 axes, so a full in-band item scores **1.0**, a
color+type-only item **0.5**, an off-band item **0.0** — where before the first two
both scored ~0.67 and were indistinguishable. In-band items now out-rank off-band
ones, so the shopper sees more of what it asked for.

## Finding 2 — borda gains 5× what veto gains (narrow vs broad retrieval)

The fix lifts **borda +0.57** but **veto only +0.12**. Veto's broad random OR-band
(`get_random_batch`, ~1,300-row band) floods the pool with off-theme items, and the
Borda-across-batches selection dilutes the sharper scoring. Borda's narrow
match_count retrieval keeps the pool on-theme, so the re-ranking bites. The effect is
**goal-dependent**: daniel (tight goal smart/minimalist + work, 70 in-band catalog
items) lifts in veto too (+0.19), while broad-goal maya is flat in veto (+0.11).

> **Takeaway for the product:** borda + the parser fix is the strongest combination
> for satisfaction. veto still buys diversity (its original trade-off) but the fix
> can't rescue its relevance the way it does borda's.

## Finding 3 — the ceiling is still 4 (no 5 has ever occurred)

0 fives in either grid, and a 5 has **never** occurred in ~3,000+ episodes (max
ever = 4). The fix made style+occasion matchable, but `pattern`/`material`/`fit`
remain uncorrelated per item, so a set is rarely "perfect" enough for a 5. **Lever
to chase the 5:** apply the same move to `pattern`/`material` — either add them to
`include` (parser; already valid `ALL_MAPPINGS` keys) or condition them on
style/occasion in the catalog repair.

---

## Finding 4 — RL still does NOT learn (the fix did not unlock it)

**Curve #22: 300 episodes, 224 PPO updates, 3,000 rewards landed, 0 dropped** (the
plumbing is sound — the transition-merge fix holds, so the curve is trustworthy).

| signal | value | read |
|--------|:--:|------|
| naive Δ(late−early) review | **−0.067** | persona-confounded (early=maya, late=sofia); ignore |
| within-persona Δ (1st→2nd half) | maya +0.18, daniel +0.08, sofia +0.04 | weakly positive… |
| **mean_return (PPO advantage)** | **−0.0169 → −0.0164, flat** across all 224 updates | **…but NO gradient — same as exp #12** |

⚠️ The curve runs **persona-major** (maya 1–100, daniel 101–200, sofia 201–300), so
the naive early-vs-late delta is a persona artifact, not learning — it must be read
**within persona**. The within-persona rating slopes are weakly positive, BUT
`mean_return` stays flat-negative (≈ −0.016) the whole run, exactly as in the exp #12
null. With a flat advantage signal the policy isn't really moving, so those small
rating upticks **cannot be credited to PPO** — they're block noise.

**Why still null, despite more review variance overall:** the curve runs in
**veto_batch** — the very mode the fix helped *least*. Curve reviews stayed
floor-bound (mean 2.16, mostly 1s/2s → rewards mostly −1/−0.5), so the critic still
learns "expect ≈ −1," advantages collapse to ≈ 0, no gradient. The reward
distribution never developed the positive mass PPO needs.

### The actionable lever (next run)
Run the curve in **borda** mode, where the fix produced 40% 3s + 8% 4s → real
positive reward mass (rewards spanning −1…+0.5 instead of pinned at the floor). That
is the single change most likely to finally give PPO a learnable signal. Optionally
also: set `EXPERIMENT_ORDER=interleave` so the early-vs-late headline is clean, and
boost RL's Borda weight so its vote actually moves the top-K.

---

## Agent-personality impact (marginal mean review per strategy)

Marginal = mean review across **every** combo where that agent used the strategy
(other 3 agents vary, n=243/cell). "Spread" = best−worst strategy for that agent =
how much that personality axis moves satisfaction.

### Borda #20 — colour and stock personalities now BITE (they were noise pre-fix)
| agent | best → worst | spread |
|-------|--------------|:--:|
| **stock** | push **2.71** › overstock_aggr 2.54 › bestsellers **2.42** | **0.29** |
| **colour** | purist **2.68** › adventurous 2.50 › harmonizer **2.49** | **0.19** |
| body | flattering_only 2.60 › lenient 2.54 › strict 2.52 | 0.08 |
| clothing | weighted_axes 2.59 › match_count 2.54 › strict_type 2.53 | 0.06 |

### Veto #21 — only colour matters; everything else flat
| agent | best → worst | spread |
|-------|--------------|:--:|
| **colour** | purist **2.23** › harmonizer 2.15 › adventurous **1.95** | **0.28** |
| body | strict 2.12 › flattering 2.11 › lenient 2.10 | 0.02 |
| clothing | match_count 2.13 › strict_type 2.10 › weighted_axes 2.10 | 0.03 |
| stock | push 2.12 › bestsellers 2.12 › overstock_aggr 2.09 | 0.03 |

**Reading it:**
- **colour-purist wins in both modes** — every persona has a specific colour goal, so
  the agent that holds the requested colour beats the ones that wander
  (harmonizer/adventurous). The "more on-theme personality ⇒ higher review" gradient
  is clean and directional.
- **stock personality matters in borda (0.29) but not veto (0.03).** In borda's narrow
  on-theme pool, a `bestsellers`/`overstock` stock agent is the main way an off-theme
  item sneaks into the top-10 → it drags the review down. In veto the pool is already
  broad+random, so the stock personality is lost in the noise.
- **body & clothing personalities barely move the metric** in either mode (≤0.08). The
  parser fix routes goal-matching through colour + the include axes; body/clothing
  re-ranking is second-order.
- **vs pre-fix (old comparison):** borda used to be "personalities are noise" (colour
  spread 0.06). After the fix borda colour spread is 0.19 and stock 0.29 — **the fix
  didn't just raise the mean, it made personalities legible in borda.**

### Differentiation — same persona, personality flips the outcome
Best vs worst combos: **borda 3.11 → 1.89** (range 1.22), **veto 2.44 → 1.56** (1.22 / 0.88).
Top combos are colour=purist|stock=push; bottom are colour=adventurous/harmonizer|stock=bestsellers.

Concrete (real `review_reason` text, borda, **same persona, same colour, stock flipped**):
- **office_daniel · purist + push → 4:** _"recommendations mostly align with my
  minimalist and neutral colour preferences, but the fit and pattern choices were not
  ideal"_ — style+colour matched (the fix), only fit/pattern off.
- **office_daniel · purist + bestsellers → 1:** _"None of the final recommendations
  match my preference for solid colours like black or gray with no patterns"_ — the
  bestsellers stock personality flooded the slate with popular-but-off-theme items.
- **party_maya · purist + push → 4:** _"The red … streetwear dress with a slim fit is
  close to perfect but lacks the shine/texture I desired"_ — colour+style+type matched;
  only material missed → **this is exactly why the ceiling is 4 and not 5** (the fix
  matches colour/type/style/occasion; pattern/material/fit are still uncorrelated).

## RL-agent impact

The RL agent is **1 of 5 Borda voters** and is **not** a varied factor in the grids
(only colour/body/clothing/stock vary), so its effect is measured by the dedicated
learning curve, not the grids.

**Curve #22 (300 eps, 224 PPO updates, 0 dropped) — RL has ~no measurable impact / does
not learn:**
| signal | value | meaning |
|--------|:--:|--------|
| naive Δ(late−early) | −0.067 | persona-confounded (early=maya, late=sofia) — discard |
| within-persona Δ (1st→2nd half) | maya +0.18 · daniel +0.08 · sofia +0.04 | weakly +ve… |
| **mean_return** | −0.0169 → −0.0164 (**flat, all 224 updates**) | **no advantage signal ⇒ policy isn't moving** |

The weak positive within-persona rating drift can't be credited to PPO because
`mean_return` never develops — the critic still predicts the (negative) constant
reward, advantages collapse to ≈0, identical to the exp #12 null. RL neither helps nor
hurts the recommendation here; it's a passive 5th vote.

**Why:** the curve runs in `veto_batch`, the mode the fix helped least → reviews stay
floor-bound (2.16) → rewards mostly ≤0 → no learnable variance. **Lever:** run the curve
in `borda` (40% 3s + 8% 4s = positive reward mass), interleave personas, boost RL's
Borda weight.

## Status / reproduce

- Reports: `multi_agent/experiments/reports/exp_20`, `exp_21`, `exp_22` (auto-gen).
- Plot: `multi_agent/experiments/plots/rl_learning_curve.png` (note: its headline Δ
  is the persona-confounded naive one — read within-persona).
- The parser fix is **uncommitted** on `rl-learning-curve` (one file:
  `LNIAGIA/query_parsing/feature_weighting.py`, `_build_intent_system_prompt`).
  Pending review before commit.
- Commands: see `docs/experiments/overnight-run-2026-06-28.md` (full journal).
