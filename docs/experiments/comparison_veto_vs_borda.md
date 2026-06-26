# A/B: `veto_batch` vs `borda` — head-to-head

_Comparison of the two selection modes, both run on current HEAD (`7870a40`) with
the subjective shopper, full factorial grid (81 combos × 3 personas × 3 repeats =
729 episodes each), `qwen2.5:14b-instruct` shopper. Both runs: 0 weight-timeouts,
0 errors, 0 abandoned._

| arm | experiment | selection | retrieval |
|-----|:--:|-----------|-----------|
| **veto_batch** | [#8](exp_8/ANALYSIS.md) | random-batch + weighted-veto (τ=0.5) + tie-aware Borda | **broad random** band (match ≥1 signal) |
| **borda** | [#9](exp_9/ANALYSIS.md) | legacy single-round weighted Borda | **narrow** hard color/type filter |

> **Use exp #9 as the borda baseline, not exp #7.** exp #7 was borda on the *old*
> objective shopper (review 1.79). exp #9 is borda on the *same* subjective shopper
> as exp #8 (review 1.57); the 0.22 gap is the shopper change alone. Only #8 vs #9
> is apples-to-apples.

---

## The result in one table

| metric | #9 borda | #8 veto_batch | winner |
|--------|:--:|:--:|:--:|
| **overall mean review** | **1.57** (σ=0.53) | 1.28 (σ=0.46) | **borda +0.29** |
| review range | 1–4 | 1–3 | borda |
| **TOTAL distinct items surfaced** | 1,258 | **8,978** | **veto 7.1×** |
| party_maya cell — distinct items / 81 combos | 22 | **724** | veto 33× |
| party_maya cell — mean pairwise Jaccard | 0.46 | **0.002** | veto |
| **do personalities move review?** | **no** (colour spread 0.06) | **yes** (colour gradient 0.16) | **veto** |
| per-persona spread | 0.55 (1.26–1.81) | 0.38 (1.13–1.51) | — |

### Per persona
| persona | #9 borda | #8 veto_batch |
|---------|:--:|:--:|
| office_daniel | 1.81 | 1.51 |
| casual_sofia | 1.65 | 1.13 |
| party_maya | 1.26 | 1.19 |

borda wins on every persona for review; the gap is largest for casual_sofia
(1.65 vs 1.13), who punishes the off-theme items that broad retrieval surfaces.

---

## What each arm buys you

**borda (#9) — "safe, on-theme, low-variety."**
- Higher satisfaction (1.57): the hard color/type filter keeps every recommendation
  on-theme, so the shopper rarely sees something it dislikes.
- **But personalities are noise** (colour 1.54–1.60, all agents ≤0.06 spread) and
  the surfaced pool is narrow (~1.3k items). This is *exactly the exp #7 failure*:
  agents only re-rank a hard-filtered sliver, so swapping a personality barely
  changes the output.

**veto_batch (#8) — "diverse, personality-driven, lower-relevance."**
- **7× more items surfaced** (8,978 vs 1,258); per-combo top-10s are essentially
  unique (Jaccard 0.002). The monochrome-16 ceiling is gone.
- **Personalities genuinely matter**: colour shows a clean monotonic gradient
  (purist 1.35 > harmonizer 1.29 > adventurous 1.19) — the *more on-theme* the
  colour personality, the better the review.
- The veto is **load-bearing**, not cosmetic: it eliminates ~47% of every 40-item
  batch (τ=0.5 weighted coalition), and 942/4342 rounds needed a 2nd batch; only
  4 rounds ever hit backfill.
- **Cost: −0.29 review.** Broad retrieval surfaces off-colour/off-type items the
  personas dock. The colour gradient is the smoking gun for *why*: relevance is
  what's being traded away.

---

## ⚠️ Confound — read before concluding "veto < borda"

The two arms differ in **both** dimensions at once:

| | selection | retrieval |
|--|--|--|
| borda | Borda | **narrow** (hard filter) |
| veto_batch | veto+Borda | **broad** (random band) |

So #8-vs-#9 measures **(veto + broad)** against **(Borda + narrow)** as bundles.
You **cannot** attribute the review drop to "the veto is bad" — most of it is the
*broad retrieval* letting off-theme items in. The veto, in isolation, is doing
sensible elimination work (~47%/batch). To isolate the veto's own contribution
you'd need a 2×2 (selection × retrieval-band) — not run here.

A second caveat: the controlled-cell diversity (Jaccard 0.002) under veto_batch is
dominated by *random batch draw*, not personality. The proof that "personalities
matter" comes from the **review marginals** (the colour gradient), not the
diversity number. The diversity number proves the *pool* is broad; the review
marginal proves *personality changes the outcome*. Two claims, two pieces of
evidence.

---

## Recommendation

**This is a genuine product trade-off, not a clear win.** It depends on the goal:

- **If the objective is raw shopper satisfaction on a known-theme query** → borda
  is better today (1.57 vs 1.28), but accept that personalities are decorative and
  variety is capped.
- **If the objective is the one that motivated the redesign** — make agent
  personalities actually matter and break the "16 red dresses" pool (the explicit
  exp #7 failure) — **veto_batch is the only arm that delivers it**, decisively.

**Suggested next step (the sweet spot):** keep veto_batch's broad retrieval but
**lower τ from 0.5 toward 0.2–0.3** so a smaller/stronger coalition trims more
off-theme items — recovering relevance while retaining far more variety than
borda. τ is the relevance↔variety dial the redesign already built for exactly
this. A τ-sweep (0.2 / 0.3 / 0.5) is the natural follow-up A/B.

**Do _not_** switch the veto to unconditional single-agent removal ("blackball"):
with clothing vetoing ~38/40 items, one strict agent would dictate the entire
slate and collapse diversity right back to the exp #7 failure.

---

### Sources
- [`exp_8/ANALYSIS.md`](exp_8/ANALYSIS.md) — veto_batch full write-up.
- [`exp_9/ANALYSIS.md`](exp_9/ANALYSIS.md) — borda full write-up.
- Auto-reports: `exp_8/`, `exp_9/` (README / combos / by_agent / by_persona).
- Diversity computed by `scratch_diversity.py` (controlled cell = customer fixed,
  repeat0, turn0, across all 81 combos).
