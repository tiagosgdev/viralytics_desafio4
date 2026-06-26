# Experiment #9 — `borda` full factorial grid — ANALYSIS

_Run 2026-06-25 21:51 → 2026-06-26 01:40 (3h48m, avg 18.8s/episode). 729 episodes
(81 combos × 3 personas × 3 repeats). Shopper `qwen2.5:14b-instruct`, agents
`qwen2.5:7b-instruct-q3_K_M`, both on GPU. **0 weight-timeouts, 0 errors,
0 abandoned.** git `7870a40`._

This is the **borda arm** (legacy single-round weighted Borda + hard color/type
retrieval filter) of the borda-vs-veto_batch A/B. The veto_batch arm is
experiment #8. **Head-to-head:**
[`../comparison_veto_vs_borda.md`](../comparison_veto_vs_borda.md).

---

## Why this is the *real* borda baseline (not exp #7)

exp #7 was also borda+hard-filter, but on the **old objective shopper**. exp #9 is
borda on the **current HEAD with the subjective shopper** (`7870a40`: fixed
per-persona mood + idiosyncratic tastes). They differ only in the shopper:

| | exp #7 (old shopper) | exp #9 (subjective shopper) |
|--|:--:|:--:|
| overall mean review | 1.79 | **1.57** |

So the **subjective shopper alone costs ~0.22 review** (harsher moods + taste
dislikes dock more). That means exp #9 — not exp #7 — is the apples-to-apples
baseline for the veto_batch arm (#8), which also runs the subjective shopper.
**The clean A/B is #8 vs #9.**

---

## Review scores

| metric | value |
|--------|:--:|
| overall mean review | **1.57** (σ=0.53) |
| range | 1–4 (one 4, eleven 3s) |
| distribution | 325× 1, 392× 2, 11× 3, 1× 4 |

### Per persona
| persona | mean | range |
|---------|:--:|:--:|
| office_daniel | **1.81** | 1–4 |
| casual_sofia | 1.65 | 1–2 |
| party_maya | 1.26 | 1–2 |

party_maya (bad/harsh mood) is the floor; office_daniel (generous) the ceiling —
the mood design is clearly biting.

### Per agent (marginal, n=243/cell)
| agent | best → worst | spread | read |
|-------|--------------|:--:|------|
| colour | harmonizer 1.60 → adventurous 1.58 → purist 1.54 | **0.06** | **noise** |
| body | flattering_only 1.60 → … → strict | ~0.06 | noise |
| clothing | weighted_axes 1.60 → … → match_count | ~0.05 | noise |
| stock | overstock_aggressive 1.65 → … → push | ~0.10 | borderline |

**Personalities are ~noise under borda** — exactly the exp #7 conclusion, and the
opposite of exp #8 (where colour was a real 0.16 monotonic gradient). This is the
crux: **borda re-ranks a hard-filtered narrow pool, so swapping a personality
barely changes which items surface.** See the diversity numbers below.

---

## Diversity

| measure | exp #7 | exp #9 | exp #8 (veto) |
|---------|:--:|:--:|:--:|
| TOTAL distinct items (all turns) | 1,112 | **1,258** | 8,978 |
| party_maya cell — distinct items | 16 | 22 | 724 |
| party_maya cell — mean Jaccard | 0.70 | 0.46 | 0.002 |

borda+new-shopper is *marginally* more varied than exp #7 (the taste-steered
conversations shift the hard-filter pool a little — Jaccard 0.70→0.46), but it's
still **fundamentally narrow**: ~1.3k items vs veto_batch's ~9k. The hard
color/type filter remains the ceiling on variety.

---

## Takeaway

borda on the current code = **higher satisfaction (1.57) but personalities don't
matter and the pool stays narrow** — the same structural limitation exp #7
diagnosed. It's the "safe, on-theme, low-variety" arm. The contrast with
veto_batch (#8) is the whole point of the A/B → see the comparison doc.

See [`README.md`](README.md), [`by_agent.md`](by_agent.md),
[`by_persona.md`](by_persona.md), [`combos.md`](combos.md).
