# Experiment #8 — full_factorial_grid

_Generated 2026-06-25 21:51:19 from `results.db`._

## Metadata
| field | value |
|-------|-------|
| experiment_id | 8 |
| name | full_factorial_grid |
| run mode | full_factorial_grid |
| repeats (K) | 3 |
| personas | party_maya, office_daniel, casual_sofia |
| combos | 81 |
| git sha | `7870a40` |
| started | 2026-06-25 18:03:29 |

## Totals
| metric | value |
|--------|-------|
| episodes (rows) | 729 |
| reviewed | 729 |
| abandoned / NULL | 0 / 0 |
| overall mean review | **1.28** (σ=0.46) |
| review range | 1 … 3 |

### Review distribution
| review | count | |
|:------:|------:|:--|
| 5 |    0 |  |
| 4 |    0 |  |
| 3 |    4 |  |
| 2 |  193 | █████ |
| 1 |  532 | ███████████████ |

## Headline — best strategy per agent (marginal)
Averaged over every combo in which that agent used the strategy. See
[`by_agent.md`](by_agent.md) for the full per-strategy tables.

| agent | best strategy | marginal mean | |
|-------|---------------|:------------:|--|
| colour | **purist** | 1.35 | baseline=`purist` |
| body | **lenient** | 1.30 | baseline=`strict` |
| clothing | **strict_type** | 1.28 | baseline=`match_count` |
| stock | **bestsellers** | 1.29 | baseline=`push` |

## Top 5 combos
| rank | combo | mean | n |
|:----:|-------|:----:|:-:|
| 1 | `colour=purist|body=strict|clothing=match_count|stock=push` | 1.56 | 9 |
| 2 | `colour=purist|body=lenient|clothing=match_count|stock=overstock_aggressive` | 1.56 | 9 |
| 3 | `colour=purist|body=lenient|clothing=weighted_axes|stock=push` | 1.56 | 9 |
| 4 | `colour=purist|body=lenient|clothing=strict_type|stock=overstock_aggressive` | 1.56 | 9 |
| 5 | `colour=purist|body=strict|clothing=weighted_axes|stock=bestsellers` | 1.44 | 9 |

## Bottom 5 combos
| combo | mean | n |
|-------|:----:|:-:|
| colour=adventurous|body=lenient|clothing=weighted_axes|stock=push | 1 | 9 |
| colour=harmonizer|body=flattering_only|clothing=match_count|stock=overstock_aggressive | 1 | 9 |
| colour=adventurous|body=flattering_only|clothing=strict_type|stock=push | 1.11 | 9 |
| colour=adventurous|body=flattering_only|clothing=weighted_axes|stock=bestsellers | 1.11 | 9 |
| colour=adventurous|body=flattering_only|clothing=weighted_axes|stock=overstock_aggressive | 1.11 | 9 |

## Files
- [`combos.md`](combos.md) — all 81 combos ranked.
- [`by_agent.md`](by_agent.md) — marginal review per agent strategy (which personality wins).
- [`by_persona.md`](by_persona.md) — per-persona means and best/worst combo.

> ⚠️ Interpret with care: with K=3 the per-combo n is small;
> small mean gaps may be noise. The marginal (`by_agent.md`) view has much larger
> n per cell and is the more reliable signal.
