# Experiment #9 — full_factorial_grid

_Generated 2026-06-26 01:40:15 from `results.db`._

## Metadata
| field | value |
|-------|-------|
| experiment_id | 9 |
| name | full_factorial_grid |
| run mode | full_factorial_grid |
| repeats (K) | 3 |
| personas | party_maya, office_daniel, casual_sofia |
| combos | 81 |
| git sha | `7870a40` |
| started | 2026-06-25 21:51:51 |

## Totals
| metric | value |
|--------|-------|
| episodes (rows) | 729 |
| reviewed | 729 |
| abandoned / NULL | 0 / 0 |
| overall mean review | **1.57** (σ=0.53) |
| review range | 1 … 4 |

### Review distribution
| review | count | |
|:------:|------:|:--|
| 5 |    0 |  |
| 4 |    1 |  |
| 3 |   11 |  |
| 2 |  392 | ███████████ |
| 1 |  325 | █████████ |

## Headline — best strategy per agent (marginal)
Averaged over every combo in which that agent used the strategy. See
[`by_agent.md`](by_agent.md) for the full per-strategy tables.

| agent | best strategy | marginal mean | |
|-------|---------------|:------------:|--|
| colour | **harmonizer** | 1.60 | baseline=`purist` |
| body | **flattering_only** | 1.60 | baseline=`strict` |
| clothing | **weighted_axes** | 1.60 | baseline=`match_count` |
| stock | **overstock_aggressive** | 1.65 | baseline=`push` |

## Top 5 combos
| rank | combo | mean | n |
|:----:|-------|:----:|:-:|
| 1 | `colour=purist|body=strict|clothing=match_count|stock=overstock_aggressive` | 1.89 | 9 |
| 2 | `colour=purist|body=strict|clothing=strict_type|stock=bestsellers` | 1.89 | 9 |
| 3 | `colour=purist|body=flattering_only|clothing=weighted_axes|stock=overstock_aggressive` | 1.89 | 9 |
| 4 | `colour=harmonizer|body=lenient|clothing=match_count|stock=overstock_aggressive` | 1.89 | 9 |
| 5 | `colour=harmonizer|body=flattering_only|clothing=match_count|stock=overstock_aggressive` | 1.89 | 9 |

## Bottom 5 combos
| combo | mean | n |
|-------|:----:|:-:|
| colour=harmonizer|body=lenient|clothing=strict_type|stock=overstock_aggressive | 1.22 | 9 |
| colour=adventurous|body=flattering_only|clothing=match_count|stock=push | 1.33 | 9 |
| colour=adventurous|body=lenient|clothing=weighted_axes|stock=bestsellers | 1.33 | 9 |
| colour=adventurous|body=lenient|clothing=match_count|stock=bestsellers | 1.33 | 9 |
| colour=harmonizer|body=lenient|clothing=match_count|stock=bestsellers | 1.33 | 9 |

## Files
- [`combos.md`](combos.md) — all 81 combos ranked.
- [`by_agent.md`](by_agent.md) — marginal review per agent strategy (which personality wins).
- [`by_persona.md`](by_persona.md) — per-persona means and best/worst combo.

> ⚠️ Interpret with care: with K=3 the per-combo n is small;
> small mean gaps may be noise. The marginal (`by_agent.md`) view has much larger
> n per cell and is the more reliable signal.
