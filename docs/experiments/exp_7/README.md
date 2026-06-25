# Experiment #7 — full_factorial_grid

_Generated 2026-06-25 00:29:21 from `results.db`._

## Metadata
| field | value |
|-------|-------|
| experiment_id | 7 |
| name | full_factorial_grid |
| run mode | full_factorial_grid |
| repeats (K) | 3 |
| personas | party_maya, office_daniel, casual_sofia |
| combos | 81 |
| git sha | `1981130` |
| started | 2026-06-24 20:28:29 |

## Totals
| metric | value |
|--------|-------|
| episodes (rows) | 729 |
| reviewed | 729 |
| abandoned / NULL | 0 / 0 |
| overall mean review | **1.79** (σ=0.61) |
| review range | 1 … 4 |

### Review distribution
| review | count | |
|:------:|------:|:--|
| 5 |    0 |  |
| 4 |    6 |  |
| 3 |   55 | ██ |
| 2 |  446 | ████████████ |
| 1 |  222 | ██████ |

## Headline — best strategy per agent (marginal)
Averaged over every combo in which that agent used the strategy. See
[`by_agent.md`](by_agent.md) for the full per-strategy tables.

| agent | best strategy | marginal mean | |
|-------|---------------|:------------:|--|
| colour | **purist** | 1.81 | baseline=`purist` |
| body | **strict** | 1.82 | baseline=`strict` |
| clothing | **match_count** | 1.81 | baseline=`match_count` |
| stock | **bestsellers** | 1.80 | baseline=`push` |

## Top 5 combos
| rank | combo | mean | n |
|:----:|-------|:----:|:-:|
| 1 | `colour=harmonizer|body=flattering_only|clothing=match_count|stock=bestsellers` | 2.33 | 9 |
| 2 | `colour=harmonizer|body=flattering_only|clothing=strict_type|stock=overstock_aggressive` | 2.33 | 9 |
| 3 | `colour=purist|body=strict|clothing=weighted_axes|stock=overstock_aggressive` | 2.22 | 9 |
| 4 | `colour=adventurous|body=strict|clothing=strict_type|stock=bestsellers` | 2.22 | 9 |
| 5 | `colour=purist|body=strict|clothing=match_count|stock=push` | 2.11 | 9 |

## Bottom 5 combos
| combo | mean | n |
|-------|:----:|:-:|
| colour=harmonizer|body=lenient|clothing=weighted_axes|stock=push | 1.22 | 9 |
| colour=adventurous|body=flattering_only|clothing=match_count|stock=push | 1.44 | 9 |
| colour=adventurous|body=lenient|clothing=match_count|stock=bestsellers | 1.44 | 9 |
| colour=adventurous|body=strict|clothing=weighted_axes|stock=bestsellers | 1.44 | 9 |
| colour=harmonizer|body=lenient|clothing=weighted_axes|stock=bestsellers | 1.44 | 9 |

## Files
- [`combos.md`](combos.md) — all 81 combos ranked.
- [`by_agent.md`](by_agent.md) — marginal review per agent strategy (which personality wins).
- [`by_persona.md`](by_persona.md) — per-persona means and best/worst combo.

> ⚠️ Interpret with care: with K=3 the per-combo n is small;
> small mean gaps may be noise. The marginal (`by_agent.md`) view has much larger
> n per cell and is the more reliable signal.
