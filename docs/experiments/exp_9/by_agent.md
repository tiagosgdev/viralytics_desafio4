# Marginal review per agent strategy — experiment #9

For each scorer agent, the mean review across **every** combo in which the agent
used that strategy (the other three agents vary). Larger n per cell than the
per-combo view, so this is the cleaner "which personality helps" signal. Δ is
versus the agent's baseline strategy.

### colour
baseline = `purist`

| strategy | marginal mean | n | std | Δ vs baseline |
|----------|:------------:|:-:|:---:|:------------:|
| harmonizer | 1.60 | 243 | 0.56 | +0.06 |
| adventurous | 1.58 | 243 | 0.52 | +0.04 |
| purist _(baseline)_ | 1.54 | 243 | 0.52 | +0.00 |
### body
baseline = `strict`

| strategy | marginal mean | n | std | Δ vs baseline |
|----------|:------------:|:-:|:---:|:------------:|
| flattering_only | 1.60 | 243 | 0.55 | +0.03 |
| strict _(baseline)_ | 1.57 | 243 | 0.52 | +0.00 |
| lenient | 1.55 | 243 | 0.52 | -0.02 |
### clothing
baseline = `match_count`

| strategy | marginal mean | n | std | Δ vs baseline |
|----------|:------------:|:-:|:---:|:------------:|
| weighted_axes | 1.60 | 243 | 0.52 | +0.07 |
| strict_type | 1.59 | 243 | 0.53 | +0.06 |
| match_count _(baseline)_ | 1.53 | 243 | 0.55 | +0.00 |
### stock
baseline = `push`

| strategy | marginal mean | n | std | Δ vs baseline |
|----------|:------------:|:-:|:---:|:------------:|
| overstock_aggressive | 1.65 | 243 | 0.54 | +0.11 |
| push _(baseline)_ | 1.54 | 243 | 0.54 | +0.00 |
| bestsellers | 1.52 | 243 | 0.51 | -0.02 |

[← back to README](README.md)
