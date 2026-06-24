# Marginal review per agent strategy — experiment #7

For each scorer agent, the mean review across **every** combo in which the agent
used that strategy (the other three agents vary). Larger n per cell than the
per-combo view, so this is the cleaner "which personality helps" signal. Δ is
versus the agent's baseline strategy.

### colour
baseline = `purist`

| strategy | marginal mean | n | std | Δ vs baseline |
|----------|:------------:|:-:|:---:|:------------:|
| purist _(baseline)_ | 1.81 | 243 | 0.64 | +0.00 |
| harmonizer | 1.79 | 243 | 0.61 | -0.02 |
| adventurous | 1.76 | 243 | 0.57 | -0.05 |
### body
baseline = `strict`

| strategy | marginal mean | n | std | Δ vs baseline |
|----------|:------------:|:-:|:---:|:------------:|
| strict _(baseline)_ | 1.82 | 243 | 0.63 | +0.00 |
| flattering_only | 1.82 | 243 | 0.60 | +0.00 |
| lenient | 1.72 | 243 | 0.58 | -0.11 |
### clothing
baseline = `match_count`

| strategy | marginal mean | n | std | Δ vs baseline |
|----------|:------------:|:-:|:---:|:------------:|
| match_count _(baseline)_ | 1.81 | 243 | 0.61 | +0.00 |
| strict_type | 1.80 | 243 | 0.62 | -0.01 |
| weighted_axes | 1.75 | 243 | 0.59 | -0.06 |
### stock
baseline = `push`

| strategy | marginal mean | n | std | Δ vs baseline |
|----------|:------------:|:-:|:---:|:------------:|
| bestsellers | 1.80 | 243 | 0.62 | +0.04 |
| overstock_aggressive | 1.79 | 243 | 0.62 | +0.03 |
| push _(baseline)_ | 1.77 | 243 | 0.58 | +0.00 |

[← back to README](README.md)
