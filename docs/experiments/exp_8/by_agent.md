# Marginal review per agent strategy — experiment #8

For each scorer agent, the mean review across **every** combo in which the agent
used that strategy (the other three agents vary). Larger n per cell than the
per-combo view, so this is the cleaner "which personality helps" signal. Δ is
versus the agent's baseline strategy.

### colour
baseline = `purist`

| strategy | marginal mean | n | std | Δ vs baseline |
|----------|:------------:|:-:|:---:|:------------:|
| purist _(baseline)_ | 1.35 | 243 | 0.48 | +0.00 |
| harmonizer | 1.29 | 243 | 0.49 | -0.06 |
| adventurous | 1.19 | 243 | 0.39 | -0.16 |
### body
baseline = `strict`

| strategy | marginal mean | n | std | Δ vs baseline |
|----------|:------------:|:-:|:---:|:------------:|
| lenient | 1.30 | 243 | 0.46 | +0.02 |
| strict _(baseline)_ | 1.28 | 243 | 0.48 | +0.00 |
| flattering_only | 1.24 | 243 | 0.44 | -0.05 |
### clothing
baseline = `match_count`

| strategy | marginal mean | n | std | Δ vs baseline |
|----------|:------------:|:-:|:---:|:------------:|
| strict_type | 1.28 | 243 | 0.48 | +0.01 |
| weighted_axes | 1.28 | 243 | 0.45 | +0.00 |
| match_count _(baseline)_ | 1.27 | 243 | 0.45 | +0.00 |
### stock
baseline = `push`

| strategy | marginal mean | n | std | Δ vs baseline |
|----------|:------------:|:-:|:---:|:------------:|
| bestsellers | 1.29 | 243 | 0.46 | +0.02 |
| overstock_aggressive | 1.28 | 243 | 0.47 | +0.01 |
| push _(baseline)_ | 1.26 | 243 | 0.45 | +0.00 |

[← back to README](README.md)
