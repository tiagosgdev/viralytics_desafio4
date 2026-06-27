# RL learning curve from satisfaction

This experiment measures whether the PPO recommender agent **learns** from the
simulated shopper's 1–5 closing review. It runs many sequential **baseline**
episodes in `veto_batch` selection mode, feeds each episode's `final_review` into
the RL reward path, runs the PPO updates that the accumulated rollouts produce,
and records one *learning-curve point* (review vs cumulative PPO `update_count`)
per episode. The plotter then shows smoothed review against PPO updates with an
early-vs-late mean delta.

> **Additive / opt-in.** This mode is entirely behind the `curve` experiment mode
> and the `RL_*` env flags. With the flags unset, production and the existing
> `ofat` / `full` (borda/veto) grid runs are **byte-identical** to before — they
> can still be run exactly as today. Nothing here changes the default learning
> path (`RL_REWARD_MODE=both`: pass-rate every round + emoji when it arrives).

---

## How it works (what the flags do)

| Flag | Value here | Effect |
| --- | --- | --- |
| `EXPERIMENT_MODE` | `curve` | Runs sequential baseline episodes, defers RL rollout consumption, feeds each review, records a curve point per episode. |
| `RL_REWARD_MODE` | `rating` | Zeroes the pass-rate reward so the learning signal is **purely** the simulated 1–5 review (`both`/`passrate` keep production behaviour). |
| `RL_FRESH_START` | `1` | Starts the policy at `update_count=0` (ignores any existing checkpoint) but still **saves** to `RL_CHECKPOINT_PATH`, so the curve starts from an untrained policy. |
| `RL_CHECKPOINT_PATH` | `models/weights/agents/rl_ppo_curve.pt` | Isolated checkpoint so the curve run never touches the production `rl_ppo.pt`. |
| `SELECTION_MODE` | `veto_batch` | Selection path used for the run. |
| `EXPERIMENT_REPEATS` | `100` | Replays per (customer × combo); the curve mode uses a single baseline combo, so this is the episode multiplier. |

In `curve` mode the round store sets `defer_consumption = True`, so each episode's
review can land on its round's transitions **before** that round is consumed. After
feedback, the episode loop drains every full rollout
(`PPO_ROLLOUT_ROUNDS = 8` rounds each) and runs one `rl_policy.learn(batch)` PPO
update per rollout.

---

## Prerequisites (GPU box)

1. **XMPP broker** (Prosody, used by the SPADE agents):
   ```
   docker compose up -d xmpp
   ```
2. **Ollama models** must be pulled and Ollama reachable. Two models are used:
   - the simulated shopper / reviewer — `OLLAMA_SHOPPER_MODEL`, default
     **`qwen2.5:14b-instruct`** (`multi_agent/experiments/shopper.py`);
   - the `FeatureWeightAgent` intent refiner — `OLLAMA_REFINER_MODEL`, default
     **`qwen2.5:7b-instruct-q3_K_M`**
     (`LNIAGIA/query_parsing/llm_query_parser.py`).
   ```
   ollama pull qwen2.5:14b-instruct
   ollama pull qwen2.5:7b-instruct-q3_K_M
   ```
3. Python deps installed (the same env that runs the normal harness; `torch`,
   `spade`, `ollama`, `matplotlib`, `numpy`).

---

## Run

From the repo root:

```
RL_FRESH_START=1 \
RL_CHECKPOINT_PATH=models/weights/agents/rl_ppo_curve.pt \
RL_REWARD_MODE=rating \
SELECTION_MODE=veto_batch \
EXPERIMENT_MODE=curve \
EXPERIMENT_REPEATS=100 \
python -m multi_agent.experiments.run_experiment
```

**Scale & runtime.** Episodes = `personas × 1 baseline combo × repeats`. With
~3 personas × 100 repeats ≈ **~300 episodes**, and one PPO update per 8 settled
rounds, that yields **~100 PPO updates** — enough resolution for a curve. Each
episode runs a multi-turn LLM conversation plus a review, so a full run takes
**hours** on the GPU box. For a quick smoke test, lower `EXPERIMENT_REPEATS`
(e.g. `5`); the curve will be short but the plumbing is identical.

**Outputs**
- `multi_agent/experiments/results.db` → `curve_points` table (one row/episode:
  `episode_index`, `update_count`, `rating`, `rewards_landed`, `rewards_dropped`,
  `mean_return`).
- `models/weights/agents/rl_ppo_curve.pt` → the trained-during-the-run checkpoint.
- An auto-generated Markdown report under `multi_agent/experiments/reports/`.
- Console summary printing total **rewards landed / dropped** and the PPO update
  count.

---

## Plot

```
python -m multi_agent.experiments.plot_learning_curve
```

(pass an `experiment_id` to plot a specific run; otherwise the latest run with
curve points is used). Read-only on `results.db`. Output:

```
multi_agent/experiments/plots/rl_learning_curve.png
```

The plot shows raw review points, a rolling mean (window ≈ 15) vs cumulative PPO
updates, and the early(first 25%)-vs-late(last 25%) mean review delta. A positive
delta is the evidence that the policy improved as it learned from satisfaction.

---

## Sanity check

`rewards_dropped` **must be `0`**. The console summary and the plotter both print
the total. A non-zero count means some reviews were dropped (the round was
consumed before its feedback landed, or the rating path was disabled by
`RL_REWARD_MODE`) — investigate before trusting the curve. In `curve` mode the
store defers consumption specifically to keep this at zero.
