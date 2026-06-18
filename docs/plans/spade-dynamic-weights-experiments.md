# SPADE: Conversation-Driven Weights + Agent Personalities + Experiment Harness

## Context

The SPADE multi-agent recommender (`multi_agent/`) works but is rigid:

- **Weights are mostly hard-set.** `StockAgent` weight is fixed at `STOCK_WEIGHT=0.20`
  (`config.py:27`); only the body/clothing/colour split reacts to conversation, via a
  rigid 1:1 mapping (color→colour, type→clothing, bodyType→body) in
  `build_agent_weights` (`aggregator.py:58`). No-answer falls back to a fixed 33/34/33.
- **Each agent has one hard-coded scoring behaviour** with magic numbers scattered
  inline (e.g. colour exact=1.0/compat=0.65/unrelated=0.20 at `colour_agent.py:58`;
  body exact=1.0/adjacent=0.55/nodata=0.20 at `body_agent.py:93`). No way to try
  alternative scoring "personalities" or compare them.
- **Aggregation (weighted Borda) is kept as-is.** `aggregator.py:21` already computes
  `final_score(item) = Σ_agents weight_agent × borda_points_agent(item)`. The Borda
  mechanism does **not** change; only `weight_agent` becomes dynamic (Part A).
- **No per-agent memory and no quality signal.** `history.py` is a single shared store of
  round metadata; it never records weights used, per-agent scores, the final ranking, or
  any user feedback. There is no experiment concept.

**Goals:**
1. Make all four agent weights conversation-driven, including stock — so the *same*
   weighted-Borda formula yields a different blend per shopper intent. The feature-weighting
   already exists (`analyze_intent`, consumed by `weight_agent`); just extend it with a
   stock dimension and have SPADE use all four (Part A).
2. Make each agent's scoring a swappable **"personality"** with configurable params (Part B).
3. Give **each agent its own memory** (Part F) — duplicated per-agent stores; a course
   requirement (agent autonomy), even though it overlaps the shared history.
4. Build an **episodic, factorial experiment harness** (Part E): a fixed bank of LLM-played
   customers × combinations of agent personalities, each run as a full multi-turn
   conversation ending in a simulated **1–5 review** — which doubles as the reward signal
   the RL branch needs (`docs/.../07_reinforcement_learning/rl_proposal.md` §3.1).

---

## Design overview

**Architectural move: decouple agent scoring from SPADE transport.** Each agent's scoring
becomes a pure strategy function that both the live SPADE agent *and* the experiment
harness call. Scoring + aggregation run **in-process** (fast, deterministic, attributable);
the LLM parts (the shopper persona and the system's chat-intent agent) are seeded and
cached so episodes are reproducible/replayable.

```
                 ┌─────────────────────────────────────────┐
                 │  multi_agent/strategies/  (pure funcs)   │
                 │  colour.py body.py clothing.py stock.py  │
                 │  + registry.py (name → fn + default params)
                 └───────────────┬───────────────┬──────────┘
        live path ───────────────┘               └─────────── experiment path
   agents/*.py wrap a strategy            experiments/run_experiment.py
   (selected via config)                  drives a multi-turn episode per (customer × combo)
```

---

## Part A — Conversation-driven weights (already wired; extend to 4 agents)

The feature-weighting already exists: `analyze_intent` (`feature_weighting.py`) is one LLM
call that returns `weights: {color, type, bodyType}` (importances summing to 100) + filters,
and `weight_agent.py` already consumes it and feeds `build_agent_weights`. So the
chat-intent → weights path is in place; only two small changes are needed plus confidence.

1. **Add a `stock` emphasis** to `analyze_intent` — extend its system prompt + output so it
   also scores stock/inventory intent (cues like "popular", "on sale", "what's trending",
   "clearance" raise it; "I specifically want X" lowers it). Keep all importances > 0 and
   renormalise to sum 100. Add a default `stock` importance to `weight_agent.py`'s
   `_rule_based_weights` / `_FALLBACK_WEIGHTS` for the no-answer fast path.
2. **Rewrite `build_agent_weights` (`aggregator.py:58`)** to derive all four weights from the
   four emphases (normalise to sum 1.0), optionally damped by detection confidence (Part C).
   Remove the fixed `STOCK_WEIGHT` budget-split; keep the missing-agent redistribution.
   `STOCK_WEIGHT`/`USER_WEIGHT` survive only as fallback importances.
3. Orchestrator (`orchestrator.py:155`) keeps calling `build_agent_weights`; the call
   simplifies (no separate stock budget passed).

> If the colleague's chat layer already calls `analyze_intent` and forwards its output,
> `weight_agent` can consume that directly instead of re-deriving — same shape either way.
> Multi-turn refinement just means each chat turn supplies a fresh `user_answer`, so the
> emphases (and therefore the weights) update turn to turn for free.

## Part B — Pluggable agent "personalities" (strategies)

1. **New package `multi_agent/strategies/`** — one module per agent exposing pure
   functions `score(candidates, context, weights_result, params) -> {item_key: float in [0,1]}`.
   Move existing inline logic in as the default ("baseline") strategy, lifting every magic
   number into a `params` dict.
2. **`strategies/registry.py`** maps `agent → {strategy_name: (fn, default_params)}`.
   Starter personalities (finalise during impl):
   - **colour**: `purist` (current), `harmonizer` (rewards complementary variation),
     `adventurous` (rewards contrast/variety).
   - **body**: `strict` (current), `lenient` (higher adjacent/no-data credit),
     `flattering_only` (adjacent penalised hard).
   - **clothing**: `match_count` (current), `weighted_axes` (type/style/occasion weighted),
     `strict_type` (type dominates).
   - **stock**: `push` (current normalised push_score), `overstock_aggressive`,
     `bestsellers` (favours sales velocity).
3. **Thin the SPADE agents** (`agents/{colour,body,clothing,stock}_agent.py`): each
   behaviour resolves its configured strategy from the registry and calls it. Strategy +
   params selected via `config.py` (env-overridable), e.g.
   `AGENT_STRATEGIES = {"colour": "purist", ...}`. The harness overrides this per run.

## Part C — Simulated detection confidence

Real detections come from CV models with a confidence; in simulation there are no models,
so confidence must be **synthesised and plumbed through**.

- Each scenario/customer carries `detected_*_conf ∈ [0,1]` for color/type/body_type.
- Confidence is added to the round `context` (extend `trigger_round`/`recommend` kwargs in
  `orchestrator.py:384` / `run.py:95`) and recorded.
- **Optional weighting knob** (`CONFIDENCE_WEIGHTING`): low confidence on a feature damps
  the corresponding agent's weight in `build_agent_weights` (a low-confidence body
  detection should not dominate). Off = today's behaviour; on = a comparable experiment arm.

## Part D — Aggregation: unchanged + shared retrieval

- **Keep weighted Borda** (`aggregator.py:21`). No new aggregation function; dynamism comes
  entirely from `weight_agent` (Part A).
- Extract orchestrator `_get_candidates` (`orchestrator.py:223`) into
  `multi_agent/retrieval.py` `get_candidates(stock_agent, weights_result, context, n)` so
  the orchestrator and the harness retrieve identical candidate sets.

## Part E — Episodic, factorial experiment harness

New package **`multi_agent/experiments/`**. The unit of study is an **episode**, not a
single round.

1. **`customers.json`** — frozen bank of personas (≈8–15). Each: personality/temperament,
   detected color/type/body_type **+ confidences** (Part C), gender, and a **hidden goal /
   preference profile** the shopper is steering toward. The LLM shopper is prompted with
   one persona and role-plays the whole conversation.
2. **`shopper.py`** — the LLM-played customer. Given `persona + conversation-so-far +
   current recommendations`, it produces the next chat message, eventually a stop, and a
   final **1–5 review** with a short reason. Occupies the **human seat** in the real
   conversation loop (same seam a real user / the colleague's chat UI fills). Seeded;
   calls cached by `(persona, conversation state, system output)` for replay.
3. **`spec.py`** — `ExperimentSpec`: per-agent `{strategy, params}`, the `CONFIDENCE_WEIGHTING`
   flag, repeats `K`. Sweep generators: **OFAT** (vary one agent, others baseline — ~12
   combos) and **full grid** (~81 combos). OFAT is the default smoke pass; full grid is
   feasible overnight on a 16GB GPU.
4. **`run_experiment.py`** — for each `(customer × personality-combo × repeat)`: run the
   episode loop — shopper message → **chat agent intent** → weights → round (strategies +
   Borda, in-process) → recommendations → shopper reacts → … → 1–5 review. Build
   `StockStats`/`StockAgent` once (`run.py` pattern).
5. **`store.py`** — SQLite `multi_agent/experiments/results.db`:
   - `experiments(experiment_id, name, spec_json, git_sha, created_at)`
   - `episodes(episode_id, experiment_id, customer_id, combo_json, seed, n_turns, final_review, abandoned)`
   - `turns(turn_id, episode_id, idx, shopper_msg, intent_json, weights_json)`
   - `turn_items(turn_id, rank, item_id, size, final_score, agent_scores_json, item_attrs_json)`
   - **`rl_dataset` export (JSONL)**: one row per episode =
     `(context, chosen personalities/action, full chat history, 1–5 reward)` — exactly the
     contextual-bandit training tuple the RL branch needs. The experiment output *is* the
     RL dataset.
6. **`metrics.py` + `report.py`** — primary metric = **mean 1–5 review** per personality
   combo; secondary = turns-to-satisfaction, abandonment rate, convergence toward the
   hidden goal. Comparison tables + plots per experiment (follow `stock_agent/sanity_plots.py`
   precedent) → `multi_agent/experiments/results/<name>/`.

## Part F — Per-agent memory (course requirement, feedback-enabled)

Each scorer agent gets its **own** memory store, recording each round from its own
perspective. Professor wants explicit agent autonomy; we make it meaningful by closing the
feedback loop (which also serves as the RL feedback channel, `rl_proposal.md` §3.2).

1. **`multi_agent/memory.py` `AgentMemory(agent_id)`** — generalise the SQLite-persistence
   logic from `history.py`. Each agent owns its own file `multi_agent/memory/<agent>.db`.
   Per-round record (from the agent's own view):
   `conv_id/episode_id, context, my_top_scores, my_weight, my_picks_in_final_top10,
   episode_review(1–5)`.
2. **Feedback broadcast (new message).** Add a feedback performative to
   `multi_agent/messages.py` (e.g. `make_feedback`). At round end the orchestrator sends
   each scorer agent: its assigned `weight`, which of its picks landed in the final top-10,
   and (at episode end, when known) the 1–5 review. The orchestrator already holds all of
   this in `_build_result` (`orchestrator.py:331`).
3. **Each agent records autonomously.** Scorer agents add a small feedback behaviour
   (template on the feedback performative) that writes the round to *its own* `AgentMemory`.
   On `setup()`, an agent loads its own memory for the comeback summary — replacing the
   current shared `history.agent_context_summary(...)` call.
4. **Keep the shared `RoundHistory`** as the global/round-level log (queue staleness, etc.).
   The per-agent duplication is intentional and documented.

> This makes each agent's memory a genuine log of *its own* decisions and how they fared —
> and emits exactly the `(context, action, reward)` linkage the RL bandit will later consume,
> per agent, matching the Borda credit-assignment design in `rl_proposal.md` §2.3.

---

## Critical files

- Modify: `multi_agent/aggregator.py`, `multi_agent/config.py`,
  `multi_agent/agents/{orchestrator,weight_agent,colour_agent,body_agent,clothing_agent,stock_agent}.py`,
  `multi_agent/run.py`, `multi_agent/messages.py` (feedback performative),
  `LNIAGIA/query_parsing/feature_weighting.py` (add stock emphasis).
- New: `multi_agent/retrieval.py`, `multi_agent/strategies/` (registry + 4 modules),
  `multi_agent/memory.py`, `multi_agent/experiments/`
  (customers.json, shopper.py, spec.py, run_experiment.py, store.py, metrics.py, report.py).
- Reuse: `run.py` startup pattern, `history.py` SQLite pattern (→ `memory.py`),
  `stock_agent/sanity_plots.py` plotting precedent, `_StockAgent`/`StockStats`.

## Dependencies & sequencing

- **Part A is unblocked:** feature-weighting (`analyze_intent`) already exists and is
  already consumed by `weight_agent`. No coordination needed beyond extending it with the
  stock dimension. If the colleague's chat layer forwards `analyze_intent` output, consume
  that directly; otherwise keep deriving it in `weight_agent`.
- **RL branch:** not in this branch. We only *produce* the reward (1–5) and the `rl_dataset`
  export; consuming it (bandit updates) is the RL branch's job.
- Everything here is internal to this repo — no hard external blockers. Natural order:
  Part A (stock dim + 4-way weights) and Parts B (strategies), D (retrieval),
  F (per-agent memory) first; then the episodic harness (Part E) on top.

## Backward compatibility

- `borda_aggregate` untouched; with default strategies + fallback weights + confidence
  weighting off, live behaviour matches today.
- `RecommendationSystem.recommend()` gains optional confidence kwargs (defaulted); result
  dict still exposes `agent_scores`/`agent_weights`, so the FastAPI layer keeps working.

## Verification

1. **Unit:** `build_agent_weights` (4-way normalisation, confidence damping, missing-agent
   redistribution); each strategy monotonic in its signal; a regression test pinning
   `borda_aggregate`. `AgentMemory` round-trips per-agent records. Run `pytest tests/`.
2. **Registry:** every configured strategy resolves and returns scores in [0,1].
3. **Live round:** `docker compose up -d xmpp` then `python -m multi_agent.run`; confirm a
   top-10 resolves, `agent_weights` includes a conversation-derived stock weight, the
   orchestrator broadcasts feedback, and each agent wrote a record (with its weight +
   picks-in-top-10) to its **own** `multi_agent/memory/<agent>.db`.
4. **Harness episode:** run one customer × OFAT sweep; confirm `results.db` is populated
   (experiments/episodes/turns/turn_items), the `rl_dataset` JSONL exports
   `(context, action, chat history, 1–5 reward)`, a report renders, and re-running with the
   same seed replays from cache identically.
5. **Scale check:** full grid on the 16GB GPU box overnight; confirm throughput and that
   mean-review rankings across personality combos are stable across `K` repeats.
