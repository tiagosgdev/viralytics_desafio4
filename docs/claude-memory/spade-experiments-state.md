---
name: spade-experiments-state
description: Current progress + the remaining roadmap for the SPADE multi-agent recommender (desafio4). START HERE to resume.
metadata: 
  node_type: memory
  type: project
  originSessionId: 3d388e58-6fca-476f-b7b3-bb57e026d8ad
---

Work on the **SPADE multi-agent fashion recommender** (Viralytics FashionSense, ISEP desafio4).
**Branch:** `spade-dynamic-weights-experiments`. **HEAD:** `e4703d4` (2026-06-24).
Plan doc in repo: `docs/plans/spade-dynamic-weights-experiments.md`.

## DONE (committed)
- Parts **A** (conversation-driven 4-agent weights), **B** (pluggable agent personalities under
  `multi_agent/strategies/`), **C** (detection confidence → weights, backend+frontend),
  **D** (shared `multi_agent/retrieval.py`), **F** (per-agent write-only memory `multi_agent/memory.py`).
- **Part E vertical slice** (`20fea1f`) — experiment harness `multi_agent/experiments/`
  (`customers.json`, `shopper.py` LLM customer via Ollama, `spec.py` OFAT, `store.py` SQLite
  `results.db`, `run_experiment.py`). Harness drives the REAL `RecommendationSystem` in-process
  and swaps personalities via `config.AGENT_STRATEGIES`. See [[recommender-architecture]].
- This session (2026-06-24):
  - **Chat → multi-agent wiring** (`156ced3` + `cb6bc95`, frontend-only). Chat now drives the
    agents via `user_answer`; confidence persists from the last scan, importance re-derived per
    turn; accumulated intent so a "correct" turn keeps the real request. See [[recommender-architecture]].
  - **Retrieval bug fix** (`e4703d4`) — `body_type` (a clothing.db attr, not a stock column) was
    injected into the stock filter → `ValueError` → silent overstock fallback every round.
    `multi_agent/retrieval.py` now prunes filters to the stock agent's `QUERY_KEYS` and logs the
    fallback. Body matching unaffected (body agent reads clothing.db independently).
  - **Pose analyzer fixed** (env, not code) — see [[spade-env-setup]] (mediapipe + model download).

## VERIFIED LIVE (2026-06-24, app + broker + Ollama up)
- Chat moves the weights (not the 30/30/25/15 fallback); an OVERRIDE ("I want blue") swaps
  include↔exclude and the agents re-score the new value (`colour.py:_resolve_detected` reads the
  override first); a pure negation ("no red") on a color you're not wearing correctly drops nothing.
- Retrieval no longer dead-ends to overstock (stock proposed real items, not the `9726/1316/600`
  overstock signature; no fallback WARNING).
- Pose/body works (`body='oval'`, body agent scores non-flat).
- **Per-agent memory works for ALL 4 scorer agents** — `multi_agent/memory/{body,clothing,colour,stock}.db`
  each hold per-round rows (`conv_id, timestamp, context_json, top_scores_json`) with the same
  round context but each agent's OWN top scores. (RL has no per-agent db by design.)

## RUN 1 DONE + FIXES APPLIED (2026-06-24)
First live OFAT run completed (27 episodes, ~70 min, exp_id=1). Findings + the two fixes applied:
see [[spade-experiment-run1-findings]]. Summary: harness valid; scores were low due to (#1) only
~7 distinct candidate items (size-flood) and (#2) a HARNESS bug — it sent only the latest shopper
msg, not accumulated intent, so the colour anchor drifted by the final turn (which is what's reviewed).
Both fixed & verified live (round now returns 10 distinct, colour/type-anchored items). #3 (price
dropped at retrieval) left unfixed. **These fixes are UNCOMMITTED** — commit before pulling elsewhere.

## NEXT — the remaining roadmap (in order)
1. **Re-run the OFAT slice with the fixes** (`docker compose up -d xmpp`, stop any uvicorn app, then
   `python3 -m multi_agent.experiments.run_experiment`). Expect higher/spread reviews. Confirm
   `results.db` populates and the mean-per-combo table prints. (Run 1 = exp_id 1; re-run = exp_id 2.)
   **GOTCHA:** harness starts its OWN `RecommendationSystem` (same XMPP JIDs as the FastAPI app), so
   **stop the uvicorn app first** (keep only the broker) or agents collide on the broker.
2. **If scores still compress** → bump shopper to a stronger model (the turn-msgs hallucinated on
   7b-q3) and raise `repeats` (spec.py) to 3 for signal. Optionally fix #3 (price/soft-attrs).
3. **Then run the FULL scope** (full factorial grid; feasible overnight on the 16GB box). Full-grid
   generator + `metrics.py`/`report.py`/plots are Part E follow-up.

## OUT OF SCOPE — IGNORE (do not build unless asked)
- `filipe` branch (TTS) merge.
- body-shape → fit mapping (the body agent matches detected shape → garment's designed-for shapes
  in clothing.db; it does NOT map to fit, and that's fine — see [[recommender-architecture]]).

## Env / how to run / workflow
[[spade-env-setup]] (system python3, mediapipe/pose fix, don't `pip install -r`, broker, tests),
[[spade-workflow]] (plan→implement-subagent→review-subagent→show→commit; route ALL impl through
subagents), [[spade-key-design-decisions]] (weight=emphasis×confidence, weighted Borda, RL slice).
Non-code gap: fine-tuned YOLO + FashionNet weights still missing locally (team SharePoint bundle,
README.md:41) — degrades detection quality but unrelated to the SPADE code.
