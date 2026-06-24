---
name: spade-key-design-decisions
description: Settled design decisions for the SPADE weights/personalities/confidence work
metadata: 
  node_type: memory
  type: project
  originSessionId: f5c32068-8876-46a2-a96b-d678ae671701
---

Settled decisions for the SPADE feature work (see [[spade-experiments-state]]):

- **Aggregation stays weighted Borda** (`multi_agent/aggregator.py borda_aggregate`, unchanged).
  Only the per-agent *weight* changed. `final = Σ_agent weight × borda_points`.
- **Weight formula:** `weight = emphasis × confidence`, normalised over the **emphasis budget
  = (1 − rl_weight)**. The four conversation emphases (colour/clothing/body/stock) share that
  budget; **RL is the one fixed slice** carved off the top (`RL_WEIGHT = 0.15`, enabled via
  `RL_ENABLED`). **stock and RL have no detection → confidence 1.0.** All confidences default
  to 1.0 → behaviour identical to before (backward compatible). This hybrid reconciled our
  Part A with the RL branch's old fixed-stock+rl design.
- **Detection confidence sources:** colour and type come from the SAME garment `Detection`
  (`src/detection/detector.py:96`), so they **share `d.confidence`**; body uses
  `body_analysis["confidence"]` (`body_classifier.classify_body_shape`). There is NO separate
  colour confidence and none is needed (confirmed across all branches). Flows:
  camera → frontend `triggerAgentRecommendations` → `/api/recommend` (`detected_*_conf`) →
  `recommend()` → `trigger_round` → context → `build_agent_weights(confidences=...)`.
- **RL agent** (merged from `origin/reinforcement-learning`) is a colleague's *separate* voting
  agent (5th scorer) trained with PPO; reward = user 1–5 rating via `submit_feedback`. It reads
  the **shared** `RoundHistory`, NOT the per-agent memories. `recommend()` now returns
  `(round_id, results)`.
- **Per-agent memory is write-only** (Part F) — a course requirement for "agent autonomy";
  nothing consumes it for decisions. The shared `RoundHistory` stays the global log.
- **Intent/emphases** come from the existing chat LLM `analyze_intent`
  (`LNIAGIA/query_parsing/feature_weighting.py`), extended with a 4th `stock` importance.
  `weight_agent` consumes it; orchestrator combines emphasis × confidence.

Known follow-ups (non-blocking): legacy `stock` fallback branch in `build_agent_weights` is
TODO-marked dead code once chat always sends stock; the missing-agent redistribution path
sums to 0.9999 (pre-existing `round(v,4)`); `_run_multiagent_round` in `src/api/main.py` is
dead (camera scan uses DB recs; multi-agent only reachable via `/api/recommend`).
