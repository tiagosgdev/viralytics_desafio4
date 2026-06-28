# Technical presentation — slide content (tables + 1-paragraph explanations)


## Slide 1 — Experimental design (two isolated simulations)

**Explanation.** An LLM shopper (`qwen2.5:14b-instruct`) role-plays three customer personas through full conversations
and returns per-item 1–5 reviews. We run **two separate simulations so each effect is isolated**: **Simulation A** sweeps
81 agent-personality combinations × 3 personas × 3 repeats (729 episodes) with the RL agent fixed — measuring whether the
*agent personalities* move the review; **Simulation B** holds personalities at default and lets the *RL agent learn*
online over 300 episodes — measuring whether learning raises satisfaction. Both run the real system end-to-end (0 errors,
0 dropped rewards).

**Key code** — `multi_agent/experiments/run_experiment.py`, `multi_agent/experiments/shopper.py`
```python
if   mode == "full":  combos = list(full_factorial_combos())  # Sim A: 81 personality combos
elif mode == "curve": combos = [baseline_combo()]; feed_review = True  # Sim B: RL learns online

# per-item feedback: each top-10 item gets ITS OWN 1–5 rating → its own RL reward
for item in last_recs[:config.TOP_K]:
    r = item_ratings.get((int(item["item_id"]), str(item["size"])), rating)
    system.submit_feedback(round_id=last_round_id, item_id=int(item["item_id"]), rating=int(r))
```

---

## Slide 2 — Retrieval strategy: focused vs broad (both aggregate with Borda)

| Metric                      | Focused retrieval (chosen) |  Broad + veto   |
| --------------------------- | :------------------------: | :-------------: |
| Mean review (1–5)           |          **1.57**          |      1.28       |
| Distinct items surfaced     |           1,258            |  8,978 (7.1×)   |
| Personalities shift review? |        no (Δ ≤ 0.06)       | yes (Δ = 0.16)  |

*(Both arms aggregate the agents' votes with the same weighted, tie-aware Borda count.)*

**Explanation.** What differs between the two arms is the **item retrieval that feeds the vote**, not the aggregation —
both finish with the same weighted, tie-aware **Borda count**. The *focused* arm hard-filters to a tight intent match
(colour ∧ type, ~45 candidates); the *broad* arm retrieves any item matching ≥1 signal (~1,300) and lets the agents
**veto** weak items down to the slate before ranking. The focused pool wins customer satisfaction (**1.57 vs 1.28**) and
is deployed; the broad+veto pool surfaces ~7× more variety and is the only one where swapping an agent's personality
measurably moves the result (Δ = 0.16) — kept as future personalization capacity, but not worth the relevance cost on
satisfaction.

**Key code** — `multi_agent/agents/orchestrator.py`, `multi_agent/aggregator.py`
```python
# orchestrator.py:146 — the arms differ in RETRIEVAL (+ veto), NOT aggregation
if SELECTION_MODE == "veto_batch":           # broad retrieval (~1,300 candidates)
    # agents veto weak items: drop when Σ weight(vetoing agents) ≥ τ (=0.5)
    pool = [it for it in survivors if reject_mass(it, vetoes, agent_weights) < τ]
    top_k = borda_aggregate(pool, agent_weights, k=TOP_K)
else:                                        # focused retrieval (~45, colour ∧ type)
    top_k = borda_aggregate(proposals, agent_weights, k=TOP_K)
# → BOTH paths finish with the SAME weighted, tie-aware Borda count (aggregator.py:95)
```

---

## Slide 3 — Simulation A: agent personalities (Borda)

| Colour personality      |   Mean review (1–5)  |
| ----------------------- | :------------------: |
| purist (on-theme)       |     `[TBD-grid]`     |
| harmonizer              |     `[TBD-grid]`     |
| adventurous (contrast)  |     `[TBD-grid]`     |
| **spread**              |   **`[TBD-grid]`**   |

Per-persona mean: office `[TBD-grid]` / casual `[TBD-grid]` / party `[TBD-grid]`.

**Explanation.** Holding the RL agent fixed, we swept all agent-personality combinations to see whether the autonomous
agents' behaviour changes the recommendation quality. The spread across personalities (`[TBD-grid]`) shows the degree to
which a single agent's strategy steers the customer's review — the core multi-agent result: the agents
`[are / are not]` consequential to the outcome under the deployed Borda mechanism.

**Key code** — `multi_agent/strategies/colour.py`, `multi_agent/strategies/registry.py`
```python
# colour.py — one scorer; the score depends only on its params
if   item_color == detected:        s = p["exact"]
elif item_color in COMPATIBLE[det]: s = p["compatible"]
else:                               s = p["unrelated"]

# registry.py — a "personality" = a params set (+ veto strictness)
"purist":      {"exact":1.0, "compatible":0.65,"unrelated":0.20}  # veto 0.7
"harmonizer":  {"exact":0.85,"compatible":1.0, "unrelated":0.30}  # veto 0.3
"adventurous": {"exact":0.40,"compatible":0.70,"unrelated":1.0}   # veto none
```

---

## Slide 4 — Simulation B: RL learning (Borda, final)

| Training phase          | Mean review (1–5) | Mean return |
| ----------------------- | :---------------: | :---------: |
| Early (first 25%)       |       2.44        |    0.185    |
| Late (last 25%)         |       2.48        |    0.078    |
| **Δ (late − early)**    |     **+0.04**     | **−0.107**  |

**Explanation.** Over 300 episodes (224 policy updates, 0 dropped rewards) the RL agent **did not learn to raise
satisfaction**: review moved only +0.04 (target ≥ +0.20) and the return declined. The plumbing is sound — the limit is
the *signal*: the agent is one diluted voter in a five-agent consensus (its reward weakly reflects its own choice), part
of the reward credits *agreeing* with the others rather than pleasing the shopper, and the reviews occupy a narrow band
(a catalog/shopper ceiling) that leaves little gradient to climb. An honest negative result that points to a clear fix.

**Key code** — `multi_agent/rl/policy.py`, `multi_agent/rl/store.py`, `multi_agent/config.py`
```python
# policy.py — 9-feature per-item vector (now incl. style/occasion)
FEATURE_NAMES = ("bias","color_match","type_match","gender_match",
                 "push_norm","price_norm","stock_norm","style_match","occasion_match")

# store.py:84 — per-item 1–5 rating → reward in [-1,+1]
def rating_reward(r): return (clamp(r,1,5) - 3) / 2.0      # 1→-1 … 3→0 … 5→+1

# config.py:79 — RL is ONE fixed slice of the vote (the "dilution")
RL_WEIGHT = float(os.environ.get("RL_WEIGHT", "0.15"))
```

---

## Slide 5 — Takeaways / future work

**Explanation.** The multi-agent Borda recommender delivers the best customer satisfaction and runs end-to-end on the
physical robot. Agent personalities `[do / do not]` measurably steer recommendations (Sim A); the RL agent does not yet
learn from feedback (Sim B), limited by reward dilution and a low-variance review signal. The most promising next step is
to **promote RL from a peer voter to the orchestrating aggregator** (so the review becomes the direct, attributable
consequence of its own decision), alongside training on **real production feedback** and a coherence-correlated catalog
to widen the learnable signal.

**Key code (where the fix lands)** — `multi_agent/aggregator.py`, `multi_agent/agents/orchestrator.py`
```python
# today: RL is one weighted voter in the consensus
top_k = borda_aggregate(proposals, agent_weights, k=TOP_K)   # RL_WEIGHT = 0.15
# proposed: RL consumes the 4 agents' scores and produces the top-10 itself,
#   so the review becomes the direct, attributable reward for RL's OWN decision
```
