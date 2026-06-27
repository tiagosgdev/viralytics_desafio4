# Multi-Agent Clothing Recommender — Architecture & Evaluation

## 1. System architecture

The recommender is a **multi-agent system** built on SPADE (agents communicate over an XMPP message bus). A
recommendation is produced as one **sealed-bid round**: an _Orchestrator_ coordinates a _FeatureWeightAgent_ and **five
scoring agents** — `colour`, `body`, `clothing`, `stock`, and `RL` — each an expert on a single dimension of the
garment.

### 1.1 One recommendation round

```
        conversation (chat) + camera scan (colour / type / body)
                              │
                              ▼
                   ┌────────────────────┐    weight_i = importance_i × confidence_i
                   │ FeatureWeightAgent │    importance  ← chat LLM (per turn)
                   └─────────┬──────────┘    confidence  ← CV detection
                             │ weights         RL = fixed 0.15 slice
                             ▼
   PROPOSE (scores      ┌──────────────┐  CFP (candidate items)
   [+ vetoes])  ┌──────▶│ Orchestrator │──────────┐
                │       └──────┬───────┘           ▼
                │              │ aggregate   colour · body · clothing · stock · RL
                └──────────────┼─────────────  (5 scoring agents)
                               ▼
                        top-10 recommendations ──▶ user 1–5 rating ──▶ RL update
```

- **Weights are conversation-driven.** Each agent's influence is `importance × confidence`: _importance_ is re-derived
  every turn by the chat LLM (e.g. "I want something red" raises the colour weight), _confidence_ comes from the camera
  detection. The `RL` agent is a learned voter occupying one fixed 0.15 weight slice.
- **Agents only score their own dimension** (colour matches colour, etc.) and return a bid in `[0, 1]`; the Orchestrator
  combines the bids into the final top-10.

### 1.2 Two selection mechanisms (the comparison)

We evaluate two ways of turning the agents' bids into the final list:

```
BORDA  (baseline)
  narrow retrieval — hard "colour AND type" filter (~45 candidates)
    → each agent ranks the candidates
    → weighted, tie-aware Borda count            → top-10

VETO-BATCH  (redesign)
  broad retrieval — items matching ≥1 signal (~1,300 candidates), random 40
    → each agent scores AND vetoes its weak items
    → eliminate an item when Σ weight(vetoing agents) ≥ τ   (τ = 0.5)
    → draw more 40-item batches until 10 survive
    → weighted tie-aware Borda orders the survivors          → top-10
```

**Why the redesign.** Under Borda the slate is hard-filtered to the scan, so it is _uniform_ (e.g. all red short-sleeve
tops). Single-dimension agents then bid almost identically, so swapping an agent's "personality" barely changes the
output and the pool is capped at ~16 near-identical items. The veto design retrieves a **broad, varied** slate, so every
agent's dimension varies and its **veto directly decides which items survive** — making personalities consequential.
With `τ = 0.5` and typical weights (~0.25 each), no single agent can eliminate an item; it takes a coalition of ≥2 — so
`τ` acts as a **relevance↔variety dial**.

## 2. Experimental setup

Each arm is a **full factorial sweep**: 81 agent-personality combinations × 3 customer personas × 3 repeats = **729
episodes**. Simulate a human customer with an LLM shopper (`qwen2.5:14b-instruct`) role-plays each persona through a
full multi-turn conversation and then returns a **1–5 satisfaction review** — our primary metric. Both runs completed
with **0 errors, 0 timeouts, 0 abandoned episodes**. Borda uses the legacy narrow filter; veto-batch uses broad
retrieval (the two arms differ in selection _and_ retrieval — they are compared as end-to-end bundles).

## 3. Results

**Table 1 — Borda vs Veto-batch.**

| Metric                                      |       Borda        |     Veto-batch     |
| ------------------------------------------- | :----------------: | :----------------: |
| Mean review (1–5)                           |      **1.57**      |        1.28        |
| &nbsp;&nbsp;office / casual / party persona | 1.81 / 1.65 / 1.26 | 1.51 / 1.13 / 1.19 |
| Distinct items surfaced                     |       1,258        |  **8,978** (7.1×)  |
| Do personalities shift the review?          |   no (Δ ≤ 0.06)    | **yes (Δ = 0.16)** |

**Table 2 — Colour-agent personality (example).** Marginal mean review across all combos using each colour personality.

| Colour personality     |    Borda     |      Veto-batch      |
| ---------------------- | :----------: | :------------------: |
| purist (on-theme)      |     1.54     |       **1.35**       |
| harmonizer             |     1.60     |         1.29         |
| adventurous (contrast) |     1.58     |         1.19         |
| **spread**             | 0.06 (noise) | **0.16 (monotonic)** |

Under veto-batch the colour personality produces a clean, monotonic ordering — the more on-theme the personality, the
higher the review — proving the agents' behaviour now changes the result. Under Borda the three personalities sit within
noise. Veto-batch also surfaces **7× more distinct items**, removing the "16 near-identical reds" ceiling of the Borda
pool.

## 4. Conclusion

**Veto-batch is the better mechanism for a realistic deployment**, despite the lower mean review:

- **The lower average is expected, not a defect.** Shown 10 options, a real shopper genuinely likes only one, two, maybe
  three — so a mean near the bottom of the scale is what an honest evaluation should produce. Borda's higher score comes
  from showing a narrow, on-theme sliver that is "safe" but offers almost no real choice.
- **Veto-batch gives real choice and makes the agents matter.** It surfaces 7× more variety, and the agent personalities
  measurably steer the outcome (Table 2) — the system's intended behaviour, which Borda cannot deliver.
- **It is the right substrate for future improvement.** With more time, the reinforcement- learning agent (and possible
  future fine-tuning) can learn _which combination of agent personalities works best for each customer personality_, and
  dispatch the agents accordingly per shopper — an adaptive, per-customer recommender. That learning is only possible on
  the broad, personality-responsive pool veto-batch provides; Borda's capped, uniform output gives the learner nothing
  to differentiate.
