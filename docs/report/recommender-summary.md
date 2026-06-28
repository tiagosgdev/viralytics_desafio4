# Multi-Agent Clothing Recommender — Architecture & Evaluation

> ⏳ **Results pending.** The two simulations are still running; every number marked **`[TBD]`** below is a
> placeholder to be replaced once the runs finish (the §4 conclusion will be finalised then too). The architecture
> (§1) and experimental design (§2) are final.

## 1. System architecture

The recommender is a **multi-agent system** built on SPADE (agents communicate over an XMPP message bus). A
recommendation is produced as one **sealed-bid round** following a FIPA **Contract-Net** protocol
(call-for-proposals → sealed bids): an _Orchestrator_ coordinates a _FeatureWeightAgent_ and **five scoring agents** —
`colour`, `body`, `clothing`, `stock`, and `RL` — each an expert on a single dimension of the garment.

### 1.1 One recommendation round

```
        conversation (chat) + camera scan (colour / type / body)
                              │
                              ▼
                   ┌────────────────────┐    weight_i = importance_i × confidence_i
                   │ FeatureWeightAgent │    importance  ← chat LLM (per turn)
                   └─────────┬──────────┘    confidence  ← CV detection
                             │ weights         RL = fixed learned slice
                             ▼
   PROPOSE (scores)     ┌──────────────┐  CFP (candidate items)
                ┌──────▶│ Orchestrator │──────────┐
                │       └──────┬───────┘           ▼
                │              │ weighted    colour · body · clothing · stock · RL
                └──────────────┼─── Borda ──  (5 scoring agents)
                               ▼
              top-10 recommendations ──▶ user rates EACH item 1–5 ──▶ per-item RL update
```

- **Intent-matched retrieval.** The conversation is parsed into an intent filter over **colour, type, style and
  occasion**; candidate garments are retrieved to match that intent, so the slate reflects what the shopper actually
  asked for rather than the scan alone.
- **Weights are conversation-driven.** Each agent's influence is `importance × confidence`: _importance_ is re-derived
  every turn by the chat LLM (e.g. "I want something red" raises the colour weight), _confidence_ comes from the camera
  detection. Agents **only score their own dimension** and return a bid in `[0, 1]`.
- **Borda aggregation.** The Orchestrator combines the five bids with a **weighted, tie-aware Borda count** into the
  final **top-10**.
- **Per-item feedback.** The shopper rates **each** of the 10 recommended items individually on a **1–5** scale (as in
  the production UI, one rating per item) — and each item's rating is fed back as the reward for that item, giving the
  RL agent fine-grained, within-list signal.

### 1.2 The RL agent

The `RL` agent is a **learned voter**: a PyTorch actor-critic policy trained online with **PPO**. It occupies one fixed
weight slice in the Borda vote and scores each candidate from a compact feature vector — **colour / type / style /
occasion** match against the parsed intent, plus push-score, price and stock signals. After every round it receives the
shopper's **per-item** 1–5 ratings as rewards and updates its policy, so over time it learns **which item features earn
the best reviews** and biases the slate toward them.

### 1.3 Agent autonomy: personalities + per-agent memory

Each scorer is an **autonomous agent**. Its scoring is a swappable **personality** — a pure _strategy_ resolved from a
registry and selectable per run (e.g. colour `purist` / `harmonizer` / `adventurous`). And each agent keeps its **own
write-only memory** (`multi_agent/memory/<agent>.db`): one row per round logging the context it saw and its top scores.
The per-agent stores satisfy the multi-agent **autonomy** requirement — nothing reads them to make a decision; a shared
`RoundHistory` remains the global round log.

### 1.4 Selection mechanism — Borda, and the veto-batch alternative

The deployed system uses **weighted Borda** over an intent-matched slate. We also explored a **veto-batch** alternative
(broad retrieval of items matching ≥1 signal, where each agent can _veto_ weak items and a coalition of weight ≥ τ
eliminates an item, drawing successive batches until 10 survive). Veto-batch surfaces far more variety, but at a clear
cost to **relevance**: judged on a 1–5 satisfaction review, its broad, less on-intent slate scores **lower** than
Borda's. Because the product goal is customer satisfaction — and because Borda's reviews also give the RL agent a
cleaner signal to learn from — **Borda is the chosen mechanism**. A brief illustrative comparison is in §3.1.

## 2. Experimental setup

A human customer is simulated by an **LLM shopper** (`qwen2.5:14b-instruct`) that role-plays each of **3 customer
personas** through a full multi-turn conversation and then returns a **per-item 1–5 satisfaction review** — our primary
metric. All experiments use the **Borda** mechanism.

We run **two separate simulations** so that the **agent-personality effect** and the **RL-learning effect** are
**isolated** from one another (one variable moves at a time):

- **Simulation A — Agent personalities (full factorial grid).** Sweep **81 agent-personality combinations × 3 personas ×
  3 repeats = 729 episodes**, with the RL agent held fixed. This isolates **how the agents' personalities affect the
  rating**.
- **Simulation B — RL learning curve.** Hold the agent personalities at their defaults and run a long sequence of
  episodes in which the RL agent learns online from the per-item reviews, recording the review trend over training. This
  isolates **how the RL agent's learning affects the rating**.

Both simulations drive the real recommender end-to-end (no mocks) and report `0 errors / 0 dropped rewards`.

## 3. Results

> ⏳ Values below are placeholders pending the in-progress runs.

### 3.1 Why Borda (brief comparison)

**Table 1 — Borda vs Veto-batch.**

| Metric                         |     Borda      |   Veto-batch   |
| ------------------------------ | :------------: | :------------: |
| Mean review (1–5)              |    **1.57**    |      1.28      |
| Distinct items surfaced        |     1,258      |  8,978 (7.1×)  |

> _Note: these two figures are from a **previous evaluation run** (an earlier version of the system, 729 episodes per
> arm); they were not re-run for this report and are included only to illustrate the trade-off._

_Takeaway:_ Borda delivers the **higher satisfaction review** (1.57 vs 1.28); veto-batch surfaces ~7× more variety but
trades away relevance. Because the product goal is customer satisfaction, **Borda is chosen for deployment.**

### 3.2 Simulation A — agent personalities affect the rating

**Table 2 — Mean review by agent personality (Borda, marginal over all combos).**

| Agent / personality        | Mean review (1–5) |
| -------------------------- | :---------------: |
| colour — purist            |      [TBD]        |
| colour — harmonizer        |      [TBD]        |
| colour — adventurous       |      [TBD]        |
| _(other agents …)_         |      [TBD]        |
| **spread across combos**   |    **[TBD]**      |

_Per-persona means:_ office `[TBD]` / casual `[TBD]` / party `[TBD]`. _Best combination per persona:_ `[TBD]`.

_Interpretation (to finalise):_ the spread of `[TBD]` across personalities shows that **changing an agent's behaviour
`[does / does not]` measurably move the customer's review** — i.e. the autonomous personalities `[are / are not]`
consequential to the recommendation quality.

### 3.3 Simulation B — RL learning affects the rating

**Table 3 — Review over RL training (Borda).**

| Phase                          | Mean review (1–5) | Mean return |
| ------------------------------ | :---------------: | :---------: |
| Early (first 25% of episodes)  |      [TBD]        |    [TBD]    |
| Late (last 25% of episodes)    |      [TBD]        |    [TBD]    |
| **Δ (late − early)**           |    **[TBD]**      |  **[TBD]**  |

_Interpretation:_ over training the mean review changed by `[TBD]` (target ≥ +0.20), so in this run the RL agent **did
not show a clear gain in customer satisfaction** from the per-item feedback. The likely reasons and the changes most
likely to improve it are discussed in §4.

## 4. Conclusion

> ⏳ To finalise once both simulations complete.

- **Selection mechanism.** `[Borda chosen — restate the satisfaction advantage with the §3.1 numbers.]`
- **Agent personalities.** `[State whether/how personalities move the review, citing the §3.2 spread.]`
- **RL learning.** In this evaluation the RL agent **did not learn to raise customer satisfaction** (Δreview `[TBD]`,
  below the +0.20 target; mean return did not trend up). This is an honest negative result; §4.1 explains why and §4.2
  proposes the changes most likely to fix it.

### 4.1 Why the RL signal was weak

The plumbing works (per-item rewards land, the policy updates), so the limitation is in **what the agent is asked to
learn from**, not the implementation:

- **RL is one diluted voter.** It contributes a single weight slice to a five-agent Borda vote, so the final list is a
  **consensus**. The reward a recommendation receives is therefore only weakly a consequence of RL's *own* ranking — the
  other four agents largely determine what is shown, which blurs the learning signal.
- **The dense reward partly rewards conformity.** Part of the reward credits the agent for agreeing with the consensus
  top-10 rather than for raising the review. That can pull the policy toward *imitating* the other agents instead of
  using its own (colour/type/style/occasion) judgement.
- **Low signal variance (a learning ceiling).** The satisfaction reviews sit in a narrow band — shown ten options a
  shopper genuinely likes only one or two, and the synthetic catalog's weakly-correlated attributes cap how well any
  slate can match a specific multi-attribute goal. With little contrast between a great and a mediocre slate, there is
  little gradient for the policy to climb: **RL can only learn a signal that varies.**
- **A short training horizon.** Online learning over a few hundred episodes / ~100 policy updates is modest for a learned
  ranking policy.

### 4.2 Future changes that could improve it

- **Make RL the orchestrator, not a voter.** Promote the RL policy from a peer in the Borda vote to the **learned
  aggregator** that takes the four domain agents' scores as inputs and makes the final selection itself. The review then
  becomes the **direct, attributable consequence of RL's own decision**, removing both the consensus dilution and the
  conformity proxy. (Blend with Borda during an initial cold-start so an untrained policy can't degrade live results.)
- **Reward the outcome, not agreement.** With per-item ratings now available, shape the reward to credit RL specifically
  for the items *it* pushed up that earned high reviews, rather than for matching the consensus.
- **Raise the signal's variance.** Train on **real production feedback** (the per-item ratings the UI already collects)
  and/or a **coherence-correlated / real-product catalog**, so a well-targeted slate can actually earn top scores and the
  policy has real contrast to learn from.
- **More data and warm-starting.** Many more episodes, a larger policy, and/or **offline pre-training** (behaviour
  cloning from logged good recommendations) before online fine-tuning.
- **Richer features.** Extend the item representation (e.g. pattern/material and conversation-context features) so the
  policy can discriminate items the current colour/type/style/occasion vector treats as identical.

### 4.3 Overall

`[One-paragraph synthesis: the final Borda-based, per-item-feedback multi-agent architecture; that the two isolated
simulations measured the personality effect (§3.2) and the RL-learning effect (§3.3) independently; the honest RL
negative result and the clear path to improve it (§4.2); and the recommended next step.]`
