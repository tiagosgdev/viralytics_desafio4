# Reinforcement Learning for Multi-Agent Recommendations
## Feature Proposal

---

> **Implementation update (2026-06).** This proposal originally recommended a
> contextual-bandit approach and argued against deep RL (§2.1) on the grounds of slow
> rounds and sparse feedback. The shipped implementation instead uses **PPO** (a PyTorch
> actor-critic), per an explicit product requirement, while keeping the bandit *framing*:
> each round is treated as a **one-step** contextual-bandit episode, so PPO runs with a
> single-step return and a value baseline rather than full multi-step credit assignment.
> The reward design below (pass-rate + emoji satisfaction) is preserved as the per-item
> return. A single **global** policy is learned and **persisted across restarts**
> (network + optimizer checkpoint). See
> [`rl_implementation.md`](rl_implementation.md) for the full technical write-up.

---

## 1. Motivation

The current multi-agent recommendation system uses fixed scoring functions and static feature importances. Every round, the `ColourAgent` applies the same compatibility matrix, the `BodyAgent` uses the same adjacency scores, and the `FeatureWeightAgent` derives importances purely from the detected context — none of them learn from whether past recommendations were actually accepted by users.

The `RoundHistory` module introduced with the fault-tolerance update provides the structural foundation for learning: it already tracks every round, which agents responded, what context was detected, and what was recommended. The missing piece is **user feedback** — a signal that tells the system whether the output was good.

This proposal describes a **multi-agent contextual bandit** approach that fits the system's constraints (slow rounds, small action space, sparse feedback) and is grounded in game-theoretic principles already present in the architecture.

---

## 2. Theoretical Grounding

### 2.1 Why Not Full Deep RL

Standard deep RL approaches (PPO, SAC, DQN) are designed for fast-cycling environments where thousands of episodes can be collected cheaply. A recommendation round in this system takes 20–30 seconds end-to-end and depends on a real user scanning a physical outfit. Collecting the volume of episodes needed for a neural policy would take months of real usage before any meaningful learning occurs.

### 2.2 Multi-Agent Contextual Bandit

A **contextual bandit** is a simplified RL problem where:
- There is no sequential state (each round is independent)
- The agent observes a context, takes an action, and receives an immediate reward
- There is no long-horizon credit assignment problem

This fits the recommendation setting exactly: each round is a context (detected type, colour, body shape), the action is the scoring function output, and the reward is whether a recommended item was selected.

Making it **multi-agent** means each agent has its own bandit and receives credit specifically for its contribution to the final outcome — preserving the sealed-bid competitive structure already in place.

### 2.3 Borda Credit Assignment

The existing Borda count aggregation provides a natural credit assignment mechanism. If agent A scored item X highly, and item X ends up being selected by the user, agent A receives proportional credit based on:
- The rank it assigned to item X in its own proposal
- The item's final rank in the Borda aggregate
- Whether the item was selected or ignored

This creates a **competitive incentive**: each agent is rewarded when its domain signal contributes to user-accepted items, and penalised when it promotes items that are ignored. Over time, agents whose signals are actually predictive of user preference will receive more reward and update their parameters more aggressively.

### 2.4 Game Theory Alignment

The sealed-bid CNP structure ensures agents cannot observe each other's scores during a round. This **information asymmetry is preserved during learning** — each agent updates its own parameters based only on its own credit signal, not on what other agents scored. This is equivalent to independent Q-learning in a cooperative-competitive multi-agent game, where agents share the goal of user satisfaction but compete for influence over the final ranking.

The Nash equilibrium in this setting is a configuration where no single agent can improve its individual credit by changing its scoring parameters, given the other agents' fixed policies — meaning the system converges to a state where each agent's domain signal is calibrated to be as useful as possible within the ensemble.

---

## 3. Required Prerequisites

Before any learning can happen, two things must be built that do not currently exist.

### 3.1 User Feedback Signal

The system currently has no record of whether a recommended item was accepted. A feedback event must be captured when a user interacts with a recommendation card in the frontend. Minimum viable signals:

| Action | Interpretation |
|--------|---------------|
| User clicks/opens a recommendation card | Strong positive signal |
| User scrolls past without interacting | Weak negative signal |
| User explicitly dismisses | Strong negative signal |
| Session ends with no interaction | Neutral / ambiguous |

A `POST /api/feedback` endpoint would receive these events, link them to the originating `conv_id`, and store them in the history.

### 3.2 Feedback Linkage in History

`RoundRecord` must be extended to store feedback entries against each round. A feedback entry links a `conv_id` to an `item_id:size` key and an interaction type. The history module would expose a method to retrieve all rewarded rounds for a given agent, suitable for a batch update pass.

---

## 4. Architecture

### 4.1 Learnable Parameters Per Agent

Each scorer agent would have a small set of learnable bias parameters on top of its existing deterministic scoring function. The deterministic function provides a strong prior; the learned parameters adjust it based on accumulated experience.

**ColourAgent** — learnable score multipliers for each compatibility tier (exact match / compatible / unrelated), potentially conditioned on clothing type. The current fixed values (1.0 / 0.65 / 0.20) become starting points that drift based on which colour relationships actually drive selections.

**BodyAgent** — per-body-shape score multipliers. Some body shapes may have stronger item-type preferences than the current adjacency graph captures. The agent would learn these empirically.

**ClothingAgent** — a learnable threshold on the filter-match fraction. Currently a raw proportion; the agent could learn that in certain contexts (e.g. casual detected intent) a partial match is sufficient, while in formal contexts a stricter match is needed.

**StockAgent** — a learnable scaling factor on the inventory push signal. The current fixed 20% budget allocation interacts with this — the agent learns how much inventory pressure the user population actually tolerates before it starts feeling like "they're pushing unsold stock on me".

**FeatureWeightAgent** — learnable adjustment to the importance distribution for detected contexts. If the current rule-based `analyze_intent` consistently over-weights colour for detected `short_sleeve_top`, the agent learns a correction term.

### 4.2 Parameter Storage

Learned parameters must persist between sessions — otherwise every app restart resets the agents to their priors. A lightweight persistence mechanism (e.g. a small JSON or numpy `.npz` file per agent under `models/weights/agents/`) would be loaded at agent startup and saved after each update.

The history module already tracks agent comebacks; parameter files tie naturally into this — an agent that missed rounds loads its last saved parameters and immediately benefits from prior learning.

### 4.3 Update Rule

After a feedback event is received and linked to a `conv_id`, the orchestrator (or a background task) triggers a credit computation pass:

1. Retrieve the round record: which items were proposed by each agent, in what rank order
2. Retrieve the feedback: which items were selected / ignored
3. Compute per-agent Borda credit: reward proportional to how highly the agent ranked the selected item, penalised by how highly it ranked ignored items
4. Apply a gradient step (or bandit update) to the relevant agent's parameters using the credit signal
5. Save updated parameters to disk

The update is **offline and asynchronous** — it does not block the recommendation round and does not require agents to be retrained online. Rounds produce data; updates happen in a background pass after feedback arrives.

### 4.4 Exploration vs Exploitation

A pure greedy policy (always use the parameters that maximise expected credit) will converge prematurely and never discover better configurations. An **epsilon-greedy** strategy is the simplest solution:
- With probability ε, an agent adds small random noise to its scoring output before sending the PROPOSE (exploration)
- With probability 1−ε, it uses its current learned parameters (exploitation)
- ε decays over time as the agent accumulates more feedback and becomes more confident

For the sealed-bid structure this is particularly appropriate: since agents cannot see each other's proposals, one agent exploring does not destabilise the others.

---

## 5. Reward Function Design

The reward function is the most critical design decision. Several formulations are possible.

### Option A — Binary Selection Reward
Reward = +1.0 if agent's top-scored item was selected; 0.0 otherwise. Simple but ignores rank information.

### Option B — Rank-Weighted Reward
Reward = 1 / rank_in_agent_proposal of the selected item. An agent that ranked the selected item #1 gets full credit; #5 gets 0.2. Penalises agents that ranked good items poorly.

### Option C — Borda-Credit Reward (recommended)
Reward = (Borda points assigned to the selected item by this agent) / (max possible Borda points). This ties the reward directly to the aggregation mechanism — the agent is rewarded for the same signal that actually influences the final ranking.

### Option D — Position-in-Final-Ranking Reward
Reward = (agent's weight) × (1 / final_rank of selected item). Rewards agents whose preferred items ended up ranked highly in the aggregate. Encourages agents to align with the ensemble, not just their own domain.

**Recommended:** Option C for individual agent updates, Option D for weight recalibration of the `FeatureWeightAgent`.

---

## 6. Integration Points

| Component | Change Required |
|-----------|----------------|
| `frontend/index.html` | Capture click events on recommendation cards; send `POST /api/feedback` |
| `src/api/main.py` | Add `POST /api/feedback` endpoint; route events to history |
| `src/api/schemas.py` | Add `FeedbackRequest` schema |
| `multi_agent/history.py` | Extend `RoundRecord` with `feedback: list[FeedbackEvent]`; add `record_feedback()` method |
| `multi_agent/agents/base.py` | Add `load_params()` / `save_params()` interface |
| `multi_agent/agents/*.py` | Each scorer agent: add learnable param attributes; apply in scoring; update on feedback |
| `multi_agent/credit.py` | New module: Borda credit computation and update dispatch |
| `multi_agent/config.py` | Add `EPSILON_START`, `EPSILON_MIN`, `EPSILON_DECAY`, `LEARNING_RATE`, `MIN_FEEDBACK_FOR_UPDATE` |
| `models/weights/agents/` | New directory: persisted parameter files per agent |

---

## 7. Data Requirements and Cold Start

Learning cannot begin until sufficient feedback has been collected. Updating parameters on a single feedback event is noisy and likely to overfit to one user's interaction.

**Recommended minimum before first update:** 20–50 feedback events per agent (i.e. 20–50 round × feedback pairs where that agent responded). Below this threshold, agents should run in pure exploitation mode using their initial parameters.

**Cold start strategy:** initial parameters are set to reproduce the current deterministic scoring functions exactly (bias = 0, multiplier = 1.0). The first update pass should use a low learning rate and high regularisation to keep parameters close to the prior. As more data accumulates, the learning rate can increase and regularisation relax.

---

## 8. Evaluation

Because the system runs in a real user environment, offline evaluation is important before enabling online learning.

**Offline evaluation approach:**
1. Collect a period of feedback data with the current deterministic system (no learning)
2. Train agent parameters on a training split
3. Evaluate on a holdout split: do the learned parameters rank selected items higher than the baseline?
4. Metrics: Mean Reciprocal Rank (MRR), NDCG@10, selection rate of top-3 recommendations

**A/B test approach:**
Once offline evaluation shows improvement, enable learned parameters for a fraction of users and compare selection rates between the learned and baseline systems.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Reward hacking — agents learn to promote items that are easy to click regardless of suitability | Include a diversity penalty; monitor that agent score distributions don't collapse |
| Feedback sparsity — most sessions produce no interaction | Use scroll-past as weak negative signal; set a minimum episode count before updating |
| Catastrophic forgetting — aggressive updates erase prior structure | L2 regularisation toward initial parameters; cap maximum parameter delta per update |
| Stock agent learns to suppress inventory pressure entirely | Clamp stock agent scaling factor to [0.5, 2.0]; inventory pressure serves a business function |
| Single-user bias — one active user dominates learning | Weight updates by session diversity; limit contribution per user per time window |

---

## 10. Implementation Phases

### Phase 1 — Feedback Infrastructure
Add the frontend click tracking, `POST /api/feedback` endpoint, and history extension. No learning yet — just collecting the signal. Run for 2–4 weeks to accumulate meaningful data.

### Phase 2 — Offline Analysis
Analyse collected feedback: which agents' top-scored items most frequently match selected items? What is the baseline MRR? This establishes whether the signal is strong enough to learn from.

### Phase 3 — Credit Module and Parameter Persistence
Implement `multi_agent/credit.py`, parameter load/save in `BaseRecommenderAgent`, and offline batch update. Validate against holdout feedback data before enabling online updates.

### Phase 4 — Online Learning with Exploration
Enable epsilon-greedy exploration in scorer agents. Monitor divergence from baseline. Gradually reduce epsilon as confidence grows.

### Phase 5 — FeatureWeightAgent Adaptation
Once individual agent parameters are stable, extend learning to the weight distribution itself — allowing the system to adapt how much it trusts each domain signal for different user contexts.
