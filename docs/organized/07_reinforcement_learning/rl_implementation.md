# Reinforcement Learning — Implementation Guide (PPO)

> How the `RLRecommenderAgent` works, written as an RL student would explain it.
> Companion to [`rl_proposal.md`](rl_proposal.md). This document describes what was
> actually built: a **PyTorch actor-critic trained with Proximal Policy Optimisation**.

---

## 1. The one-paragraph summary

A fifth recommendation agent (`RLRecommenderAgent`) joins the sealed-bid round. Like
the other scorers it receives the 40 candidates and submits scores that feed the Borda
count — but its scores come from a **neural policy** trained with **PPO**. After each
round it learns from whether its picks survived (a **pass-rate** reward) and from the
**1–5 emoji ratings** users give items. There is **one global policy** shared by all
customers; it is checkpointed to disk (network **and** optimizer state) and reloaded on
startup, so the learned pattern is never lost when the app restarts.

---

## 2. Formulating recommendation as an RL problem

To use PPO we first have to say what the **state**, **action**, **reward** and **policy**
are. RL normally assumes a *Markov Decision Process* (MDP): the agent observes a state
`s_t`, takes an action `a_t`, receives a reward `r_t`, and transitions to `s_{t+1}`, and
it wants to maximise the discounted return `G_t = Σ γ^k r_{t+k}`.

Our setting is a special, simpler case of an MDP — a **contextual bandit** (a "one-step"
MDP):

| MDP concept | Here |
|-------------|------|
| **State** `s` | the features of a candidate item in the current round's context |
| **Action** `a` | a real number `z` (a *score-logit*); the proposed score is `sigmoid(z)` |
| **Policy** `π_θ(a\|s)` | a Gaussian: `z ~ N(μ_θ(s), σ)` |
| **Reward** `r` | pass-rate (round outcome) + emoji satisfaction |
| **Next state** | *there is none* — each round is one independent step |

Because every episode is a single step, there is **no temporal credit assignment** and
the discount factor `γ` is irrelevant (equivalently `γ = 0`). The return is simply the
immediate reward: `G = r`. This matters because it simplifies PPO a lot — the
"advantage" reduces to `reward − value` (Section 5).

> **Why this is still legitimate PPO.** PPO is defined for general MDPs; a one-step MDP
> is just the degenerate case. We keep the full machinery (a stochastic policy, a value
> baseline, the clipped objective, multiple epochs over an on-policy rollout); we only
> drop the multi-step return because our problem genuinely has one step.

The policy acts **per item** (it factorises over candidates): each candidate is scored
independently from its own feature vector. This is standard in learning-to-rank with
policy gradients and keeps the action space tiny and the interface identical to the other
scorer agents.

---

## 3. The actor-critic network

File: [`multi_agent/rl/policy.py`](../../../multi_agent/rl/policy.py)

PPO is an **actor-critic** method: it trains two things at once.

- **Actor** `π_θ` — *the policy*. Decides the action. Here it outputs the mean `μ_θ(x)`
  of a Gaussian over the score-logit. A separate learnable parameter `log σ` gives the
  spread. We act by sampling `z ~ N(μ, σ)` and proposing `s = sigmoid(z)`.
- **Critic** `V_φ` — *the value function*. Estimates the expected reward for an item,
  `V_φ(x) ≈ E[reward]`. It is **not** used to pick actions; it is a **baseline** that
  reduces the variance of the policy-gradient estimate (Section 5).

Both share a small trunk (parameter sharing is standard and data-efficient):

```
x (7 features) ─► Linear(7,32) ─► tanh ─► Linear(32,32) ─► tanh ─┬─► Linear(32,1) = μ(x)   (actor)
                                                                 └─► Linear(32,1) = V(x)   (critic)
log σ : a single learnable parameter (state-independent), σ = exp(log σ)
```

### 3.1 Input features

Only information available during a **sealed** bid is used (CFP payload + context), so the
agent never sees the other agents' scores:

| feature | meaning |
|---|---|
| `bias` | constant 1.0 |
| `color_match` | 1 if item colour == detected colour |
| `type_match` | 1 if item type == detected type |
| `gender_match` | 1 if item gender ∈ {user gender, `unisex`} |
| `push_norm` | inventory push_score, min-max normalised across the 40 candidates |
| `price_norm` | price, normalised across the 40 candidates |
| `stock_norm` | stock_count, normalised across the 40 candidates |

Built by `extract_features(candidates, context)`. Inputs are already roughly in `[0,1]`,
which keeps the network well-conditioned without extra observation normalisation.

---

## 4. The two reward signals

File: [`multi_agent/rl/store.py`](../../../multi_agent/rl/store.py)

The reward is what defines *what we want*. Both signals from the original design are kept;
they are summed onto each item's transition to form its scalar return `R`.

### 4.1 Pass-rate (automatic, every round)

The agent's **chosen set** = its top-10 items by proposed score. Let `p = passed / 10`
be the fraction that reached the final top-10. `passrate_reward(passed, 10)`:

```
passed == 0   →  -1.5                  # 0% advanced → bigger negative
p >= 0.10     →  (p - 0.10) / 0.90     #  0 .. +1
0 < p < 0.10  →  (p - 0.10) / 0.10     # -1 ..  0
```

So 1 item (10%) → 0 (neutral), 10 → +1, 0 → −1.5. This reward is added to **each chosen
item's** transition (episodic credit assignment — the items the agent committed to share
the outcome of the bid).

### 4.2 Emoji satisfaction (delayed, per item)

`rating_reward(rating) = (rating − 3) / 2`, so 😍(5) → +1, 🙂(4) → +0.5, 😐(3) → 0,
😕(2) → −0.5, 😣(1) → −1. Added to that specific item's transition when the rating
arrives.

Final per-item return: `R = r_passrate + r_emoji` (each term 0 when not applicable).

---

## 5. Advantage and the value baseline

The vanilla policy-gradient update is

```
∇θ J = E[ ∇θ log π_θ(a|s) · G ]
```

i.e. *"make actions that led to high return more likely."* Using the raw return `G` as the
multiplier is unbiased but **high variance**. The standard fix is to subtract a baseline
`b(s)` that doesn't depend on the action — the **advantage**:

```
A(s,a) = G − b(s)
```

The critic provides the baseline, `b(s) = V_φ(s)`. Because our episodes are one step, the
general *Generalised Advantage Estimation* (GAE) formula collapses to simply:

```
A = R − V_φ(x)
```

Intuitively: *did this item do better (A > 0) or worse (A < 0) than the critic expected?*
Better-than-expected actions get reinforced; worse-than-expected get suppressed. We also
**normalise** advantages across the rollout (zero mean, unit std) — a common PPO trick
that stabilises the step size.

---

## 6. Why PPO, and the clipped objective

A plain policy gradient takes one step per batch and is fragile: too large a step can
collapse the policy. We'd like to **reuse each batch for several gradient steps** without
moving the policy too far from the one that generated the data. That is exactly what PPO's
**clipped surrogate objective** does.

Define the **probability ratio** between the policy being optimised and the policy that
collected the data (the *behaviour* policy):

```
r(θ) = π_θ(a|s) / π_θ_old(a|s) = exp( log π_θ(a|s) − log π_θ_old(a|s) )
```

PPO maximises

```
L_clip(θ) = E[ min( r(θ)·A ,  clip(r(θ), 1−ε, 1+ε)·A ) ]
```

The `clip` caps the ratio inside `[1−ε, 1+ε]` (ε = `PPO_CLIP_EPS` = 0.2). The `min` makes
the bound **pessimistic**: if an action was good (`A > 0`) the objective stops rewarding it
once `r > 1+ε`, so the policy can't lurch toward it; if it was bad (`A < 0`) it stops
beyond `r < 1−ε`. This keeps each update inside a *trust region* even though we take
several gradient steps on the same data — the key reason PPO is stable and sample-efficient
compared to vanilla policy gradients.

The full loss adds a **value loss** (train the critic toward the observed returns) and an
**entropy bonus** (keep the policy stochastic → keep exploring):

```
loss = −L_clip  +  c_v · E[(V_φ(x) − R)²]  −  c_e · H[π_θ]
```

with `c_v = PPO_VALUE_COEF = 0.5`, `c_e = PPO_ENTROPY_COEF = 0.01`, and `H` the Gaussian's
entropy. Minimising `loss` ⇔ maximising `L_clip`, fitting the critic, and keeping entropy
up.

### 6.1 The update, in pseudocode

`PPOPolicy.learn(rollout)`:

```
returns  = [ t.reward            for t in rollout ]
advantages = returns − V_old ; normalise(advantages)
for epoch in 1..PPO_EPOCHS (=4):
    for each minibatch (size PPO_MINIBATCH=64):
        μ, V        = net(x)
        logπ_new    = Normal(μ, σ).log_prob(action)
        ratio       = exp(logπ_new − logπ_old)
        L_clip      = mean( min(ratio·A, clip(ratio,1±ε)·A) )
        L_value     = mean( (V − returns)² )
        H           = mean( Normal(μ, σ).entropy() )
        loss        = −L_clip + c_v·L_value − c_e·H
        Adam.step( ∇ loss )            # with global grad-norm clip 0.5
update_count += 1 ; save_checkpoint()
```

---

## 7. On-policy rollouts and the data flow

PPO is **on-policy**: the ratio `r(θ)` only makes sense if the data was collected by
(a recent version of) the policy being updated. So we **freeze the policy, collect a few
rounds, then update once and discard the batch**. Concretely:

- `RLScoreBehaviour` (on a CFP): `act()` → sample scores, build transitions, cache them in
  `rl_store`, send the PROPOSE. The policy does **not** change here.
- `RoundResultBehaviour` (on the orchestrator's end-of-round INFORM): `settle_round()`
  applies the pass-rate reward. After **`PPO_ROLLOUT_ROUNDS` (=8)** rounds have settled,
  the store hands back the whole batch and the agent calls `policy.learn(batch)` → one PPO
  update → checkpoint. Between updates the policy is frozen ⇒ the batch is on-policy. ✔

```
CFP ─► RLScoreBehaviour
         extract_features(); act() → sample z, score=sigmoid(z), logπ_old, V
         rl_store.record_transitions(conv_id, transitions)
         PROPOSE scores ─► Orchestrator → Borda → final top-10
Orchestrator ─► round_result INFORM ─► RoundResultBehaviour
         rl_store.settle_round(): add pass-rate reward to chosen items
         every 8th settled round → batch → policy.learn() → rl_ppo.pt
```

Emoji feedback (the delayed signal) takes a separate path but lands in the **same** buffered
transition:

```
Frontend emoji click ─► POST /api/feedback {round_id, item_id, size, rating}
   → RecommendationSystem.submit_feedback()
       rl_store.add_reward(round_id, "item_id:size", rating_reward(rating))
```

`submit_feedback` only **mutates a reward float** (thread-safe); it never runs a torch op,
so all backprop stays on the agent's single event-loop thread. The reward is incorporated
whenever the round's batch is next consumed by `learn()`.

### 7.1 The honest caveat: delayed feedback vs on-policy

There is a real tension worth stating like an RL student would. PPO wants fresh on-policy
data, but emoji ratings arrive *after* the round. We handle it pragmatically:

- A round's transitions stay editable until its rollout batch is consumed (up to 8 rounds
  later), so ratings that arrive within that window **are** included on-policy.
- A rating that arrives after the batch was already used is **dropped** (logged), because
  applying it would be off-policy and break the ratio assumption. Given rounds are slow and
  users rate quickly, this is rare.

A stricter alternative (future work) is an off-policy correction or a separate replay
buffer for late ratings; we deliberately kept the first version simple and on-policy-clean.

---

## 8. Exploration

No ε-greedy. Exploration is intrinsic to a **stochastic policy**: we *sample* `z ~ N(μ, σ)`
rather than always taking the mean, so the agent naturally tries scores around its current
estimate. The **entropy bonus** (`c_e`) stops `σ` from collapsing to 0 too quickly, keeping
exploration alive while the policy is still uncertain. `σ` is reported in every update log
and in `snapshot()`.

(For evaluation you can call `act(deterministic=True)` to use the mean and silence
exploration — the tests use this.)

---

## 9. Persistence — one global policy that survives restarts

This was a hard requirement: **the learned pattern must not reset when the app restarts.**

- There is exactly **one** `PPOPolicy` (a module-level singleton), shared by every customer
  and every round. It learns a *single global pattern*, not a per-user model.
- After every PPO update it is checkpointed to
  `models/weights/agents/rl_ppo.pt` via `torch.save`, storing **both** the network
  `state_dict` **and** the Adam optimizer `state_dict` (so momentum/variance estimates also
  resume — important for smooth continued training), plus `update_count` and the feature
  schema.
- On startup `PPOPolicy.__init__` calls `load()`: if the checkpoint exists and its feature
  schema matches the code, the network + optimizer are restored and training continues from
  where it left off. If the schema changed, it starts fresh instead of crashing.

So closing and reopening the app resumes the same policy — no re-learning from scratch.

---

## 10. How it counts in the final ranking

Unchanged from the bandit version: the RL agent gets a **fixed Borda slice**
`RL_WEIGHT = 0.15`, carved out alongside the stock agent's `0.20`
(`build_agent_weights` in [`aggregator.py`](../../../multi_agent/aggregator.py)); the rest
is split among body/clothing/colour by feature importance. If the RL agent doesn't respond
in a round, its weight is redistributed; with `RL_ENABLED = False` it isn't started and the
legacy 4-agent split is reproduced exactly.

---

## 11. Files and hyperparameters

**Code**

| File | Role |
|------|------|
| [`multi_agent/rl/policy.py`](../../../multi_agent/rl/policy.py) | `ActorCritic` network + `PPOPolicy` (act, learn, checkpoint). |
| [`multi_agent/rl/store.py`](../../../multi_agent/rl/store.py)   | `Transition`, the on-policy rollout buffer, the two reward functions. |
| [`multi_agent/agents/rl_agent.py`](../../../multi_agent/agents/rl_agent.py) | the SPADE agent: act-on-CFP, learn-on-round-result. |
| `run.py` / `main.py` / `schemas.py` | `submit_feedback`, `POST /api/feedback`, `round_id` plumbing. |
| `frontend/index.html` | the 😣😕😐🙂😍 rating row (unchanged by the PPO switch). |
| [`tests/test_rl_policy.py`](../../../tests/test_rl_policy.py) | rewards, acting, a real "does it learn?" test, checkpoint round-trip. |

**Hyperparameters** ([`config.py`](../../../multi_agent/config.py))

| Constant | Default | Meaning |
|----------|--------:|---------|
| `PPO_HIDDEN` | 32 | trunk width |
| `PPO_LR` | 3e-4 | Adam learning rate |
| `PPO_CLIP_EPS` | 0.2 | clip width ε (trust region) |
| `PPO_EPOCHS` | 4 | optimisation passes per rollout |
| `PPO_MINIBATCH` | 64 | minibatch size |
| `PPO_VALUE_COEF` | 0.5 | critic-loss weight `c_v` |
| `PPO_ENTROPY_COEF` | 0.01 | entropy-bonus weight `c_e` |
| `PPO_MAX_GRAD_NORM` | 0.5 | gradient-norm clip |
| `PPO_INIT_LOG_STD` | −0.5 | initial `log σ` (σ ≈ 0.61) |
| `PPO_ROLLOUT_ROUNDS` | 8 | rounds collected per on-policy update |
| `PASSRATE_ANCHOR` / `ZERO_PASS_PENALTY` | 0.10 / −1.5 | pass-rate reward shape |
| `RL_WEIGHT` / `RL_ENABLED` | 0.15 / True | Borda slice / master switch |
| `RL_CHECKPOINT_PATH` | `models/weights/agents/rl_ppo.pt` | the persisted policy |

---

## 12. How to verify

**Unit tests (no broker):**

```bash
python -m pytest tests/test_rl_policy.py -q
```

Covers the reward functions, that `act()` returns valid scores/transitions, that a
deterministic pass is reproducible, that `learn()` runs and increments `update_count`, that
**the policy actually learns** to prefer a consistently-rewarded feature pattern, and that a
checkpoint round-trips (weights survive a reload).

**Live, end-to-end:**

```bash
docker compose up -d xmpp
uvicorn src.api.main:app --port 8000
```

`POST /api/recommend` → response has a `round_id` and an `rl` entry in `agent_scores`; rate
items via `POST /api/feedback`; after `PPO_ROLLOUT_ROUNDS` rounds a PPO update fires (watch
the `PPO update #N` log line) and `rl_ppo.pt` is written. Restart the app and confirm
`update_count` resumes from the checkpoint rather than 0.

---

## 13. Limitations and future work

- **One-step framing.** We model each round independently. If you later want the agent to
  reason across a *session* (state carries between rounds), you'd reintroduce `γ` and proper
  GAE — the network and PPO loop already support it.
- **On-policy late feedback** is dropped once a batch is consumed (Section 7.1). An
  off-policy correction or a dedicated late-rating replay would capture more signal.
- **Shared trunk, fixed σ schedule.** Separate actor/critic networks and/or a learned
  state-dependent σ can help if learning plateaus.
- **Reward scale.** Pass-rate ∈ [−1.5, 1] and emoji ∈ [−1, 1] are summed 1:1; reward
  normalisation or per-source weights are easy knobs if one signal dominates.
- **Features.** `body_type` is still omitted (needs a `clothing.db` lookup like
  `BodyRecommenderAgent`); richer/per-context features are the most promising next step.
```
