# RL Learning Fix — Staged Implementation Plan

> **Status:** PLAN ONLY. No production code is changed by this document.
> **Branch context:** `rl-learning-curve`.
> **Diagnosis it acts on:** the prior audit's conclusion — *design flaw, not a bug.*
> The PPO plumbing is correct (`rewards_dropped=0` across exp #12/#22/#23/#24); the
> agent cannot learn because **the reward is not a learnable function of the
> state/action the RL agent controls.** This plan fixes that, in impact-ranked stages.

---

## 0. Root-cause recap (with code anchors)

| # | Severity | Root cause | Evidence |
|---|----------|-----------|----------|
| 1 | fatal | RL is **blind to the reward-driving axes** (style/occasion). Orchestrator ships RL only 8 fields; the feature vector is 7 generic features; match flags compare to the **raw scan**, not the refined `include`. | `orchestrator.py:94`; `policy.py:58-66`, `:89-115` (`:94-95`) |
| 2 | fatal | In `rating` mode the reward is **not attributable to RL's own picks** — pass-rate is zeroed and the episode rating is applied to the Borda top-K (all 5 agents, RL only 0.15 weight). | `store.py:169`; `run_experiment.py:221-238` |
| 3 | major | **Tiny leverage:** `RL_WEIGHT=0.15`; the other 4 agents share 0.85. | `config.py:79` |
| 4 | major | **Floor-bound, near-constant reward** → low contrast / nothing to fit. | empirical, all 4 curves flat |
| 5 | minor bug | Rollout **diluted by ~80–90% zero-reward transitions** (every candidate enters the PPO batch; only ~10 final items carry a reward). | `store.py:190`, `:209` |
| 6 | minor | **Advantage normalization amplifies noise** when reward is near-constant. | `policy.py:210-211` |

**Empirical baseline to beat:** mean_return flat ≈ −0.016 (veto) / −0.005 (borda);
within-persona review slopes flat or slightly negative; `rewards_dropped=0`.

**Key just-shipped enabler:** the parser (`LNIAGIA/query_parsing/feature_weighting.py`,
`analyze_intent`, prompt at `:147`, `:165-176`, example `:276-282`) now populates
`include.style` and `include.occasion`. So `weights_result["filters"]["include"]` is now
a usable source of style/occasion (and refined color/type) match flags — exactly what the
reviewer rates on.

---

## 1. Goal & success criterion

**Goal:** make the RL policy *learn* — i.e. its updates move the reward signal in the
direction of higher reviews, and that shows up as a measurable upward trend over a curve
run.

**"It learns" is declared when, on an isolated curve run (Stage-gated, see §3):**

1. **Primary — within-run review trend:** the early(first 25%)-vs-late(last 25%) mean
   `final_review` delta (already computed by `plot_learning_curve`) is **positive and
   larger than the noise band** seen in the flat baselines (baselines sit at Δ≈−0.05..0).
   Target: **Δreview ≥ +0.20** sustained across ≥2 seeds.
2. **Secondary — return trend:** `curve_points.mean_return` trends **up** vs cumulative
   `update_count` (linear-fit slope > 0), instead of the flat ≈−0.005..−0.016 baseline.
3. **Mechanism check (necessary, not sufficient):** with reward-relevant features in
   place, the policy's `value_loss` should fall and `sigma` (`policy.snapshot()`) should
   contract over the run (the policy is committing), and RL's *own* per-item scores should
   correlate with style/occasion match — verifiable by logging.

**Measurement instruments (already exist):**
- `curve_points` table (episode_index, update_count, rating, rewards_landed,
  rewards_dropped, mean_return) — `run_experiment.py:247-255`, `experiments/store.py`.
- `python -m multi_agent.experiments.plot_learning_curve` — raw points, rolling mean,
  early-vs-late Δreview.
- PPO stats logged each update: `policy.py:244-252` (mean_return, sigma, policy/value
  loss, entropy).
- **Invariant:** `rewards_dropped` MUST stay `0` on every curve run.

---

## 2. Staged steps (ranked by impact / effort)

### Stage A — Give RL reward-relevant FEATURES *(highest impact, structural)*

> Addresses root cause **#1 (fatal)**, and is a prerequisite for #4 to resolve (a
> learnable contrast can only exist once the inputs explain the reward).

**A0. Wire `include` into the RL feature extractor (plumbing — do this first).**
- File: `multi_agent/agents/rl_agent.py`, `RLScoreBehaviour.run` (`:44-55`).
  - Currently: `features = extract_features(candidates_info, context)` (`:54`), using only
    `context`. The CFP body already carries `weights_result` (`messages.py:make_cfp`,
    field `"weights_result"`), but the behaviour never reads it.
  - Change: `weights_result = data.get("weights_result", {})` then
    `extract_features(candidates_info, context, weights_result)`.
- File: `multi_agent/rl/policy.py`, `extract_features` signature (`:89`).
  - Add a 3rd param `weights_result: dict | None = None` (default-None keeps the existing
    call sites and tests working).
  - Derive the active include once: `include = (weights_result or {}).get("filters", {}).get("include", {})`.
    Mirror the clothing agent's matching semantics (`strategies/clothing.py:67-72`,
    `_axis_hit`): exact membership in the values list per axis (style/occasion/color/type
    are exact-list axes; `age_group` is substring but we don't use it here).

**A1. Widen the orchestrator's RL CFP allowlist** so style/occasion (and pattern/material
if used) reach RL on the wire.
- File: `multi_agent/agents/orchestrator.py`, `_CFP_FIELDS["rl"]` (`:94-95`).
  - From: `{"item_id","size","color","type","gender","price","stock_count","push_score"}`
  - To (add): `"style"`, `"occasion"`. (Optionally `"pattern"`, `"material"` only if we
    decide to add features for them — keep minimal for now; style/occasion are what the
    reviewer rates on per MEMORY.) These fields already exist on the candidate dict
    (`retrieval.py:_collapse_pairs_to_candidates`, `:107-111`), so this is purely a
    serialization allowlist widening.

**A2. Add the new match features** in `extract_features` (`policy.py:102-115`).
- Compute, per candidate, against the **`include`** filter (refined conversation), with a
  fallback to the raw scan for color/type so behaviour is safe when `include` is empty:
  - `style_match`    = 1.0 if `include.style`    present and `item.style`    ∈ values else 0.0
  - `occasion_match` = 1.0 if `include.occasion` present and `item.occasion` ∈ values else 0.0
  - **Refine** `color_match`/`type_match`: if `include.color`/`include.type` present, match
    against those values (the refined filter); else fall back to the raw
    `context["detected_color"]/detected_type` (current behaviour at `:94-95`, `:108-109`).
    This preserves today's signal when the parser produced no refined filter and upgrades it
    when it did.
- Keep `gender_match`, `push_norm`, `price_norm`, `stock_norm`, `bias` unchanged.

**A3. New `FEATURE_NAMES` schema** (`policy.py:58-66`). Proposed ordered schema (append
new flags at the END so the persisted-index contract is least surprising):

```
FEATURE_NAMES = (
    "bias",
    "color_match",      # now: include.color if present else detected_color
    "type_match",       # now: include.type  if present else detected_type
    "gender_match",
    "push_norm",
    "price_norm",
    "stock_norm",
    "style_match",      # NEW — include.style membership
    "occasion_match",   # NEW — include.occasion membership
)
```
`N_FEATURES` becomes 9. `ActorCritic` is built from `N_FEATURES` (`policy.py:148`), so the
network widens automatically.

**A4. Checkpoint migration / fresh-start handling.** `FEATURE_NAMES` is persisted and
hard-validated (`policy.py:273-275`: schema mismatch → start fresh). Changing the schema
**invalidates `rl_ppo.pt`**. Plan:
- The validation already **fails safe**: an old 7-feature checkpoint will mismatch the new
  9-feature schema and the loader logs a warning and starts fresh (`:274`). No crash, no
  silent corruption. This is the intended migration path.
- **Production:** on first deploy of Stage A, the live policy resets to update_count=0.
  This is acceptable (the live policy demonstrably wasn't learning), but call it out in the
  rollout note. There is no meaningful weight to migrate from 7→9 features (the semantics of
  the trunk's input layer change), so **no state-dict surgery** is attempted.
- **Experiments:** always run Stage A curves with `RL_FRESH_START=1` and an isolated
  `RL_CHECKPOINT_PATH` (see §3) so prod `rl_ppo.pt` is never touched and the curve starts
  from a clean 9-feature policy.
- Optional hardening (low priority): bump a `SCHEMA_VERSION` int into the checkpoint dict
  (`_save_locked`, `:293-301`) for human-readable diagnostics. Not required for correctness.

**Risk (Stage A):** Low–moderate. The feature widening is additive and default-safe
(empty `include` → flags are 0.0 → reduces to today's vector plus two always-zero columns,
which the network can ignore). The only breaking effect is the **forced checkpoint reset**,
which is intended and fail-safe. Tests in `tests/test_rl_policy.py` that call
`extract_features(candidates, context)` keep working because `weights_result` defaults to
None; but any test asserting `N_FEATURES==7` or the exact `FEATURE_NAMES` tuple **must be
updated** — grep `tests/test_rl_policy.py` for `FEATURE_NAMES`/`N_FEATURES`/vector length.

---

### Stage B — Reward attribution (make reward reflect RL's *own* picks)

> Addresses root cause **#2 (fatal)** and **#4**.

The problem: in `rating` mode (`store.py:169`) pass-rate is zeroed, and the episode review
is attributed to the **final Borda top-K** (`run_experiment.py:224-230`), which is the
*consensus* of all 5 agents — RL's 0.15-weighted ranking barely moves it. So the reward an
RL transition receives is almost independent of how RL scored that item.

Two complementary options, in increasing ambition:

**B1 (preferred, low effort): use `both` (or `passrate`) instead of `rating` for the
curve.** Pass-rate IS RL-attributable: `settle_round` (`store.py:163-171`) ranks RL's *own*
transitions by RL's proposed score, takes RL's top-K, and rewards how many reached the final
top-K. That reward is a direct function of RL's ranking.
- Change: run curves with `RL_REWARD_MODE=both` (pass-rate every round + review when it
  lands) rather than `rating`. **No code change** — just the env flag (`config.py:110`).
  This is also the production default, so it's the most prod-faithful test.
- Why it was set to `rating` before: to isolate the review signal. But with Stage A
  features in place, pass-rate becomes a *meaningful, attributable* dense reward, and `both`
  combines it with the (sparse, consensus) review. Expect contrast to appear here.
- Caveat: pass-rate rewards RL for *agreeing with the consensus*, not for raising reviews
  directly. That is a proxy; it's why we keep the review in `both` and measure Δreview as
  the primary success metric.

**B2 (higher fidelity, more effort): per-item credit for the review.** Instead of applying
the flat episode rating to all final top-K items equally (`run_experiment.py:224-230`),
weight each item's review-reward by **RL's own contribution** to that item making the
top-K — e.g. scale `rating_reward(rating)` by RL's normalized proposed score for that item
(available on the transition: `Transition.score`, `store.py:53`), or by
`agent_scores["rl"]` already surfaced in the result (`orchestrator.py:562-565`).
- File: `multi_agent/run.py` `submit_feedback` (`:150-192`) and/or
  `run_experiment.py:221-238`. Add an optional `credit: float` multiplier to
  `submit_feedback`, passed to `rl_store.add_reward` (`store.py:136-147`).
- Risk: changes the live `submit_feedback` contract → must be **opt-in / gated** (default
  `credit=1.0` ⇒ byte-identical to today). Feasibility is uncertain (see §6) — the per-item
  credit signal is noisy. **Do B1 first; only attempt B2 if B1's Δreview is still flat.**

**Risk (Stage B):** B1 is zero-code, zero-risk (flag only). B2 touches the prod feedback
path → must default-identical and be covered by a unit test on the multiplier.

---

### Stage C — Leverage (`RL_WEIGHT` experiment)

> Addresses root cause **#3**.

- File: `multi_agent/config.py:79` (`RL_WEIGHT = 0.15`). Already env-overridable? **No** —
  it's a bare literal. Change to `RL_WEIGHT = float(os.environ.get("RL_WEIGHT", "0.15"))`
  (mirrors the existing env-override pattern used for `SELECTION_MODE`, `VETO_TAU`, etc.).
  This is a tiny, default-identical change that unlocks A/B without code edits.
- Then run a curve with `RL_WEIGHT=0.5` (RL gets half the budget; the 4 emphases share the
  rest via `build_agent_weights`, `orchestrator.py:257-263`).
- Why: with more weight, RL's ranking actually moves the final top-K, which (a) sharpens
  pass-rate attribution and (b) makes the review respond to RL's picks — amplifying any
  signal Stage A/B created.
- **Sequencing:** only meaningful AFTER Stage A (otherwise you amplify a blind agent).
- **Risk:** moderate in production (RL would dominate before it's trusted). Keep prod
  default at 0.15; treat 0.5 as an experiment-only env override. Document that raising prod
  `RL_WEIGHT` should wait until a curve proves learning.

---

### Stage D — The two minor bugs

> Addresses root causes **#5** and **#6**. Low effort, do alongside A.

**D1. Rollout dilution.** `settle_round` (`store.py:185-192`) and `drain_rollout`
(`store.py:204-210`) push **every** transition of every settled round into the PPO batch
(`rd_done.transitions.values()`), but only ~TOP_K items per round ever received a reward;
the other ~30+ candidates sit at `reward=0.0` (`Transition.reward` default, `:54`). With
near-zero rewards everywhere, the advantage signal is swamped by zero-reward noise.
- Option D1a (safest): in the batch-assembly loops, **filter to transitions that received
  any reward** (or were among the agent's chosen top-K). Concretely, only include
  `t` where `t.reward != 0.0` OR the item was in the round's `chosen`/`final_keys`.
  - Files: `store.py:188-190` and `:207-209`.
  - Risk: changing which transitions train the critic can bias the value baseline; the
    zero-reward items are legitimate negatives for the critic. **Prefer D1b.**
- Option D1b (preferred): keep all transitions but ensure the **chosen/non-chosen split is
  meaningful** — i.e. rely on Stage B so non-final items still carry the (negative)
  pass-rate signal where appropriate, and verify empirically whether trimming helps. Treat
  D1 as an *experiment*, gated behind a config flag `RL_TRIM_ZERO_REWARD` (default False ⇒
  byte-identical), measured against the un-trimmed curve.
- **Risk:** medium — touches what the critic sees. Must be flag-gated and A/B'd, not made
  default.

**D2. Advantage-norm guard.** `adv = (adv - adv.mean()) / (adv.std() + 1e-8)`
(`policy.py:210-211`). When reward is near-constant, `adv.std()→0` and the `+1e-8`
denominator blows tiny real differences into large, noisy normalized advantages.
- Change: guard with a meaningful floor and/or skip normalization when the batch advantage
  variance is below a threshold:
  - e.g. `std = adv.std(); adv = (adv - adv.mean()) / std if std > 1e-3 else (adv - adv.mean())`
    (or simply `adv = adv - adv.mean()` when `std < eps`, leaving raw centered advantages).
- File: `policy.py:210-211`.
- **Risk:** low. This is a numerical-stability improvement; with Stage A/B producing real
  reward contrast, `std` will be healthy and the guard rarely triggers. Add a unit test:
  near-constant returns → finite, non-exploding advantages.

---

## 3. Validation protocol (per stage)

**Isolation rules (apply to EVERY curve run):**
- Always set `RL_FRESH_START=1` (start untrained) and a **dedicated, throwaway**
  `RL_CHECKPOINT_PATH` per stage so prod `rl_ppo.pt` is never touched
  (`config.py:102-103`, `:112-115`). e.g.
  `models/weights/agents/rl_ppo_curve_A.pt`, `…_B.pt`, etc.
- `rewards_dropped` must read `0` in the console summary / curve_points — if not, stop and
  fix (curve invalid). `curve` mode sets `rl_store.defer_consumption=True`
  (`run_experiment.py:331`) precisely to keep this at zero.
- Run **≥2 seeds / repeats** per stage to separate signal from LLM-review noise; compare
  the early-vs-late Δreview and the mean_return slope across seeds.

**Baseline (reproduce the flat curve first, as a control):**
```
RL_FRESH_START=1 \
RL_CHECKPOINT_PATH=models/weights/agents/rl_ppo_curve_base.pt \
RL_REWARD_MODE=rating \
SELECTION_MODE=veto_batch \
EXPERIMENT_MODE=curve \
EXPERIMENT_REPEATS=100 \
python -m multi_agent.experiments.run_experiment
python -m multi_agent.experiments.plot_learning_curve   # expect flat Δreview≈0
```

**After Stage A (features) — same harness, isolated checkpoint:**
```
RL_FRESH_START=1 \
RL_CHECKPOINT_PATH=models/weights/agents/rl_ppo_curve_A.pt \
RL_REWARD_MODE=both \                # Stage B1 folded in: use attributable pass-rate+review
SELECTION_MODE=veto_batch \
EXPERIMENT_MODE=curve \
EXPERIMENT_REPEATS=100 \
python -m multi_agent.experiments.run_experiment
python -m multi_agent.experiments.plot_learning_curve
```
**Compare vs baseline:** Δreview should turn positive (target ≥ +0.20) and mean_return
slope should turn upward. Also eyeball `value_loss`/`sigma` in the PPO logs
(`policy.py:252`) — value_loss should decline, sigma should contract.

**After Stage C (leverage):** rerun the Stage-A command with `RL_WEIGHT=0.5` and a
`…_C.pt` checkpoint; compare Δreview and slope to Stage A (expect amplification).

**After Stage D (bugfixes):** rerun with `RL_TRIM_ZERO_REWARD=1` (D1) and the D2 guard,
isolated `…_D.pt`; compare to Stage A to confirm the fixes *help or are neutral* (they must
not regress).

**Borda cross-check:** repeat the winning configuration with `SELECTION_MODE=borda` to
confirm the result isn't veto-specific (the borda baselines were also flat, exp #23/#24).

**Quick smoke test** before any long run: `EXPERIMENT_REPEATS=5` — plumbing-only, confirms
`rewards_dropped=0` and that a curve point is written per episode.

---

## 4. Sequencing & dependencies

```
A0 (wire include) ─┐
A1 (CFP allowlist) ─┼─► A2/A3 (features+schema) ─► A4 (migration) ═══ Stage A complete
                    │                                   │
D2 (adv guard) ─────┘ (independent, do anytime)         │
                                                        ▼
B1 (RL_REWARD_MODE=both)  ── independent flag, but only INFORMATIVE after Stage A ──► measure
                                                        │
C  (RL_WEIGHT env+0.5)  ── requires Stage A ────────────┤──► measure
                                                        │
D1 (rollout trim, gated) ── requires Stage A to judge ──┘──► measure
B2 (per-item credit) ── ONLY if B1 still flat ──────────────► measure
```

- **A0+A1 are inseparable** (both needed for style/occasion to reach the extractor and the
  flags to be non-zero). A2/A3/A4 follow immediately.
- **D2 is fully independent** — safe to land with A.
- **B1 is a flag** (`RL_REWARD_MODE=both`), independent of code, but only worth measuring
  once Stage A gives the features that make pass-rate/review learnable.
- **C depends on A** (don't amplify a blind agent).
- **D1 depends on A** to be evaluable (and is gated/experimental).
- **B2 is last-resort**, only if B1's Δreview is still flat.

**Recommended landing order:** A (with D2) → measure → B1 (measure) → C (measure) → D1
(measure) → B2 (only if needed).

---

## 5. Rollback / safety (per change)

| Change | Default-safe? | Rollback |
|--------|---------------|----------|
| A0 wiring (`rl_agent.py`) | yes (`weights_result` optional, empty→0 flags) | revert one line + the param |
| A1 CFP allowlist (`orchestrator.py:94`) | yes (just more fields on the wire; ignored if extractor doesn't use them) | remove `style`/`occasion` from the set |
| A2/A3 features+schema (`policy.py`) | **No — resets `rl_ppo.pt`** (fail-safe reset, not a crash) | revert `FEATURE_NAMES`/`extract_features`; old checkpoints already mismatch-and-reset, so reverting also resets once more |
| A4 migration | n/a (relies on existing mismatch→fresh path `:273-275`) | none needed |
| B1 `RL_REWARD_MODE` | yes (env flag; prod already `both`) | unset env |
| B2 per-item credit | must be gated `credit=1.0` default | revert multiplier; flag default keeps identical |
| C `RL_WEIGHT` env-ize | yes (default `"0.15"`) | keep default; never raise prod without curve proof |
| D1 rollout trim | must be gated `RL_TRIM_ZERO_REWARD=False` default | unset flag |
| D2 adv guard | yes (only changes near-degenerate batches) | revert to `/(adv.std()+1e-8)` |

**Production protection (all experiments):** every curve uses `RL_FRESH_START=1` + an
isolated `RL_CHECKPOINT_PATH`, so the live `models/weights/agents/rl_ppo.pt` is never
written by an experiment. The ONE unavoidable prod effect is the **one-time checkpoint
reset** when Stage A's 9-feature schema first ships — this is intended (the old policy
wasn't learning) and is fail-safe (load mismatch → fresh policy, `policy.py:273-275`).

---

## 6. Open questions / risks (where I'm unsure)

1. **Match-flags vs one-hot / graded encodings.** Stage A uses binary match flags
   (cheap, robust, schema-stable). But style/occasion are *multi-valued* axes — a one-hot or
   a "fraction of include-axes satisfied" scalar (mirroring `clothing.score_match_count`,
   `clothing.py:86-89`) might give the policy more gradient. **Recommendation:** start with
   flags (Stage A as written); if Δreview is positive-but-weak, try a `style_occasion_axes`
   graded scalar before going full one-hot (one-hot explodes `N_FEATURES` and worsens the
   checkpoint-migration churn).
2. **Is pass-rate a good enough proxy (B1)?** Pass-rate rewards agreeing with the *consensus*
   top-K, not directly with the review. It's attributable and dense, which is why it should
   produce *a* learnable signal — but learning to match consensus ≠ learning to raise
   reviews. If Δreview rises while pass-rate is the dominant reward, confirm via the review
   trend, not just mean_return.
3. **Per-item credit feasibility (B2).** Weighting the review by RL's own score is
   intuitively right but the signal is noisy (one review per episode, spread over ~10 items).
   Unsure it has enough resolution at ~300 episodes / ~100 updates. May need far more
   episodes or a variance-reduction scheme. Treat as exploratory.
4. **Rollout trim (D1) vs critic health.** Trimming zero-reward transitions removes
   legitimate negatives the critic uses for its baseline. Trimming *might* sharpen the actor
   but *might* destabilize the value head. Hence gated + A/B, not default.
5. **Reward scale / floor-bound reviews (#4).** Even with features, if the simulated shopper
   reviews are floor-bound (clustered low), contrast stays low. Worth checking the review
   distribution from prior runs (`scratch_diversity.py` reads `final_review`); if reviews are
   nearly constant, consider the shopper-LLM / catalog-coherence track (see MEMORY:
   `analyze_intent` style/occasion fix and catalog repair) as a parallel prerequisite — RL
   can only learn a signal that varies.
6. **Confidence-scaling interaction.** RL is never confidence-scaled
   (`orchestrator.py:248-256`), so Stage C's larger `RL_WEIGHT` is applied flat. Confirm
   that doesn't unbalance low-confidence rounds in production before raising prod weight.

---

## 7. File/line change index (quick reference)

| Stage | File | Lines | Change |
|-------|------|-------|--------|
| A0 | `multi_agent/agents/rl_agent.py` | 44-55 | read `weights_result` from CFP, pass to `extract_features` |
| A0/A2/A3 | `multi_agent/rl/policy.py` | 58-66, 89-115 | new `FEATURE_NAMES` (9), 3rd param, style/occasion + include-refined color/type flags |
| A1 | `multi_agent/agents/orchestrator.py` | 94-95 | add `style`, `occasion` to `_CFP_FIELDS["rl"]` |
| A4 | `multi_agent/rl/policy.py` | 273-275, 293-301 | (no functional change; rely on mismatch→fresh; optional schema-version stamp) |
| B1 | `multi_agent/config.py` | 110 | env only — run curves with `RL_REWARD_MODE=both` |
| B2 | `multi_agent/run.py`, `run_experiment.py` | 150-192, 221-238 | optional gated `credit` multiplier (default 1.0) |
| C | `multi_agent/config.py` | 79 | env-ize `RL_WEIGHT`; experiment at 0.5 |
| D1 | `multi_agent/rl/store.py` | 185-192, 204-210 | gated `RL_TRIM_ZERO_REWARD` zero-reward filter |
| D2 | `multi_agent/rl/policy.py` | 210-211 | advantage-norm std guard |
| tests | `tests/test_rl_policy.py` | (grep) | update any `N_FEATURES`/`FEATURE_NAMES`/vector-length assertions |
```
