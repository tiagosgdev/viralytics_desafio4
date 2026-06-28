# Critical Review — RL Learning Fix Plan

**Reviewed:** `docs/plans/rl-learning-fix-plan.md`
**Method:** every code anchor in the plan was checked against the actual source on `rl-learning-curve`.

---

## Verdict: **APPROVE-WITH-CHANGES**

The plan's diagnosis is accurate and its code anchors are almost all correct — this is a
well-grounded plan, not a hallucinated one. But it has **three changes that must land before
any multi-hour curve run**, or the run will produce an uninterpretable or confounded result:

1. The Stage-A validation command **omits `EXPERIMENT_ORDER=interleave`**, yet the success
   criterion is early-vs-late Δreview — which is confounded by the default `persona-major`
   ordering. (Material — wastes the run.)
2. The baseline-vs-Stage-A comparison **changes two variables at once** (features *and*
   `rating`→`both`), so it cannot attribute any movement to the feature work. (Material.)
3. New match flags **inherit the clothing agent's case-sensitive matching**, which can make
   `style_match`/`occasion_match` silently always-zero and can *regress* today's
   case-insensitive `color`/`type` flags. (Material — could make Stage A a no-op.)

Plus one correctness bug in the test plan (a hardcoded 7-length vector will raise, not just
fail an assertion) and a sequencing reframing (A and B1 are co-necessary, not A-then-maybe-B1).

---

## 1. Verified claims (checked out against code)

| Plan claim | Anchor | Status |
|---|---|---|
| `weights_result` is in the CFP body but `rl_agent` never reads it | `messages.py:56-79` (make_cfp serialises `weights_result`); `rl_agent.py:44-54` parses only `conv_id/candidates/context`, calls `extract_features(candidates_info, context)` | **TRUE** |
| Candidate dict carries `style`/`occasion`; RL CFP allowlist omits them | `retrieval.py:107` (`style`), `:112` (`occasion`); `orchestrator.py:94-95` `_CFP_FIELDS["rl"]` = `{item_id,size,color,type,gender,price,stock_count,push_score}` | **TRUE** (minor: occasion is at `:112`, not within the cited `:107-111`) |
| `extract_features` takes only `(candidates, context)`; 7-feature schema | `policy.py:89`, `:58-67` (`N_FEATURES=7`), `:112-114` | **TRUE** |
| Network widens automatically from `N_FEATURES` | `policy.py:148` `ActorCritic(N_FEATURES, …)` | **TRUE** |
| Checkpoint schema-mismatch is fail-safe (reset, no crash) | `policy.py:273-275` returns on mismatch; `:276-282` wraps `load_state_dict` in try/except → fresh | **TRUE** |
| `rating` mode zeroes pass-rate | `store.py:169` `reward = 0.0 if RL_REWARD_MODE == "rating" else passrate_reward(...)` | **TRUE** |
| Pass-rate ranks/credits RL's **own** transitions | `store.py:163-165`: sorts `rd.transitions` by `t.score`, takes `[:TOP_K]`, counts those in `final_keys` | **TRUE** — `both`/`passrate` is genuinely self-attributable |
| Rollout dilution: every transition enters the PPO batch | `store.py:188-190` and `:207-209` `extend(rd_done.transitions.values())`; `Transition.reward` default `0.0` (`:54`) | **TRUE** |
| Advantage normalisation `/(std+1e-8)` | `policy.py:210-211` | **TRUE** |
| `RL_WEIGHT` is a bare literal needing env-wiring | `config.py:79` `RL_WEIGHT = 0.15` (no `os.environ`) | **TRUE** |
| `RL_REWARD_MODE`, `RL_FRESH_START`, `RL_CHECKPOINT_PATH` already env-overridable | `config.py:110`, `:115`, `:102-103` | **TRUE** |
| Curve mode sets `defer_consumption=True`; flat-rating per-item attribution | `run_experiment.py:331`; `:224-230` applies `rating` to every `last_recs[:TOP_K]` item | **TRUE** |
| `Transition.score` / `add_reward` exist for B2 | `store.py:53`, `:136-147` | **TRUE** |
| Parser enabler — prompt now populates `include.style`/`include.occasion` | `feature_weighting.py:165-176` (STYLE/OCCASION rules), example H `:276-282` emits `style`/`occasion` | **TRUE at the prompt level** — resolves the MEMORY note ("analyze_intent never extracts style/occasion"): that was the *pre-branch* state; the prompt has since been updated. See Risk R3 for the residual empirical caveat. |

**Net:** the plan's factual spine is sound. No fabricated line references of consequence.

---

## 2. Incorrect / imprecise / unverified claims

- **Test impact is understated (correctness bug).** The plan says tests calling
  `extract_features(candidates, context)` "keep working because `weights_result` defaults to
  None," and to "update any `N_FEATURES==7` assertions." That covers
  `test_extract_features_shapes_and_matches` (`tests/test_rl_policy.py:51-57`, which imports
  `N_FEATURES` dynamically and survives). **But `test_act_outputs_valid_scores_and_transitions`
  (`:62-71`) hardcodes 7-element vectors** (`:64` `{"1:M": [1.0,1,0,0,0.5,0.5,0.5], …}`). With
  `N_FEATURES=9` this is no longer a failed assertion — `p.act(feats)` builds a `[2,7]` tensor
  and the matmul against a 9-wide input layer **raises a RuntimeError**. The plan's grep
  instruction would catch it *if* "vector length" is interpreted broadly, but the prose
  ("update N_FEATURES assertions") implies a softer change. Call this out explicitly: the
  hardcoded vectors at `:64` must be widened to 9 elements.
- **"`config.py:110` — B1 is env only, no code change."** Correct, but note `submit_feedback`
  also early-returns for `RL_REWARD_MODE=="passrate"` (`run.py:171-172`). `both` is fine; just
  be aware `passrate` mode would drop the review-reward path entirely (intended, but it means
  the §3 "passrate" alternative would feed *no* review signal — don't use it for a Δreview run).
- **B2 `agent_scores["rl"]` at `orchestrator.py:562-565`** — not verified in this pass; treat as
  unconfirmed until B2 is actually attempted (B2 is last-resort anyway).
- **Severity table row #2 framing.** In `rating` mode a transition's reward is
  `rating_reward(rating)` for items in the final top-K and `0` otherwise (pass-rate zeroed).
  So it is *weakly* attributable (RL is rewarded for items that reached the consensus top-K),
  not "independent." The plan's conclusion (low attribution) is right; the wording slightly
  overstates it. Minor.

---

## 3. Material gaps & risks (ranked)

### R1 — Validation command omits `EXPERIMENT_ORDER=interleave` (would waste the run)
The default ordering is `persona-major` (`run_experiment.py:320, 383-387`): episodes march
through persona 1's repeats, then persona 2's, etc. The success metric is `plot_learning_curve`'s
**early(first 25%)-vs-late(last 25%) Δreview** (§1.1). Under persona-major, the early window is
*entirely different personas* from the late window, so Δreview measures **persona difficulty
drift, not learning**. The harness already supports `EXPERIMENT_ORDER=interleave` (`:378-382`)
precisely to de-confound this.
**Fix:** add `EXPERIMENT_ORDER=interleave` to **every** curve command in §3, and state that
early-vs-late is only valid under interleave (ideally with a within-persona slope as a
cross-check). Without this, even a real learning effect can be masked or faked by persona order.

### R2 — Baseline vs Stage A changes two variables at once (un-attributable result)
§3's baseline uses `RL_REWARD_MODE=rating`; the Stage-A command uses `RL_REWARD_MODE=both`
("Stage B1 folded in"). So the A/B comparison conflates **the feature change (A)** with **the
reward-mode change (B1)**. If Δreview moves, you cannot tell whether features or attribution did
it; if it doesn't, you can't tell which is still missing. This also contradicts §4's stated
"A → measure → B1 → measure" order.
**Fix:** pick one of:
 (a) Run the *baseline* in `both` too, so only features change between baseline and Stage A; or
 (b) Explicitly **bundle A+B1 as one intervention** and drop the pretense of measuring A alone
     (see Sequencing). Given that A-alone is unlikely to learn (R-seq below), (b) is honest and
     cheaper. Either way, hold `RL_REWARD_MODE` constant across the pair you're comparing.

### R3 — New match flags inherit case-sensitive matching (can zero out the whole feature)
The plan says to "mirror the clothing agent's matching semantics" (`clothing.py:67-72`,
`_axis_hit`). But `_axis_hit` does `item_val in values` with **no case folding** for non-`age_group`
axes. `include` values come from the LLM (`analyze_intent`), e.g. `"smart casual"`, `"work"`. If
the DB stores style/occasion with different casing/spacing, `style_match`/`occasion_match` are
**silently always 0**, and Stage A becomes a no-op — the exact failure the plan is trying to
escape. Worse: the plan also routes `color_match`/`type_match` through `include` "with clothing
semantics," which would **regress** today's *case-insensitive* color/type matching
(`policy.py:108-109` lowercases both sides).
**Fix:** for the RL feature flags, lowercase/strip **both** the item value and the `include`
values before membership testing (keep RL's existing case-insensitive behaviour, do *not* copy
clothing's case-sensitive `in`). Before the run, empirically confirm DB style/occasion values vs
the LLM's emitted casing on a 5-episode smoke test (log `include` and a few `style_match` flags).
This is a 30-minute check that protects a multi-hour run.

### R4 — Optimising a proxy (pass-rate) while measuring the target (review)
`both`/`passrate` reward **agreeing with the consensus top-K**, not raising reviews
(`store.py:163-171`). So `mean_return` can trend up (RL learns to match consensus) while Δreview
stays flat — and the plan's §1.2 (return slope) would read as "success" while §1.1 (the real
goal) doesn't move. The plan acknowledges this (open Q#2) but still lists return-slope as a
co-equal criterion. **Fix:** demote return-slope to a *mechanism check only*; gate "it learned"
strictly on Δreview under interleave. Add a sanity correlation: does RL's own per-item score
correlate with `style_match`/`occasion_match` after training (the §1.3 check)?

### R5 — Floor-bound reviews are a hard prerequisite, not a footnote
Open Q#5 correctly flags that if simulated reviews are floor-bound, contrast stays low and RL
can't learn *anything*, regardless of A/B/C/D. Per MEMORY this is the live concern (catalog/intent
coherence track). The plan treats this as a "parallel prerequisite" but does **not gate** the
expensive curve runs on it. **Fix:** make a cheap review-variance check a **blocking
precondition**: pull `final_review` distribution from a recent run (`scratch_diversity.py` /
`curve_points.rating`); if reviews are near-constant, do **not** spend hours on RL curves until
the shopper/catalog track lands. This is the single biggest "don't waste a multi-hour run" guard.

### R6 — Critic / advantage interaction not fully addressed
D2 fixes the `std→0` blow-up, good. But two related points are unaddressed: (a) when adding
two **always-zero columns** (empty `include`), the input distribution is fine, but if R3 isn't
fixed the new features carry no variance and the actor head has nothing to grip — same outcome as
no features. (b) The entropy coefficient (`PPO_ENTROPY_COEF=0.01`) and `sigma` contraction are
the exploration knobs; the plan relies on `sigma` contracting as a success signal (§1.3) but
never checks that exploration is adequate *early*. Minor, but worth a sentence: confirm `sigma`
starts healthy (`PPO_INIT_LOG_STD=-0.5`, σ≈0.61) and doesn't collapse before features matter.

---

## 4. Sequencing critique

**The plan's dependency graph under-couples A and B1.** The audit says feature-blindness AND
non-attributable-reward AND low-leverage are *each* near-fatal. Walk it through for **Stage A
alone in `rating` mode**: a transition's reward is `rating_reward` iff the item reached the
**consensus** final top-K (RL weight 0.15), else 0. RL's score barely changes which items are
final, so the reward a transition receives **barely covaries with RL's action** → policy gradient
≈ 0 → no learning, *even with perfect features*. The plan implicitly concedes this by folding B1
(`both`) into the Stage-A command — but then still describes B1 as merely "informative after A."

**Reality:** A and B1 are **co-necessary**, and C meaningfully amplifies both. My recommended
ordering for minimum wasted compute:

1. **Gate:** confirm review variance (R5) and DB-vs-LLM style/occasion casing (R3). Cheap. If
   reviews are flat, stop here.
2. **First curve = A + B1 + C(0.5) bundled, interleaved.** Maximise the chance of seeing *any*
   movement before spending hours. (Yes, this bundles three knobs — but the goal of run #1 is
   "does the signal exist at all," not attribution.)
3. **If movement appears:** ablate downward (drop C to 0.15; drop B1 to a both-baseline) to
   attribute. **If flat:** the problem is upstream (reviews/catalog), not RL knobs — do not
   proceed to B2/D1.
4. D2 lands with A (independent, safe). D1 is experimental/gated — only after a learnable
   baseline exists. B2 only if B1-bundled is still flat *and* review variance is confirmed.

This inverts the plan's "A → measure → B1 → measure → C" into "bundle-then-ablate," which is the
right strategy when each run costs hours and any single knob alone is expected to be inert.

---

## 5. Specific edits the plan needs before implementation

1. **§3 commands:** add `EXPERIMENT_ORDER=interleave` to every curve invocation (R1).
2. **§3 baseline:** hold `RL_REWARD_MODE` constant across the compared pair, or explicitly relabel
   the first run as "A+B1 bundled" (R2).
3. **§Stage A2 / A0:** specify case-insensitive (`.lower().strip()`) membership for **all** match
   flags including the new style/occasion, and keep color/type case-insensitive — do **not** copy
   `clothing._axis_hit`'s case-sensitive `in` (R3).
4. **§7 / tests:** explicitly note `tests/test_rl_policy.py:64` hardcodes 7-length vectors that
   will **raise** under N_FEATURES=9 (not just assert-fail); widen them to 9 (Section 2).
5. **Add a blocking precondition** (new §0 step): verify `final_review` variance and style/occasion
   casing on a 5-episode smoke run before any 100-repeat run (R5, R3).
6. **§1 success criteria:** demote `mean_return` slope to mechanism-only; gate "learned" on
   interleaved within-persona Δreview; add the "RL score correlates with style/occasion match"
   check as the proof that features are actually driving scores (R4).
7. **Sequencing §4:** replace the A→B1→C chain with "gate → bundle A+B1+C → ablate" (Section 4).
8. **Minor:** fix the `occasion` line ref (`retrieval.py:112`, not `:107-111`).

---

## 6. Bottom line

The plan is technically honest and its anchors verify. It is safe for production
(`RL_FRESH_START`+isolated `RL_CHECKPOINT_PATH` protect `rl_ppo.pt`; the one forced 9-feature
reset is genuinely fail-safe per `policy.py:273-282`; every new knob defaults byte-identical).
The danger is **not** in the source edits — it's in the **experiment design**: as written, the
first multi-hour curve would change two variables, measure a persona-confounded delta, and could
be silently zeroed by case-mismatched style/occasion flags, on top of a possibly floor-bound
review signal that caps the whole effort. Fix the six §5 edits — especially interleave (R1),
constant reward-mode (R2), case-insensitive flags (R3), and the review-variance gate (R5) —
and bundle A+B1(+C) for run #1, and the plan is implementation-ready.
