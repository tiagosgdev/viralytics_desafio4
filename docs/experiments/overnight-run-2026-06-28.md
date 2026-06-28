# Overnight run — 2026-06-28 (agents grid + RL curve)

_Launched by Claude on request: run BOTH simulations, log results, summaries at the
end, check every 30 min, stop+fix+rerun on any major issue._

## ⚠️ CODE CHANGE THIS RUN — parser fix (style/occasion → include)
Before this run, `analyze_intent` (`LNIAGIA/query_parsing/feature_weighting.py`)
extracted only color/type/body_type into `include`; style+occasion were dropped
("casual is style and is ignored"). Every persona goal hinges on style+occasion,
so the recommender couldn't tell in-band from off-band → reviews capped ~2.0.

**Fix (prompt-only, localized to `_build_intent_system_prompt`):** the scan-mode
prompt now extracts STYLE + OCCASION into `include` (added the enums + rules +
Example H; removed the "ignored" rule). Verified the whole path was already built
for it: `_validate` keeps them (valid `ALL_MAPPINGS` keys), the clothing agent
scores any include axis (`SKIP_AXES={body_type}` only), candidates already carry
style/occasion columns, and retrieval is match_count (re-ranks, never dead-ends).

**Smoke (passed before launch):**
- analyze_intent now returns style+occasion for all 3 personas (daniel
  style=[smart casual,minimalist] occasion=[work]; maya elegant/party; sofia
  casual,minimalist/everyday) — and daniel's type stayed long_sleeve_top (old
  misparse gone).
- clothing `score_match_count` now differentiates: full in-band **1.0**,
  color+type-only **0.5**, off-band **0.0** (before: 1.0 vs 0.67 — indistinguishable).

The fix is **uncommitted** on `rl-learning-curve` (pending user review in the AM).
This **resets the metric epoch** — #20+ not comparable to #15/#12/#17.

## Plan
All sims share the same XMPP JIDs → run **sequentially**, not in parallel.
Order (per user): **borda grid → veto grid → RL curve.** K=3 → **729 eps/grid** (per user).

> **Episode count: 243 per grid, not 729.** `EXPERIMENT_REPEATS` defaults to 1
> (81 combos × 3 personas × 1). The exp #9/#15 baselines were K=3 (729). So K=1
> grids are noisier per-cell — compare *means* with that caveat. Chosen to fit all
> three sims overnight at ~25s/episode.
>
> **Experiment IDs:** #18 is a pre-existing empty placeholder (0 episodes). Real IDs:
> borda **#19**, veto **#20**, RL curve **#21**.

1. **Borda grid** (`SELECTION_MODE=borda EXPERIMENT_MODE=full`) → experiment **#19**. 243 eps.
   - RL isolated to `rl_ppo_grid_scratch.pt` (`RL_FRESH_START=1`) so prod `rl_ppo.pt` untouched.
   - No veto-pool knobs (borda is legacy single-round). Compare to exp #9 (old rubric, 1.57).
2. **Veto grid** (`SELECTION_MODE=veto_batch EXPERIMENT_MODE=full`) → experiment **#20**. 243 eps.
   - `SURVIVOR_TARGET=40 BATCH_SIZE=80 MAX_BATCHES=10` (WS-B bigger veto pool). RL isolated to scratch.
   - Compare to exp #15 (full veto_batch grid, mean **1.99**).
3. **RL curve** (`EXPERIMENT_MODE=curve`) → experiment **#21**. ~300 eps.
   - `RL_REWARD_MODE=rating RL_FRESH_START=1 RL_CHECKPOINT_PATH=…/rl_ppo_curve.pt EXPERIMENT_REPEATS=100`
   - Compare to exp #12 (Δreview −0.05, mean 1.31, **old** rubric — not directly comparable post-recalibration).

Logs: `/tmp/borda_overnight.log`, `/tmp/veto_overnight.log`, `/tmp/curve_overnight.log`.
Prereqs verified at launch: XMPP up, Ollama up (14b + 7b-q3), tree clean, no run in flight.

> NOTE on expectations (from memory): the real metric blocker is `analyze_intent`
> never extracting style/occasion into `include`. These runs are expected to land
> ~2.0 again; that CONFIRMS the parser is the lever. A flat result here is a valid
> negative, not a failure of the run.

## Monitoring journal
(times are local; appended every ~30 min)

> Real experiment IDs this run: **borda #20, veto #21, RL curve #22** (729/729/~300 eps).
> (#18 empty placeholder; #19 = the earlier K=1 borda, killed when scope changed to K=3 + parser fix.)

| time | check | status |
|------|-------|--------|
| 00:20 | (superseded) K=1 borda #19 | killed — user changed scope to K=3 + parser fix |
| 00:58 | borda grid #20 launched (PID 2236611), K=3/729 eps, parser fix live | ✅ agents online, no errors, reviews landing (ep5 maya=2). Chain orchestrator (PID 2240742) armed for veto→RL. |
| 01:21 | borda #20 @ 67/729 (all party_maya so far) | ✅ **PARSER FIX SHOWING LIFT.** maya mean **2.73** (range 1–4), dist {1:1, 2:23, 3:35, 4:7}. vs baselines: exp#15 maya **1.85**, exp#9 borda maya **1.26**. 4s now ~11% of eps (exp#15 had 5 fours in ALL 729). No errors/abandons. ~20s/ep → borda ~4h. daniel+sofia still to come. |
| 02:28 | borda #20 @ 269/729 | ✅ maya **final 2.48** (243 eps, vs exp#15 1.85 → **+0.63**). office_daniel starting: **2.88** @ 25 eps (vs exp#15 2.22, exp#9 borda 1.81 → **+0.66**). Daniel hinges on style=smart/minimalist + occasion=work — exactly what the fix now captures, and the persona whose catalog conjunction went 2→70. Strong confirmation. No errors. |
| 02:57 | borda #20 @ 365/729 | ✅ daniel **2.93** @ 121 eps (+0.71 vs exp#15 2.22). overall 2.63. No errors. |
| 03:56 | borda #20 @ 561/729, all 3 personas landed | ✅ **ALL PERSONAS LIFTED.** daniel **final 2.91** (+0.69 vs 2.22), maya **final 2.48** (+0.63 vs 1.85), sofia **2.32** @ 74 eps (+0.42 vs exp#15 1.90, exp#9 1.65). overall **2.64**. sofia lowest (her goal = casual/minimalist + everyday — broad, less discriminating). No errors. Borda ETA ~04:50 → veto #21 next. |
| 04:46 | **BORDA #20 DONE (729 eps)** | ✅ overall **2.56** (vs exp#15 1.99 → **+0.57**). per-persona daniel **2.91** / maya **2.48** / sofia **2.28**. dist {1:10, 2:364, 3:295, **4:60**, 5:0}. 3s now ~40%, 4s ~8% (exp#15: 65% 2s, 5 fours total). Ceiling still 4. Top combos 3.11 (all colour=purist). 0 errors/abandons. |
| 04:55 | **VETO #21 auto-started** (chain handoff clean) | ▶ running, maya 2.0 @ 26 eps (early). veto's broad OR-band dilutes the lift vs borda — expected. RL still isolated to scratch. |
| 06:23 | veto #21 @ 298/729, daniel landing | maya **final 1.96** (+0.11 vs exp#15 1.85 — nearly flat) but daniel **2.65** @ 54 eps (+0.43 vs 2.22 — clear lift). So the fix lifts veto too, just less than borda (daniel +0.43 veto vs +0.69 borda). **Pattern: tightly-specified goals (daniel: smart/minimalist+work, 70 in-band items) survive the broad pool; broad goals (maya) get diluted.** overall veto 2.09 (rising as daniel fills). No errors. |
| 08:34 | **VETO #21 DONE (729 eps)** | overall **2.11** (vs exp#15 1.99 → **+0.12**, modest). per-persona daniel **2.41** / maya **1.96** / sofia **1.96**. Veto's broad OR-band absorbs most of the fix's benefit (vs borda +0.57). daniel — the tightest goal — is the only clear veto winner (+0.19). 0 errors. |
| 08:46 | **RL CURVE #22 auto-started** (final handoff clean) | ▶ running 39/300, mean rating 2.03, 28 PPO updates, **rewards_dropped=0 ✓** (plumbing intact, transition-merge fix holding). Too early to judge learning — the early-vs-late Δ at the end is the verdict. |
| 09:16 | curve #22 @ 134/300 — ⚠️ **CONFOUND CAUGHT** | Naive early25%(2.03)→late25%(2.44)=+0.41 looks like learning but is **persona-order artifact**: curve is persona-major (maya eps1–100, daniel 101–200, sofia 201–300), so early=maya(~2.0) vs late=daniel(~2.4) is just the persona gap, NOT PPO learning. **DE-CONFOUNDED (within-persona, the valid view):** maya first-50 **2.00** → 2nd-50 **2.18** = **+0.18** over ~75 updates — weak positive hint (vs exp#12's flat null), but small/possibly noise. ⚠️ FOLLOW-UP: I should have set `EXPERIMENT_ORDER=interleave` for the curve to get a clean early-vs-late; will report within-persona slopes for all 3 at the end + recommend an interleave re-run for a definitive verdict. rewards_dropped=0. No errors. |
| 10:06 | **ALL DONE** ✅ — all 3 sims complete, 0 errors, 0 rewards dropped | See **`comparison_parser_fix_2026-06-28.md`** for the full write-up. Headlines below. |

## ━━ FINAL RESULTS (all 3 done, 0 errors, 0 abandoned) ━━
**Grids (vs exp#15 baseline 1.99):**
- **borda #20: 2.56 (+0.57)** ✅ — daniel 2.91 / maya 2.48 / sofia 2.28. 3s 16→40%, 4s 0.7→8.2% (60 fours vs 5 in all of #15).
- **veto #21: 2.11 (+0.12)** — daniel 2.41 / maya 1.96 / sofia 1.96. Broad OR-band absorbs most of the fix; daniel (tight goal) the only clear veto winner.
- **Ceiling still 4** — 0 fives in either grid; a 5 has NEVER occurred (~3000+ eps). Chase-5 lever = same fix on pattern/material.

**RL curve #22 (300 eps, 224 PPO updates, 0 dropped): STILL NULL.**
- Within-persona rating slopes weakly +ve (maya +0.18, daniel +0.08, sofia +0.04) BUT **mean_return flat −0.016 across all 224 updates** = no PPO gradient (same mechanism as exp#12) → upticks are noise, not learning.
- Root cause: **curve runs in veto_batch** (the mode the fix helped least) → reviews still floor-bound (2.16) → rewards mostly ≤0 → advantages collapse.
- **LEVER FOR NEXT:** run the curve in **borda** mode (40% 3s + 8% 4s → real +reward mass) + `EXPERIMENT_ORDER=interleave` + boost RL Borda weight.

**Verdict on the two goals:** scores ✅ (borda +0.57, legit, no goalpost-moving). RL learning ❌ (still null — needs the curve in borda mode, not the parser fix).

**Uncommitted:** `LNIAGIA/query_parsing/feature_weighting.py` (parser fix) — pending your review before commit. Prod `rl_ppo.pt` untouched (scratch ckpts only).

## NO 5s YET — ceiling still 4 (follow-up lever for the morning)
exp#20 dist @ 288 eps: {1:4, 2:145, 3:115, **4:24**}. The **4-rate jumped ~12×**
(24/288 ≈ 8% vs exp#15's 5 fours in ALL 729 ≈ 0.7%) — mass moved up the scale —
but **0 fives, and a 5 has NEVER occurred in any experiment (max ever = 4).**
Why: the fix made style+occasion matchable, but `pattern`/`material`/`fit` are
still random per item (not in `include`; catalog repair conditioned style/type but
left those uncorrelated), so a set is rarely "perfect" enough for a 5.
**To chase a 5 (NOT done — morning decision):** either (a) add `pattern`/`material`
to `include` (parser change; already valid `ALL_MAPPINGS` keys), or (b) extend the
catalog repair to condition pattern/material on style/occasion so a style match
drags them on-theme. Same playbook as this fix, one more axis.
