# Section IV update — IV-A and IV-B (paste-ready)

> Source: `docs/report/recommender-summary.md`. IV-B is written from the summary's Simulation B + the
> "why the signal was weak" + "future changes" material, in the paper's tone. IV-A is **preserved as in the current
> PDF** (its numbers already match the summary); see the note at the end about the one sentence that now conflicts
> with the RL result.
>
> ✅ **RL results final (exp #29, 300 episodes, 224 PPO updates, 0 dropped):** early review 2.44 → late 2.48,
> **Δ = +0.04** (below the +0.20 target); mean return **declined** 0.185 → 0.078. Filled into Table VI and the prose below.

---

## A. Analysis of Baseline Borda and Veto-Batch Frameworks

**Reframed so Borda is the clear, better choice** (it wins the primary metric — customer satisfaction), with Veto-Batch
as an explored alternative that trades relevance for variety. The **data and Tables II–IV are unchanged**; only the
framing flips. The MAS story is preserved by keeping (a) the agents drive Borda through their *weighted scores* and
(b) veto's agent-sensitivity as demonstrated capacity for future personalization.

**Replace the opening paragraph** (was: "trade-off between baseline metric inflation and architectural utility … Borda
functionally neutralizes the decentralized expertise … rendering the autonomous architecture redundant"):

> The empirical findings highlight a trade-off between **customer satisfaction** and output variety. The baseline Borda
> framework, operating over a tightly intent-matched candidate pool, achieves the **higher mean satisfaction review
> (1.57 versus 1.28)**. Its narrower pool does make the individual scoring agents produce more similar bids, so swapping
> an agent's personality moves the final slate only modestly (Δ ≤ 0.06); the agents nonetheless drive the result through
> their confidence- and intent-weighted scores. On the primary objective of customer satisfaction, the focused Borda
> slate is the stronger mechanism.

**Replace the "lower review is more honest" paragraph** (was: "the lower mean review represents a more honest,
high-entropy evaluation of a realistic retail environment"):

> Part of Veto-Batch's lower review is attributable to its breadth: presenting 7.1× more distinct items (8,978 versus
> 1,258) spreads satisfaction across a more varied top-10, of which a shopper typically selects only one or two. Even
> allowing for this, Borda's tighter, intent-matched slate yields the **higher satisfaction across all three personas**
> and is therefore the mechanism we adopt. Veto-Batch's variety and its agent-sensitivity remain valuable for future
> personalization, but on the present evidence they do not justify the relevance cost.

**Replace the final paragraph** (the RL-substrate claim):

> The broad Veto-Batch pool was expected to provide richer variance for the reinforcement-learning agent; in practice
> this advantage did not materialise, as the policy did not improve customer satisfaction under either mechanism
> (Section IV-B). We therefore adopt the **Borda count as the deployed aggregation mechanism**, while retaining
> Veto-Batch's demonstrated agent-sensitivity as a direction for future, variety-driven personalization.

**Optional — Table IV "Primary Failure Mode" row** reads pro-veto ("Borda: Safe but redundant uniformity"). To match the
reframe: Borda → *"Lower output variety"*; Veto-Batch → *"Lower satisfaction from high-entropy slate."*

---

## B. Reinforcement Learning

The reinforcement-learning agent was evaluated **independently of the agent-personality analysis** so that the two
effects could be isolated. Whereas Section IV-A varies the scoring agents' personalities while holding the RL policy
fixed, this evaluation holds the personalities at their defaults and lets the RL policy learn online over a long sequence
of recommendation episodes, recording the satisfaction review as a function of training. In each episode an LLM shopper
role-plays a customer through a full dialogue and then rates **each** of the ten final recommendations individually on a
1–5 scale; these per-item ratings are the agent's reward, matching the per-item feedback collected by the production
interface.

Over 300 training episodes (224 policy updates) the policy updated continuously — rewards landed on every round with no
dropped updates — yet the mean review did not improve: the early-to-late change was **Δ = +0.04** (against a target of
≥ +0.20) and the mean return did not trend upward but in fact declined (0.185 → 0.078). Table VI summarizes the early-
versus late-training reviews. In this configuration the RL agent **did not learn to raise customer satisfaction** — an
honest negative result whose causes are instructive.

**TABLE VI. RL TRAINING — EARLY VS LATE REVIEW**

| Training phase                | Mean review (1–5) | Mean return |
| ----------------------------- | :---------------: | :---------: |
| Early (first 25% of episodes) |       2.44        |    0.185    |
| Late (last 25% of episodes)   |       2.48        |    0.078    |
| **Δ (late − early)**          |     **+0.04**     | **−0.107**  |

The limitation lies in the **learning signal** rather than the implementation. First, the RL agent contributes only a
single weighted slice to a five-agent consensus, so the final slate — and therefore the reward a recommendation receives
— is only weakly a consequence of the agent's own ranking, which blurs the gradient. Second, part of the reward credits
the agent for *agreeing with the consensus* top-10 rather than for raising the review, which can drive the policy toward
imitating the other agents instead of exercising its own judgement. Third, the satisfaction reviews occupy a narrow band
— shown ten options a shopper genuinely prefers only one or two, and the catalog's weakly-correlated attributes cap how
well any slate can match a specific multi-attribute goal — leaving little variance for the policy to exploit. Finally,
online learning over a few hundred episodes is a short horizon for a learned ranking policy.

Several changes are expected to unlock learning. The most promising is to promote the RL policy from a peer voter to the
**orchestrating aggregator** that consumes the other agents' scores and produces the final selection itself, so the
review becomes the direct, attributable consequence of its own decision. Complementary directions include shaping the
reward to credit the agent specifically for the items it elevated that earned high reviews, training on **real
production feedback** and a coherence-correlated catalog to widen the signal's variance, and warm-starting the policy
from logged recommendations before online fine-tuning.

---

---

## Other consistency updates (rest of the report)

I read the whole paper for places that recent developments (per-item feedback; the RL reward = pass-rate + per-item
satisfaction; the null learning result) make inconsistent. The robotics (IV-C / III-B) and IoT/sensor sections are
unaffected. Two genuine fixes and a few optional polish items, all small and length-preserving:

### 1. III-D — RL reward description (recommended; also fixes a grammar slip)

The closing paragraph of *D. Multi-Agent system* currently describes the RL reward as **only** consensus survival, which
(a) is incomplete now that per-item ratings are used and (b) is the exact mechanism IV-B critiques — so they should line
up. It also has a missing verb ("an embedded RL Agent [37] **to adjust**").

> ~~To maintain dynamic optimization within this negotiation structure, an embedded RL Agent [37] to adjust its bids over
> time, maximizing its reward based on how many of its individual selections successfully survive into the final
> aggregate consensus.~~

**→**

> To maintain dynamic optimization within this negotiation structure, an embedded RL Agent [37] **adjusts** its bids over
> time, maximizing a reward that combines **how many of its selections survive into the final aggregate consensus (a
> pass-rate signal) with the per-item 1–5 satisfaction ratings the customer later returns.**

### 2. III-E — cognitive-layer reward (recommended)

*E. Reinforcement Learning*, cognitive layer, says feedback is "explicit user selections or rejections." Recent
development: the customer rates **each** recommended item 1–5 (per-item), which is what IV-B reports on.

> ~~Explicit user selections or rejections serve as direct reward signals, allowing the agent to dynamically maximize
> cumulative user satisfaction and refine future recommendation vectors.~~

**→**

> **Per-item 1–5 satisfaction ratings on each recommended garment** serve as direct reward signals, allowing the agent
> to dynamically maximize cumulative user satisfaction and refine future recommendation vectors.

### 3. Optional polish (only if you want tighter consistency)

- **III-A, RL Agent bullet:** "…automatic pass-rate signals and explicit user satisfaction feedback…" → "…automatic
  pass-rate signals and explicit **per-item** user satisfaction ratings…" (one word, matches §1/§2 above).
- **III-A / III-E wording of "learns":** these design sentences state the agent "learns which characteristics produce
  satisfying recommendations" as a capability. IV-B reports it did **not yet** show that gain. If you want them strictly
  consistent, hedge to "is **designed to** learn…". (Optional — design-intent vs measured-result phrasing is common and
  not strictly a contradiction.)
- **III-E state space:** "(body type, color preferences)" → "(body type, colour, type, style and occasion
  preferences)" if you want it to match the RL feature set named in IV-B. (Optional.)

### Not changed (deliberately)

- **Abstract / Intro / State-of-the-Art.** No claims there are contradicted by the new results.
- **III-D mechanism descriptions.** III-D still presents Borda (baseline) and Veto-filtered Borda (redesign)
  neutrally — that's fine and consistent with adopting Borda as deployed. If you want it to foreshadow the IV-A
  conclusion, you could append to the Veto-filtered Borda bullet: *"…though at a measurable cost to mean satisfaction
  (see Section IV-A)."* (Optional.)

### Consistency check after the flip

The IV-A reframe now **aligns the paper with `recommender-summary.md`** (Borda = deployed choice). Make sure nothing
else still implies veto is the preferred/deployed mechanism — I did not find such a claim outside IV-A, but worth a
final read of any sentence that calls Veto-Batch "the redesign" approvingly.
