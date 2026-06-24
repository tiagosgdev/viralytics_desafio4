---
name: spade-workflow
description: "How the user wants SPADE feature work done — plan, then implement→review via fresh subagents, show review before committing"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: f5c32068-8876-46a2-a96b-d678ae671701
---

The user's established way of working on the SPADE feature (see [[spade-experiments-state]]):

**Per part: plan first, then implement→review→show→commit.**
1. Plan the change (often in plan mode; keep `docs/plans/spade-dynamic-weights-experiments.md` updated).
2. Implement using a **fresh general-purpose subagent with NO extra context** (cold brief).
3. Review using a **separate fresh subagent** (independent, no implementer context) — so reviews aren't context-polluted.
4. **Show the review results to the user BEFORE committing anything.** Apply the agreed nits, then commit.

**Why:** the user explicitly wants implementer and reviewer isolated, and wants to see review
findings before any commit lands.

**How to apply:** for each Part (and for merges), spawn one implementer subagent then one reviewer
subagent; relay the verdict + blocking/nit findings; wait for the user's go-ahead on nits; commit
with a `Co-Authored-By: Claude Opus 4.8` trailer. **Route ALL implementation through a fresh
subagent — including small follow-up nit fixes; do NOT hand-edit files yourself.** (2026-06-23: the
user corrected me mid-task — "dont implement yourself. run the pipeline agents" — after I applied a
one-var nit fix directly. When code is already committed, don't re-implement it via a subagent; just
run the reviewer subagent on it.) The user trims speculative scope aggressively —
when a feature isn't load-bearing, say so honestly rather than over-building. Be honest when
something isn't actually wired/working (they value "it's not doing X" over reassurance).
