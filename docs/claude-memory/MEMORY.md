# Project Memory Index

- [SPADE experiments — current state](spade-experiments-state.md) — A–F + Part E slice + chat wiring/retrieval/pose fixes all done & verified. NEXT: test harness OFAT live → full grid. START HERE.
- [SPADE key design decisions](spade-key-design-decisions.md) — weighted Borda, weight=emphasis×confidence, RL fixed slice, confidence sources, write-only memory.
- [SPADE workflow](spade-workflow.md) — plan→implement(fresh subagent)→review(fresh subagent)→show before commit.
- [SPADE env/setup](spade-env-setup.md) — system python3 + spade 4.1.4, don't pip install -r (numpy<2), XMPP broker, test commands.
- [Recommender architecture](recommender-architecture.md) — two disjoint recommenders (camera→SPADE, chat→search); chat→agent wiring added 156ced3; confidence-from-scan/importance-from-chat model.
- [Experiment run 1 findings](spade-experiment-run1-findings.md) — first live OFAT run done; harness valid; 3 recommender bugs (size-dup, final-turn drift, soft-attrs ignored); shopper-LLM verdict. Fix dedup first.
