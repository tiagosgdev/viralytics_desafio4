"""
multi_agent/experiments
────────────────────────
Part E — episodic experiment harness (vertical slice).

Compares agent *personality* combos (``config.AGENT_STRATEGIES``) by running
simulated multi-turn shopping conversations against the REAL SPADE pipeline
(``multi_agent.run.RecommendationSystem``). An LLM plays the shopper; each
episode ends in a 1–5 review — the primary comparison metric.

Modules:
  * ``customers``        — persona catalogue (loaded from ``customers.json``).
  * ``shopper``          — the LLM-played customer (ollama).
  * ``spec``             — ``ExperimentSpec`` + the OFAT combo generator.
  * ``store``            — SQLite results store (``results.db``).
  * ``run_experiment``   — CLI entry point that drives the whole loop.
"""
