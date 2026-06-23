"""
multi_agent/experiments/spec.py
───────────────────────────────
Experiment specification + the OFAT (one-factor-at-a-time) combo generator.

A *combo* is a full ``{agent_id: strategy_name}`` mapping for the four scorer
agents (colour / body / clothing / stock) — exactly the shape of
``config.AGENT_STRATEGIES`` that the harness mutates between episodes.

OFAT design:
  * the all-baseline combo (``config._DEFAULT_AGENT_STRATEGIES``) is the
    reference run;
  * for each agent, every *non-baseline* registered strategy yields one combo in
    which only that agent deviates from baseline, the other three held at
    baseline.

This keeps the run small (~one combo per off-baseline strategy + the reference)
while still attributing any review delta to a single agent's personality. The
full factorial grid is deferred (see the Part E plan).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from multi_agent.config import _DEFAULT_AGENT_STRATEGIES
from multi_agent.strategies import registry

# The four scorer agents whose personality we vary. The RL agent has no
# swappable strategy registry, so it is intentionally excluded.
SCORER_AGENTS: tuple[str, ...] = ("colour", "body", "clothing", "stock")


@dataclass(frozen=True)
class Combo:
    """One full personality assignment + a human-readable label.

    ``strategies`` is a complete ``{agent_id: strategy_name}`` mapping (all four
    scorers). ``name`` is the OFAT label, e.g. ``"baseline"`` or
    ``"colour=adventurous"``.
    """

    name: str
    strategies: dict[str, str]


@dataclass
class ExperimentSpec:
    """Top-level description of one experiment run.

    Attributes:
        name:         free-text label persisted alongside the results.
        repeats:      K — how many times each (customer × combo) is replayed to
                      average out shopper LLM noise.
        customer_ids: which personas (by ``id`` in ``customers.json``) to run.
        combos:       the personality combos to compare. Defaults to the OFAT
                      set if left empty.
    """

    name: str = "ofat_vertical_slice"
    repeats: int = 1
    customer_ids: list[str] = field(default_factory=list)
    combos: list[Combo] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.combos:
            self.combos = list(ofat_combos())


def baseline_combo() -> Combo:
    """The all-baseline reference combo (``config._DEFAULT_AGENT_STRATEGIES``)."""
    return Combo(name="baseline", strategies=dict(_DEFAULT_AGENT_STRATEGIES))


def ofat_combos() -> Iterator[Combo]:
    """Yield the OFAT combos: baseline first, then one per off-baseline strategy.

    For each scorer agent, every registered strategy that is NOT the baseline
    becomes a combo where only that agent deviates (others at baseline).
    """
    yield baseline_combo()

    for agent in SCORER_AGENTS:
        baseline_strategy = _DEFAULT_AGENT_STRATEGIES[agent]
        for strategy in registry.strategy_names(agent):
            if strategy == baseline_strategy:
                continue  # baseline already covered by the reference combo
            combo = dict(_DEFAULT_AGENT_STRATEGIES)
            combo[agent] = strategy
            yield Combo(name=f"{agent}={strategy}", strategies=combo)
