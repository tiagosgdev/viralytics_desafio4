"""
Pluggable scorer strategies ("personalities") for the SPADE agents.

Each scorer agent's scoring is a pure, swappable function with the signature::

    score(candidates, context, weights_result, params) -> {item_key: float in [0,1]}

where ``item_key = f"{item_id}:{size}"``. The live SPADE agents resolve their
configured strategy from :mod:`registry` and call it after doing any needed
IO/enrichment (DB / StockStats lookups stay in the agents, never in the pure
strategies). See each module's docstring for its IO convention and params.
"""

from multi_agent.strategies.registry import get_strategy, strategy_names

__all__ = ["get_strategy", "strategy_names"]
