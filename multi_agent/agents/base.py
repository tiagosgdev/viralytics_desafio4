"""Base agent shared by all recommendation agents."""

import logging

from spade.agent import Agent

logger = logging.getLogger(__name__)


class BaseRecommenderAgent(Agent):
    """
    Thin wrapper around spade.agent.Agent that:
      - disables TLS certificate verification (dev Prosody has no signed cert)
      - exposes a convenience `agent_name` property
    """

    def __init__(self, jid: str, password: str):
        super().__init__(jid, password, verify_security=False)

    @property
    def agent_name(self) -> str:
        """Local part of the JID, e.g. 'orchestrator' from 'orchestrator@localhost'."""
        return str(self.jid).split("@")[0]
