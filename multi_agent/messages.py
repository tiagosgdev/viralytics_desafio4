"""
ACL message builders and parser for the multi-agent recommendation protocol.

Message flow per round:
  Orchestrator  → FeatureWeightAgent : REQUEST  (context)
  FeatureWeightAgent → Orchestrator  : INFORM   (weights + filters + query)
  Orchestrator  → each scorer        : CFP      (40 candidates + weights + context)
  each scorer   → Orchestrator       : PROPOSE  (sealed Borda scores)
  Orchestrator  → all agents         : INFORM   (final top-10 result)

Every message carries a `conv_id` metadata key so concurrent rounds can be
distinguished without ambiguity.
"""

import json

from spade.message import Message


# ── Communication trace ───────────────────────────────────────────────────────

def comm_log(
    from_agent: str,
    to_agent: str,
    performative: str,
    conv_id: str,
    detail: str = "",
) -> None:
    """Print a one-line trace of every XMPP message exchanged between agents."""
    cid = conv_id[:8] if conv_id else "--------"
    detail_str = f"  {detail}" if detail else ""
    print(
        f"  [XMPP] {from_agent:>14} → {to_agent:<14}  "
        f"{performative.upper():<8}  [{cid}]{detail_str}",
        flush=True,
    )

# ── ACL performatives ─────────────────────────────────────────────────────────
CFP     = "cfp"      # call for proposals — sent to scorer agents
PROPOSE = "propose"  # sealed bid — scorer agents reply with ranked scores
INFORM  = "inform"   # broadcast result / weight reply
REQUEST = "request"  # orchestrator asks FeatureWeightAgent to compute weights


# ── Builders ──────────────────────────────────────────────────────────────────

def make_request(to_jid: str, conv_id: str, context: dict) -> Message:
    """Orchestrator → FeatureWeightAgent: trigger weight computation."""
    msg = Message(to=to_jid)
    msg.body = json.dumps({"conv_id": conv_id, "context": context})
    msg.set_metadata("performative", REQUEST)
    msg.set_metadata("conv_id", conv_id)
    return msg


def make_cfp(
    to_jid: str,
    conv_id: str,
    candidates: list[dict],
    weights_result: dict,
    context: dict,
) -> Message:
    """
    Orchestrator → scorer agent: sealed-bid call for proposals.

    `candidates`    — list of item dicts (item_id, size, color, type, …)
    `weights_result` — full analyze_intent output: {query, filters, weights}
    `context`       — round context: detected_color, detected_type, detected_body_type, …
    """
    msg = Message(to=to_jid)
    msg.body = json.dumps({
        "conv_id":       conv_id,
        "candidates":    candidates,
        "weights_result": weights_result,
        "context":       context,
    })
    msg.set_metadata("performative", CFP)
    msg.set_metadata("conv_id", conv_id)
    return msg


def make_propose(
    to_jid: str,
    conv_id: str,
    agent_id: str,
    scores: dict[str, float],
    vetoes: list[str] | None = None,
) -> Message:
    """
    Scorer agent → Orchestrator: sealed proposal.

    `scores` maps item_key ("item_id:size") → raw score in [0, 1].
    Higher score = agent believes this item better suits its criterion.

    `vetoes` is the optional list of item_keys this agent rejects (score below
    its personality `veto_threshold`). Defaults to an empty list, so legacy
    callers that pass only `scores` are unaffected. Consumed by the veto_batch
    selection path; ignored by legacy Borda.
    """
    msg = Message(to=to_jid)
    msg.body = json.dumps({
        "conv_id":  conv_id,
        "agent_id": agent_id,
        "scores":   scores,
        "vetoes":   list(vetoes or []),
    })
    msg.set_metadata("performative", PROPOSE)
    msg.set_metadata("conv_id", conv_id)
    return msg


def make_inform(to_jid: str, conv_id: str, payload: dict) -> Message:
    """Generic INFORM message (weights reply or final result broadcast)."""
    msg = Message(to=to_jid)
    msg.body = json.dumps({"conv_id": conv_id, **payload})
    msg.set_metadata("performative", INFORM)
    msg.set_metadata("conv_id", conv_id)
    return msg


def make_round_result(to_jid: str, conv_id: str, final_keys: list[str]) -> Message:
    """
    Orchestrator → RL agent: end-of-round notification carrying the final top-K
    item keys.  Tagged with event="round_result" so the recipient can tell it
    apart from other INFORM messages.  This realises the "Orchestrator → agents
    INFORM (final result)" step described at the top of this module.
    """
    return make_inform(
        to_jid  = to_jid,
        conv_id = conv_id,
        payload = {"event": "round_result", "final_keys": list(final_keys)},
    )


# ── Parser ────────────────────────────────────────────────────────────────────

def parse(msg: Message) -> dict:
    """Deserialise a message body to a dict."""
    return json.loads(msg.body)
