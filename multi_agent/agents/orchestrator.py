"""
OrchestratorAgent
─────────────────
Coordinates one recommendation round per invocation of trigger_round().

Round protocol (closed letter / sealed bid):
  1. REQUEST → FeatureWeightAgent   : compute importance weights + DB filters
  2. INFORM  ← FeatureWeightAgent   : weights{color,type,bodyType} + filters
  3. Retrieve 40 candidates from StockAgent.get_candidates() (synchronous DB call)
  4. CFP     → all four scorer agents (simultaneously — sealed bid)
  5. PROPOSE ← body, clothing, colour, stock (each independently, no cross-talk)
  6. Weighted Borda count aggregation → top-10
  7. INFORM  → all agents (final result broadcast)
  8. Result future resolved → caller unblocks

Rounds are serialised through an asyncio.Queue.  Multiple concurrent round
triggers are queued and processed one at a time by a single CyclicBehaviour,
so there is never any cross-contamination between rounds.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from uuid import uuid4

from spade.behaviour import CyclicBehaviour

from multi_agent.agents.base import BaseRecommenderAgent
from multi_agent.aggregator import borda_aggregate, build_agent_weights
from multi_agent.config import (
    COLLECT_TIMEOUT_S,
    JIDS,
    N_CANDIDATES,
    SCORER_NAMES,
    STOCK_WEIGHT,
    TOP_K,
    WEIGHTS_TIMEOUT_S,
)
from multi_agent.history import history
from multi_agent.messages import (
    CFP,
    INFORM,
    PROPOSE,
    comm_log,
    make_cfp,
    make_request,
    parse,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STOCK_DIR = _REPO_ROOT / "stock_agent"
if str(_STOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_STOCK_DIR))

from stock_agent import StockAgent as _StockAgent  # noqa: E402

logger = logging.getLogger(__name__)

# Keys that exist only inside the orchestrator and must never be serialised
# into an XMPP message body (asyncio.Future is not JSON-serialisable).
_INTERNAL_CTX_KEYS = frozenset({"conv_id", "result_future"})

# Per-agent CFP field sets — each agent only receives the candidate fields it
# actually reads.  item_id and size are always included (they are the item key).
# The full candidate dict is kept in-process for _build_result(); these sets
# only control what is put on the wire.
_CFP_FIELDS: dict[str, frozenset[str]] = {
    # body_agent: fetches body_type from clothing.db by item_id — needs nothing else
    "body":     frozenset({"item_id", "size"}),
    # colour_agent: scores by item colour field against detected colour
    "colour":   frozenset({"item_id", "size", "color"}),
    # stock_agent: fetches push_scores from StockStats by (item_id, size)
    "stock":    frozenset({"item_id", "size"}),
    # clothing_agent: counts how many include-filter axes each item satisfies
    "clothing": frozenset({"item_id", "size", "type", "age_group", "occasion",
                           "season", "style", "pattern", "material", "gender", "fit"}),
}


def _msg_context(context: dict) -> dict:
    """Return a JSON-safe copy of context with internal orchestration keys removed."""
    return {k: v for k, v in context.items() if k not in _INTERNAL_CTX_KEYS}


def _slim_candidates(candidates: list[dict], fields: frozenset[str]) -> list[dict]:
    """Return a copy of candidates keeping only the requested fields."""
    return [{k: c[k] for k in fields if k in c} for c in candidates]


# ── Orchestration behaviour ───────────────────────────────────────────────────

class OrchestratorBehaviour(CyclicBehaviour):
    """
    One CyclicBehaviour iteration = one complete recommendation round.
    Blocks on the round queue until trigger_round() puts a context dict in.
    """

    async def run(self) -> None:
        context: dict = await self.agent._round_queue.get()
        conv_id: str  = context["conv_id"]
        fut: asyncio.Future = context["result_future"]

        # ── Staleness check ───────────────────────────────────────────────
        if history.is_stale(conv_id):
            wait_s = round(time.monotonic() - context.get("queued_at", time.monotonic()), 1)
            logger.warning(
                f"[{conv_id}] Round dropped — was queued for {wait_s}s "
                f"(queue TTL exceeded). User has likely moved on."
            )
            history.record_stale(conv_id)
            if not fut.done():
                fut.set_result([])
            return

        history.record_started(conv_id)
        logger.info(f"[{conv_id}] Round started.")

        try:
            # ── 1. Request weights ────────────────────────────────────────
            weights_result = await self._request_weights(context, conv_id)

            # ── 2. Retrieve 40 candidates (blocking DB call → executor) ───
            loop = asyncio.get_event_loop()
            candidates_info: list[dict] = await loop.run_in_executor(
                None, lambda: self._get_candidates(weights_result, context)
            )
            if not candidates_info:
                logger.warning(f"[{conv_id}] No candidates; resolving with empty list.")
                history.record_complete(conv_id, 0, [], list(SCORER_NAMES))
                fut.set_result([])
                return

            logger.info(f"[{conv_id}] {len(candidates_info)} candidates retrieved.")

            # ── 3. Broadcast CFP to all scorers (sealed bid) ───────────────
            await self._broadcast_cfp(conv_id, candidates_info, weights_result, context)
            logger.info(f"[{conv_id}] CFP broadcast to {SCORER_NAMES}.")

            # ── 4. Collect sealed proposals ────────────────────────────────
            proposals = await self._collect_proposals(conv_id)
            responded = sorted(proposals.keys())
            missing   = sorted(set(SCORER_NAMES) - set(responded))
            logger.info(f"[{conv_id}] Proposals from: {responded}.")
            if missing:
                logger.warning(
                    f"[{conv_id}] Agents did not respond: {missing}. "
                    f"Redistributing their weight among: {responded}."
                )
                comm_log("orchestrator", "—", "WARN", conv_id,
                         f"missing agents {missing} — redistributing weights")

            # ── 5. Weighted Borda count (with redistribution if agents missing) ──
            agent_weights = build_agent_weights(
                weights_result.get("weights", {}),
                STOCK_WEIGHT,
                present_agents=frozenset(responded),
            )
            top_k_keys = borda_aggregate(proposals, agent_weights, k=TOP_K)

            # ── 6. Build rich result ───────────────────────────────────────
            result = _build_result(
                top_k_keys, candidates_info, proposals, agent_weights
            )

            history.record_complete(conv_id, len(result), responded, missing)
            fut.set_result(result)
            logger.info(f"[{conv_id}] Round complete — top-{TOP_K} resolved.")

        except Exception as exc:
            logger.error(f"[{conv_id}] Round failed: {exc}", exc_info=True)
            history.record_failed(conv_id, str(exc))
            if not fut.done():
                fut.set_exception(exc)

    # ── private helpers ───────────────────────────────────────────────────────

    async def _request_weights(self, context: dict, conv_id: str) -> dict:
        req = make_request(
            to_jid  = JIDS["weights"],
            conv_id = conv_id,
            context = _msg_context(context),
        )
        await self.send(req)
        comm_log(
            "orchestrator", "weights", "REQUEST", conv_id,
            f"color={context.get('detected_color')!r}  "
            f"type={context.get('detected_type')!r}  "
            f"body={context.get('detected_body_type')!r}",
        )

        loop     = asyncio.get_event_loop()
        deadline = loop.time() + WEIGHTS_TIMEOUT_S

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            msg = await self.receive(timeout=remaining)
            if msg is None:
                break
            if (
                msg.get_metadata("performative") == INFORM
                and msg.get_metadata("conv_id") == conv_id
            ):
                data = parse(msg)
                # Filter out the final-result INFORM (sent at end of prev rounds)
                if "weights" in data:
                    return data

        logger.warning(f"[{conv_id}] Weight timeout — using equal fallback.")
        return {
            "query":   "",
            "filters": {"include": {}, "exclude": {}},
            "weights": {
                "color":    {"importance": 33},
                "type":     {"importance": 34},
                "bodyType": {"importance": 33},
            },
        }

    def _get_candidates(self, weights_result: dict, context: dict) -> list[dict]:
        query_filters: dict = dict(weights_result.get("filters") or {})

        # Inject user gender as a soft include so gender-appropriate items rank first
        gender = str(context.get("user_gender") or "").strip().lower()
        if gender in ("male", "female"):
            inc = dict(query_filters.get("include") or {})
            if "gender" not in inc:
                inc["gender"] = [gender, "unisex"]
                query_filters = {**query_filters, "include": inc}

        stock_agent = self.agent._stock_agent

        # get_candidates raises if the query is completely empty
        try:
            pairs = stock_agent.get_candidates(query_filters, n=N_CANDIDATES)
        except Exception:
            pairs = stock_agent.stats.get_overstock_items(top_k=N_CANDIDATES)

        info: list[dict] = []
        for iid, sz in pairs:
            try:
                row = stock_agent.stats.get_row(iid, sz)
            except KeyError:
                continue
            info.append({
                "item_id":    int(iid),
                "size":       sz,
                "color":      row.get("color", ""),
                "type":       row.get("type", ""),
                "fit":        row.get("fit", ""),
                "season":     row.get("season", ""),
                "style":      row.get("style", ""),
                "pattern":    row.get("pattern", ""),
                "material":   row.get("material", ""),
                "gender":     row.get("gender", ""),
                "age_group":  row.get("age_group", ""),
                "occasion":   row.get("occasion", ""),
                "brand":      row.get("brand", ""),
                "price":      row.get("price"),
                "stock_count": int(row.get("stock_count", 0)),
                "push_score": float(row.get("push_score", 0.0)),
            })
        return info

    async def _broadcast_cfp(
        self,
        conv_id: str,
        candidates_info: list[dict],
        weights_result: dict,
        context: dict,
    ) -> None:
        ctx = _msg_context(context)
        for name in SCORER_NAMES:
            fields = _CFP_FIELDS.get(name)
            payload_candidates = (
                _slim_candidates(candidates_info, fields) if fields else candidates_info
            )
            cfp = make_cfp(
                to_jid         = JIDS[name],
                conv_id        = conv_id,
                candidates     = payload_candidates,
                weights_result = weights_result,
                context        = ctx,
            )
            await self.send(cfp)
            comm_log("orchestrator", name, "CFP", conv_id,
                     f"{len(candidates_info)} candidates ({len(payload_candidates[0]) if payload_candidates else 0} fields/item)")

    async def _collect_proposals(self, conv_id: str) -> dict[str, dict[str, float]]:
        proposals: dict[str, dict[str, float]] = {}
        expected  = len(SCORER_NAMES)
        loop      = asyncio.get_event_loop()
        deadline  = loop.time() + COLLECT_TIMEOUT_S

        while len(proposals) < expected:
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.warning(
                    f"[{conv_id}] Proposal timeout: "
                    f"{len(proposals)}/{expected} received."
                )
                break
            msg = await self.receive(timeout=remaining)
            if msg is None:
                break
            if (
                msg.get_metadata("performative") == PROPOSE
                and msg.get_metadata("conv_id") == conv_id
            ):
                data     = parse(msg)
                agent_id = data.get("agent_id") or str(msg.sender).split("@")[0]
                if agent_id not in proposals:
                    proposals[agent_id] = data.get("scores", {})

        return proposals



# ── Result builder ────────────────────────────────────────────────────────────

def _build_result(
    top_k_keys: list[str],
    candidates_info: list[dict],
    proposals: dict[str, dict[str, float]],
    agent_weights: dict[str, float],
) -> list[dict]:
    info_by_key = {f"{c['item_id']}:{c['size']}": c for c in candidates_info}
    result: list[dict] = []

    for rank, item_key in enumerate(top_k_keys, 1):
        item      = info_by_key.get(item_key, {})
        iid, sz   = item_key.split(":", 1)
        agent_scores = {
            a: round(proposals.get(a, {}).get(item_key, 0.0), 4)
            for a in SCORER_NAMES
        }
        entry = {
            "rank":          rank,
            "item_id":       int(iid),
            "size":          sz,
            "agent_scores":  agent_scores,
            "agent_weights": {k: round(v, 4) for k, v in agent_weights.items()},
        }
        # Merge all item attributes (skip item_id/size — already above)
        for k, v in item.items():
            if k not in ("item_id", "size"):
                entry[k] = v
        result.append(entry)

    return result


# ── Agent class ───────────────────────────────────────────────────────────────

class OrchestratorAgent(BaseRecommenderAgent):
    def __init__(
        self,
        jid: str,
        password: str,
        stock_agent: "_StockAgent | None" = None,
    ) -> None:
        super().__init__(jid, password)
        self._round_queue: asyncio.Queue | None = None
        self._stock_agent: _StockAgent | None   = stock_agent  # shared instance or None

    async def setup(self) -> None:
        self._round_queue = asyncio.Queue()
        if self._stock_agent is None:
            loop = asyncio.get_event_loop()
            self._stock_agent = await loop.run_in_executor(None, _StockAgent)
        self.add_behaviour(OrchestratorBehaviour())
        logger.info("OrchestratorAgent ready.")

    def trigger_round(
        self,
        *,
        detected_color:      str = "",
        detected_type:       str = "",
        detected_body_type:  str = "",
        user_answer:         str = "",
        user_gender:         str = "",
        user_height_cm:      float | None = None,
        result_future:       asyncio.Future,
    ) -> str:
        """
        Queue a new recommendation round.  Returns the conv_id (UUID hex).
        The caller awaits `result_future` to get the top-10 list.
        """
        conv_id   = uuid4().hex
        context   = {
            "conv_id":            conv_id,
            "result_future":      result_future,
            "queued_at":          time.monotonic(),
            "detected_color":     detected_color,
            "detected_type":      detected_type,
            "detected_body_type": detected_body_type,
            "user_answer":        user_answer,
            "user_gender":        user_gender,
            "user_height_cm":     user_height_cm,
        }
        history.record_enqueued(conv_id, context)
        self._round_queue.put_nowait(context)
        return conv_id
