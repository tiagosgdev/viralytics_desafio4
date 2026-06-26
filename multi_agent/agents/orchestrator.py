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
  7. Result future resolved → caller unblocks

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
from multi_agent.aggregator import (
    borda_aggregate,
    build_agent_weights,
    reject_mass,
    survives_from_mass,
)
from multi_agent.config import (
    BATCH_SIZE,
    COLLECT_TIMEOUT_S,
    JIDS,
    MAX_BATCHES,
    QUEUE_MAX_SIZE,
    RL_ENABLED,
    RL_WEIGHT,
    SCORER_NAMES,
    SELECTION_MODE,
    STOCK_WEIGHT,
    TOP_K,
    VETO_MODE,
    VETO_TAU,
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
    make_round_result,
    parse,
)
from multi_agent.retrieval import get_candidates, get_random_candidates

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
    # rl_agent: builds match + normalised inventory/price features from these fields
    "rl":       frozenset({"item_id", "size", "color", "type", "gender",
                           "price", "stock_count", "push_score"}),
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

            # ── 2-5. Selection: legacy single-round Borda, or random-batch +
            #         weighted-veto loop, depending on SELECTION_MODE. Both
            #         return the same (top_k_keys, candidates_info, proposals,
            #         agent_weights, responded, missing) shape for the result
            #         builder below.
            if SELECTION_MODE == "veto_batch":
                outcome = await self._run_veto_batch(context, conv_id, weights_result)
            else:
                outcome = await self._run_legacy_borda(context, conv_id, weights_result)

            if outcome is None:
                # No candidates / no survivors — resolve with an empty list.
                logger.warning(f"[{conv_id}] No candidates; resolving with empty list.")
                history.record_complete(conv_id, 0, [], list(SCORER_NAMES))
                fut.set_result([])
                return

            (top_k_keys, candidates_info, proposals,
             agent_weights, responded, missing) = outcome

            # ── 6. Build rich result ───────────────────────────────────────
            result = _build_result(
                top_k_keys, candidates_info, proposals, agent_weights
            )

            history.record_complete(conv_id, len(result), responded, missing)
            fut.set_result(result)
            logger.info(f"[{conv_id}] Round complete — top-{TOP_K} resolved.")

            # ── 7. Notify the RL agent of the outcome so it can learn ──────────
            # (pass-rate reward — see RoundResultBehaviour). Fire-and-forget; it
            # must not affect the result already returned to the caller.
            if RL_ENABLED and "rl" in responded:
                try:
                    await self.send(make_round_result(JIDS["rl"], conv_id, top_k_keys))
                    comm_log("orchestrator", "rl", "INFORM", conv_id,
                             f"round_result — {len(top_k_keys)} final keys")
                except Exception as exc:
                    logger.warning(f"[{conv_id}] Could not notify RL agent: {exc}")

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

        logger.warning(f"[{conv_id}] Weight timeout — using fallback weights.")
        # Mirror weight_agent._FALLBACK_WEIGHTS so stock keeps a fair share even
        # on a timeout (all four importances present and > 0, summing to 100).
        return {
            "query":   "",
            "filters": {"include": {}, "exclude": {}},
            "weights": {
                "color":    {"importance": 30},
                "type":     {"importance": 30},
                "bodyType": {"importance": 25},
                "stock":    {"importance": 15},
            },
        }

    def _compute_agent_weights(
        self,
        context: dict,
        weights_result: dict,
        responded: list[str],
    ) -> dict[str, float]:
        """Per-agent budget from the conversation emphases + detection confidence.

        All four weights (colour/clothing/body/stock) derive from the four
        conversation-driven emphases in ``weights_result["weights"]``. STOCK_WEIGHT
        is only a fallback importance for callers that omit the stock emphasis.
        Real detection confidence quiets a low-confidence agent (stock=1.0, RL
        never confidence-scaled). With all-1.0 confidences this is identical to
        the legacy split. Missing agents have their weight redistributed.
        """
        confidences = {
            "colour":   context.get("detected_color_conf",     1.0),
            "clothing": context.get("detected_type_conf",      1.0),
            "body":     context.get("detected_body_type_conf", 1.0),
            "stock":    1.0,
        }
        return build_agent_weights(
            weights_result.get("weights", {}),
            STOCK_WEIGHT,
            rl_weight=RL_WEIGHT if RL_ENABLED else 0.0,
            confidences=confidences,
            present_agents=frozenset(responded),
        )

    async def _retrieve_candidates(
        self,
        context: dict,
        weights_result: dict,
        *,
        random_batch: bool = False,
        exclude_item_ids=None,
    ) -> list[dict]:
        """Run the (blocking) DB retrieval off the event loop.

        random_batch=False → legacy exact-filter retrieval (get_candidates).
        random_batch=True  → broad-random batch (get_random_candidates), excluding
                             item_ids already seen in earlier batches.
        """
        loop = asyncio.get_event_loop()
        if random_batch:
            return await loop.run_in_executor(
                None,
                lambda: get_random_candidates(
                    self.agent._stock_agent, weights_result, context,
                    n=BATCH_SIZE, exclude_item_ids=exclude_item_ids,
                ),
            )
        return await loop.run_in_executor(
            None,
            lambda: get_candidates(self.agent._stock_agent, weights_result, context),
        )

    async def _run_legacy_borda(self, context: dict, conv_id: str, weights_result: dict):
        """Legacy single-round path: retrieve → CFP → collect → tie-aware Borda."""
        candidates_info = await self._retrieve_candidates(context, weights_result)
        if not candidates_info:
            return None

        logger.info(f"[{conv_id}] {len(candidates_info)} candidates retrieved.")

        await self._broadcast_cfp(conv_id, candidates_info, weights_result, context)
        logger.info(f"[{conv_id}] CFP broadcast to {SCORER_NAMES}.")

        proposals, _vetoes = await self._collect_proposals(conv_id)
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

        agent_weights = self._compute_agent_weights(context, weights_result, responded)
        top_k_keys = borda_aggregate(proposals, agent_weights, k=TOP_K)
        return top_k_keys, candidates_info, proposals, agent_weights, responded, missing

    async def _run_veto_batch(self, context: dict, conv_id: str, weights_result: dict):
        """Random-batch + weighted-veto loop.

        Repeatedly draws a random broad batch (excluding seen items), broadcasts a
        CFP, collects proposals+vetoes, eliminates items via the veto rule, and
        accumulates the survivors (with their per-agent raw scores) into a running
        pool until ≥ TOP_K survive or MAX_BATCHES is reached. The accumulated pool
        is then ranked once with tie-aware weighted Borda → top-TOP_K. If still
        short, best-effort fills from the least-vetoed / highest-ranked items seen.
        """
        # Accumulated across batches:
        pool_proposals: dict[str, dict[str, float]] = {}  # agent → {item_key: score} (survivors)
        pool_candidates: dict[str, dict] = {}             # item_key → candidate dict (survivors)
        seen_item_ids: set[int] = set()
        responded_union: set[str] = set()
        # Best-effort fill pool: every item ever scored, with reject_mass, kept so
        # we can backfill if too few survive.
        fallback_proposals: dict[str, dict[str, float]] = {}
        fallback_candidates: dict[str, dict] = {}
        fallback_reject: dict[str, float] = {}

        last_weights: dict[str, float] = {}

        for batch_no in range(1, MAX_BATCHES + 1):
            candidates_info = await self._retrieve_candidates(
                context, weights_result,
                random_batch=True, exclude_item_ids=seen_item_ids,
            )
            if not candidates_info:
                logger.info(f"[{conv_id}] veto_batch: batch {batch_no} empty; stopping.")
                break

            for c in candidates_info:
                seen_item_ids.add(int(c["item_id"]))

            logger.info(
                f"[{conv_id}] veto_batch: batch {batch_no}/{MAX_BATCHES} — "
                f"{len(candidates_info)} candidates."
            )

            await self._broadcast_cfp(conv_id, candidates_info, weights_result, context)

            proposals, vetoes = await self._collect_proposals(conv_id)
            responded = sorted(proposals.keys())
            responded_union.update(responded)

            agent_weights = self._compute_agent_weights(context, weights_result, responded)
            last_weights = agent_weights

            batch_keys = {k for sc in proposals.values() for k in sc}
            info_by_key = {f"{c['item_id']}:{c['size']}": c for c in candidates_info}

            # Surface each responding agent's veto count for THIS batch.
            for agent_id in responded:
                n_vetoed = len(vetoes.get(agent_id, []))
                comm_log(agent_id, "orchestrator", "VETO", conv_id,
                         f"vetoed {n_vetoed}/{len(proposals.get(agent_id, {}))}")

            # Reject-mass per item for THIS batch (shared survival rule).
            batch_survived = 0
            batch_eliminated = 0
            for item_key in batch_keys:
                mass = reject_mass(item_key, vetoes, agent_weights, mode=VETO_MODE)
                # Track every scored item for best-effort fill.
                fallback_reject[item_key] = mass
                if item_key in info_by_key:
                    fallback_candidates[item_key] = info_by_key[item_key]
                for agent_id, sc in proposals.items():
                    if item_key in sc:
                        fallback_proposals.setdefault(agent_id, {})[item_key] = sc[item_key]

                survives = survives_from_mass(mass, mode=VETO_MODE, tau=VETO_TAU)
                if survives:
                    batch_survived += 1
                    if item_key in info_by_key:
                        pool_candidates[item_key] = info_by_key[item_key]
                    for agent_id, sc in proposals.items():
                        if item_key in sc:
                            pool_proposals.setdefault(agent_id, {})[item_key] = sc[item_key]
                else:
                    batch_eliminated += 1

            # How many items the weighted veto eliminated vs survived this batch.
            logger.info(
                f"[{conv_id}] veto_batch: batch {batch_no}: {batch_eliminated} "
                f"vetoed-out, {batch_survived} survived "
                f"(mode={VETO_MODE}, τ={VETO_TAU})."
            )

            # Distinct surviving item_ids so far.
            distinct_survivors = {k.split(":", 1)[0] for k in pool_candidates}
            logger.info(
                f"[{conv_id}] veto_batch: {len(distinct_survivors)} distinct "
                f"survivors after batch {batch_no}."
            )
            if len(distinct_survivors) >= TOP_K:
                break

        if not last_weights:
            # No batch ever produced proposals.
            return None

        responded = sorted(responded_union)
        missing   = sorted(set(SCORER_NAMES) - responded_union)

        # Rank the accumulated survivor pool with tie-aware weighted Borda.
        # NOTE: the pool spans every batch, but is ranked with the LAST batch's
        # agent_weights — an accepted approximation when earlier batches had a
        # different responded set (and thus a slightly different weight split).
        top_k_keys = borda_aggregate(pool_proposals, last_weights, k=TOP_K) if pool_proposals else []
        n_from_pool = len(top_k_keys)

        # Best-effort fill: if short, backfill from the least-vetoed / highest-Borda
        # items seen across all batches (distinct by item_id, skipping ones taken).
        if len(top_k_keys) < TOP_K and fallback_proposals:
            taken_ids = {k.split(":", 1)[0] for k in top_k_keys}
            ranked_fallback = borda_aggregate(
                fallback_proposals, last_weights, k=len(fallback_candidates) or TOP_K
            )
            # Order the fallback list by (reject_mass asc, borda rank) — the borda
            # call already orders by score; we re-sort by reject_mass first so the
            # least-vetoed items fill first.
            ranked_fallback.sort(key=lambda ik: fallback_reject.get(ik, float("inf")))
            for item_key in ranked_fallback:
                if len(top_k_keys) >= TOP_K:
                    break
                iid = item_key.split(":", 1)[0]
                if iid in taken_ids:
                    continue
                taken_ids.add(iid)
                top_k_keys.append(item_key)

        if not top_k_keys:
            return None

        # Round end: survivor pool size, how many slots came from best-effort fill,
        # and the veto mode/τ in effect.
        distinct_pool = {k.split(":", 1)[0] for k in pool_candidates}
        logger.info(
            f"[{conv_id}] veto_batch: round end — {len(distinct_pool)} distinct "
            f"survivors in pool, {len(top_k_keys) - n_from_pool} filled, "
            f"{len(top_k_keys)} final (mode={VETO_MODE}, τ={VETO_TAU})."
        )

        # Candidate dicts for whatever ended up in the top-k (survivors + fills).
        candidates_info = list({**fallback_candidates, **pool_candidates}.values())
        proposals = {
            agent_id: {**fallback_proposals.get(agent_id, {}), **pool_proposals.get(agent_id, {})}
            for agent_id in responded_union
        }
        return top_k_keys, candidates_info, proposals, last_weights, responded, missing

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

    async def _collect_proposals(
        self, conv_id: str
    ) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
        proposals: dict[str, dict[str, float]] = {}
        vetoes: dict[str, list[str]] = {}
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
                    vetoes[agent_id]    = data.get("vetoes", []) or []
            else:
                # Stale message from a previous round or wrong type — discard.
                logger.debug(
                    f"[{conv_id}] _collect_proposals: discarding stale message "
                    f"(performative={msg.get_metadata('performative')!r}, "
                    f"conv_id={(msg.get_metadata('conv_id') or '')[:8]!r})"
                )

        return proposals, vetoes



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
        self._round_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
        if self._stock_agent is None:
            loop = asyncio.get_event_loop()
            self._stock_agent = await loop.run_in_executor(None, _StockAgent)
        self.add_behaviour(OrchestratorBehaviour())
        logger.info("OrchestratorAgent ready.")

    def trigger_round(
        self,
        *,
        detected_color:           str = "",
        detected_type:            str = "",
        detected_body_type:       str = "",
        detected_color_conf:      float = 1.0,
        detected_type_conf:       float = 1.0,
        detected_body_type_conf:  float = 1.0,
        user_answer:              str = "",
        user_gender:              str = "",
        user_height_cm:           float | None = None,
        result_future:            asyncio.Future,
    ) -> str:
        """
        Queue a new recommendation round.  Returns the conv_id (UUID hex).
        The caller awaits `result_future` to get the top-10 list.

        The three `*_conf` values are the real detection confidences (0–1) for
        the colour/type/body signals; they ride in the context to the weight
        calculation so a low-confidence detection quiets its agent. They default
        to 1.0, which reproduces the legacy (confidence-agnostic) behaviour.
        """
        conv_id   = uuid4().hex
        context   = {
            "conv_id":                 conv_id,
            "result_future":           result_future,
            "queued_at":               time.monotonic(),
            "detected_color":          detected_color,
            "detected_type":           detected_type,
            "detected_body_type":      detected_body_type,
            "detected_color_conf":     detected_color_conf,
            "detected_type_conf":      detected_type_conf,
            "detected_body_type_conf": detected_body_type_conf,
            "user_answer":             user_answer,
            "user_gender":             user_gender,
            "user_height_cm":          user_height_cm,
        }
        history.record_enqueued(conv_id, context)
        try:
            self._round_queue.put_nowait(context)
        except asyncio.QueueFull:
            logger.warning(
                f"[{conv_id}] Round queue full ({QUEUE_MAX_SIZE} pending) — dropping."
            )
            history.record_failed(conv_id, "queue full")
            if not result_future.done():
                result_future.set_result([])
        return conv_id
