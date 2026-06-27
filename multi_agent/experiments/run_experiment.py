"""
multi_agent/experiments/run_experiment.py
──────────────────────────────────────────
CLI entry point for the Part E experiment harness (vertical slice).

    python -m multi_agent.experiments.run_experiment

What it does:
  1. Starts the REAL pipeline once — ``RecommendationSystem`` from run.py — using
     the same ``spade.run(main())`` startup the standalone demo uses. The XMPP
     broker must already be running (``docker compose up -d xmpp``) and Ollama
     must be reachable (the FeatureWeightAgent and the LLM shopper both use it).
  2. Loads the personas (``customers.json``) and builds the OFAT combo set
     (``spec.ofat_combos()``).
  3. For each (customer × combo × repeat):
       * mutates ``config.AGENT_STRATEGIES`` to the combo (restored afterwards —
         each scorer agent reads that dict LIVE every round, so no restart is
         needed);
       * runs the episode loop: ``recommend(...)`` → ``shopper.next_message`` →
         feed its message back as the next ``user_answer`` → … until the shopper
         stops or ``MAX_TURNS`` → ``shopper.final_review``;
       * persists the experiment / episode / turns / items to ``results.db``.
  4. Prints a mean-review-per-combo summary table and stops the system in a
     ``finally``.

This NEVER edits the live pipeline. The only state it touches at runtime is the
``config.AGENT_STRATEGIES`` dict, which it always restores.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import spade

from multi_agent import config
from multi_agent.experiments import shopper
from multi_agent.experiments.spec import (
    Combo,
    ExperimentSpec,
    baseline_combo,
    full_factorial_combos,
    ofat_combos,
)
from multi_agent.experiments.store import ResultsStore
from multi_agent.rl.policy import rl_policy
from multi_agent.rl.store import rl_store
from multi_agent.run import RecommendationSystem

logger = logging.getLogger(__name__)

_CUSTOMERS_PATH = Path(__file__).resolve().parent / "customers.json"


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_personas() -> list[dict]:
    """Load the persona catalogue from customers.json."""
    with open(_CUSTOMERS_PATH, encoding="utf-8") as fh:
        personas = json.load(fh)
    if not isinstance(personas, list):
        raise ValueError("customers.json must contain a JSON array of personas.")
    return personas


def _git_sha() -> str:
    """Best-effort short git SHA for provenance (empty string if unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def _apply_combo(combo: Combo) -> dict[str, str]:
    """Mutate config.AGENT_STRATEGIES IN PLACE to the combo; return the prior state.

    The dict is mutated in place (not reassigned) so the scorer agents — which
    hold a reference to the same dict and read it live each round — see the new
    personalities immediately. The returned snapshot is used to restore it.
    """
    snapshot = dict(config.AGENT_STRATEGIES)
    config.AGENT_STRATEGIES.update(combo.strategies)
    return snapshot


def _restore_strategies(snapshot: dict[str, str]) -> None:
    """Restore config.AGENT_STRATEGIES to a previously captured snapshot."""
    config.AGENT_STRATEGIES.clear()
    config.AGENT_STRATEGIES.update(snapshot)


# ── one episode ────────────────────────────────────────────────────────────────

async def _run_episode(
    system: RecommendationSystem,
    store: ResultsStore,
    experiment_id: int,
    persona: dict,
    combo: Combo,
    repeat_idx: int,
    feed_review: bool = False,
    episode_no: int = 0,
) -> int | None:
    """Run one full simulated conversation; persist it; return the 1–5 review.

    The persona supplies the (faked) camera detections; the LLM shopper supplies
    each successive ``user_answer``. Intent is re-derived every round by the real
    FeatureWeightAgent — exactly the multi-turn refinement the human flow uses.
    """
    detected_color     = str(persona.get("detected_color", ""))
    detected_type      = str(persona.get("detected_type", ""))
    detected_body_type = str(persona.get("detected_body_type", ""))
    color_conf         = float(persona.get("detected_color_conf", 1.0))
    type_conf          = float(persona.get("detected_type_conf", 1.0))
    body_conf          = float(persona.get("detected_body_type_conf", 1.0))
    gender             = str(persona.get("gender", ""))
    height_cm          = persona.get("height_cm")
    height_cm          = float(height_cm) if height_cm is not None else None

    history: list[dict] = []        # persisted turns: [{shopper_msg, agent_weights, items}]
    shopper_msgs: list[str] = []    # individual shopper utterances (for accumulation)
    last_recs: list[dict] = []
    last_round_id: str = ""         # conv_id of the most recent round (for feedback)
    abandoned = False

    for turn_idx in range(shopper.MAX_TURNS):
        # Accumulated intent mirrors the frontend's accumulatedUserIntent(): join
        # the last 6 shopper utterances so the colour/type anchor survives a later
        # turn that only refines fit/style. Empty on the first (camera-only) round,
        # which keeps the rule-based weights for that round.
        user_answer = ". ".join(shopper_msgs[-6:])
        try:
            round_id, results = await system.recommend(
                detected_color          = detected_color,
                detected_type           = detected_type,
                detected_body_type      = detected_body_type,
                detected_color_conf     = color_conf,
                detected_type_conf      = type_conf,
                detected_body_type_conf = body_conf,
                user_answer             = user_answer,
                user_gender             = gender,
                user_height_cm          = height_cm,
            )
        except Exception as exc:  # noqa: BLE001 — a failed round abandons the episode
            logger.warning(
                f"[experiment] recommend() failed on turn {turn_idx} "
                f"(customer={persona.get('id')}, combo={combo.name}): {exc}"
            )
            abandoned = True
            break

        last_recs = results
        last_round_id = round_id
        agent_weights = results[0].get("agent_weights", {}) if results else {}

        # The shopper reacts to the items it was just shown. The turn rows
        # (shopper_msg + weights + items) are written together AFTER the episode
        # row exists, so we buffer this turn in `history`. The recorded
        # `shopper_msg` is the answer that PRODUCED these recs (empty on the
        # first, camera-only round).
        #
        # Ask the shopper for its next line (run the blocking LLM off the loop).
        # It sees its own prior INDIVIDUAL lines (not the accumulated string) so
        # the prompt stays clean and non-repetitive.
        loop = asyncio.get_event_loop()
        shopper_history = [{"shopper_msg": m} for m in shopper_msgs]
        reaction = await loop.run_in_executor(
            None, shopper.next_message, persona, shopper_history, last_recs
        )
        history.append(
            {
                "shopper_msg": user_answer,
                "agent_weights": agent_weights,
                "items": results,
            }
        )

        if reaction.get("stop"):
            # Satisfied / gave up: this turn's recs are the final ones.
            break
        new_msg = reaction.get("message", "").strip()
        if new_msg:
            shopper_msgs.append(new_msg)
    else:
        # Loop exhausted MAX_TURNS without a stop — still a valid (capped) episode.
        pass

    # Closing review over the final recommendations. An abandoned episode (the
    # first round failed, so there are no recs to react to) records a NULL review
    # so it does not pollute the per-combo mean.
    if abandoned:
        rating = None
        reason = ""
    else:
        loop = asyncio.get_event_loop()
        shopper_history = [{"shopper_msg": m} for m in shopper_msgs]
        review = await loop.run_in_executor(
            None, shopper.final_review, persona, shopper_history, last_recs
        )
        rating = review.get("rating")
        reason = review.get("reason", "")

    # ── Learning-curve feed (curve mode only) ──────────────────────────────────
    # Push this episode's 1–5 review into the RL reward path: each of the final
    # top-K items gets the rating's reward attributed to its transition, then any
    # full rollout is drained for an explicit PPO update. A curve point ties the
    # review to the policy's cumulative update_count.
    if feed_review and not abandoned and rating is not None:
        rewards_landed = 0
        rewards_dropped = 0
        for item in last_recs[: config.TOP_K]:
            res = system.submit_feedback(
                round_id = last_round_id,
                item_id  = int(item["item_id"]),
                size     = str(item["size"]),
                rating   = int(rating),
            )
            if res.get("ok"):
                rewards_landed += 1
            else:
                rewards_dropped += 1
                logger.warning(
                    f"[experiment] feedback dropped (round={last_round_id}, "
                    f"item={item.get('item_id')}:{item.get('size')}): {res.get('reason')}"
                )

        # Drain every full rollout that the deferred store now holds, running one
        # PPO update per rollout. mean_return is the most recent update's value.
        mean_return = None
        while (batch := rl_store.drain_rollout()):
            stats = rl_policy.learn(batch)
            mean_return = stats.get("mean_return")

        store.add_curve_point(
            experiment_id   = experiment_id,
            episode_index   = episode_no,
            update_count    = rl_policy.update_count,
            rating          = int(rating),
            rewards_landed  = rewards_landed,
            rewards_dropped = rewards_dropped,
            mean_return     = mean_return,
        )

    # Persist the episode, then its turns/items.
    episode_id = store.create_episode(
        experiment_id = experiment_id,
        customer_id   = str(persona.get("id", "")),
        combo         = {"name": combo.name, "strategies": combo.strategies},
        repeat_idx    = repeat_idx,
        n_turns       = len(history),
        final_review  = rating,
        review_reason = reason,
        abandoned     = abandoned,
    )
    for turn_idx, turn in enumerate(history):
        store.add_turn(
            episode_id    = episode_id,
            idx           = turn_idx,
            shopper_msg   = turn.get("shopper_msg", ""),
            agent_weights = turn.get("agent_weights", {}),
            items         = turn.get("items", []),
        )

    logger.info(
        f"[experiment] episode done: customer={persona.get('id')} "
        f"combo={combo.name} repeat={repeat_idx} turns={len(history)} "
        f"review={rating} ({reason})"
    )
    return rating


# ── orchestration ──────────────────────────────────────────────────────────────

def _clock() -> str:
    """Wall-clock HH:MM:SS for stamping progress lines."""
    return datetime.now().strftime("%H:%M:%S")


def _fmt_dur(seconds: float) -> str:
    """Human-readable elapsed time, e.g. '1h 03m 12s' or '47.3s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    return str(timedelta(seconds=round(seconds)))


async def main() -> None:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    personas = _load_personas()

    # Run shape is env-configurable so the same entry point serves the quick OFAT
    # slice (default) and the full factorial grid, without editing code.
    #   EXPERIMENT_MODE    = ofat (default) | full
    #   EXPERIMENT_REPEATS = K replays per (customer × combo)   [default 1]
    #   EXPERIMENT_MODE    = ofat (default) | full | curve
    # "curve" runs many sequential baseline episodes, feeds each review into the
    # PPO reward path, and records a learning-curve point per episode.
    import os
    mode = (os.getenv("EXPERIMENT_MODE") or "ofat").strip().lower()
    repeats = int(os.getenv("EXPERIMENT_REPEATS") or "1")
    # Episode ordering. "persona-major" (default) reproduces today's order/numbering
    # exactly (customer → combo → repeat). "interleave" round-robins personas
    # (combo → repeat → customer) to de-confound the persona-major loop.
    order = (os.getenv("EXPERIMENT_ORDER") or "persona-major").strip().lower()
    feed_review = False
    if mode == "full":
        combos = list(full_factorial_combos())
        name = "full_factorial_grid"
    elif mode == "curve":
        combos = [baseline_combo()]
        name = "rl_learning_curve"
        feed_review = True
        # Defer consumption so each episode's review can land BEFORE its round is
        # consumed; the episode loop drains rollouts explicitly after feedback.
        rl_store.defer_consumption = True
    else:
        combos = list(ofat_combos())
        name = "ofat_vertical_slice"
    spec = ExperimentSpec(
        name         = name,
        repeats      = repeats,
        customer_ids = [p["id"] for p in personas],
        combos       = combos,
    )
    persona_by_id = {p["id"]: p for p in personas}

    store = ResultsStore()
    experiment_id = store.create_experiment(
        name    = spec.name,
        spec    = {
            "name": spec.name,
            "repeats": spec.repeats,
            "customer_ids": spec.customer_ids,
            "combos": [{"name": c.name, "strategies": c.strategies} for c in spec.combos],
        },
        git_sha = _git_sha(),
    )

    print("\n━━  Viralytics Multi-Agent Experiment Harness (Part E)  ━━")
    print("(XMPP broker must be running: docker compose up -d xmpp; Ollama reachable)\n")
    from multi_agent.experiments.shopper import OLLAMA_SHOPPER_MODEL
    print(f"Experiment #{experiment_id}: {spec.name}")
    print(f"  personas : {', '.join(spec.customer_ids)}")
    print(f"  combos   : {len(spec.combos)}")
    print(f"  repeats  : {spec.repeats}")
    print(f"  episodes : {len(spec.customer_ids) * len(spec.combos) * spec.repeats}")
    print(f"  shopper  : {OLLAMA_SHOPPER_MODEL}\n")

    system = RecommendationSystem()
    print(f"[{_clock()}] Starting agents…", flush=True)
    await system.start()
    print(f"[{_clock()}] All agents online.\n")

    total_episodes = len(spec.customer_ids) * len(spec.combos) * spec.repeats
    episode_no   = 0
    run_started  = time.monotonic()

    # Flat episode plan. The default ("persona-major") order is customer → combo →
    # repeat, byte-identical to the old nested loop (and its episode_no numbering).
    # "interleave" round-robins personas (combo → repeat → customer) so consecutive
    # episodes vary the persona rather than marching through one persona at a time.
    if order == "interleave":
        plan = [(cid, combo, r)
                for combo in spec.combos
                for r in range(spec.repeats)
                for cid in spec.customer_ids]
    else:
        plan = [(cid, combo, r)
                for cid in spec.customer_ids
                for combo in spec.combos
                for r in range(spec.repeats)]

    try:
        for customer_id, combo, repeat_idx in plan:
            persona = persona_by_id[customer_id]
            episode_no += 1
            snapshot = _apply_combo(combo)
            try:
                print(
                    f"[{_clock()}] ▸ ({episode_no}/{total_episodes}) "
                    f"customer={customer_id}  combo={combo.name}  "
                    f"repeat={repeat_idx} …",
                    flush=True,
                )
                ep_started = time.monotonic()
                rating = await _run_episode(
                    system, store, experiment_id,
                    persona, combo, repeat_idx,
                    feed_review = feed_review,
                    episode_no  = episode_no,
                )
                print(
                    f"[{_clock()}]    review = {rating}  "
                    f"(took {_fmt_dur(time.monotonic() - ep_started)})"
                )
            finally:
                _restore_strategies(snapshot)

        # ── Summary table ──────────────────────────────────────────────────────
        print("\n━━  Mean review per combo  ━━")
        print(f"  {'combo':<28} {'mean':>6} {'n':>4}")
        print(f"  {'-' * 28} {'-' * 6} {'-' * 4}")
        for label, mean, n in store.mean_review_per_combo(experiment_id):
            print(f"  {label:<28} {mean:>6.2f} {n:>4}")

        # ── Learning-curve feed summary ─────────────────────────────────────────
        if feed_review:
            points = store.curve_points(experiment_id)
            total_landed  = sum(int(p[3] or 0) for p in points)
            total_dropped = sum(int(p[4] or 0) for p in points)
            print("\n━━  RL learning-curve feed  ━━")
            print(f"  curve points : {len(points)}")
            print(f"  PPO updates  : {rl_policy.update_count}")
            print(f"  rewards landed  : {total_landed}")
            print(f"  rewards dropped : {total_dropped}")
            if total_dropped > 0:
                print(
                    "  ⚠ WARNING: some ratings were dropped (round consumed before "
                    "feedback, or disabled by RL_REWARD_MODE). Investigate before "
                    "trusting the curve."
                )
            print("  Plot: python -m multi_agent.experiments.plot_learning_curve")

    finally:
        await system.stop()
        store.close()
        total_elapsed = time.monotonic() - run_started
        print("\nAgents stopped. Results in multi_agent/experiments/results.db")
        print(
            f"[{_clock()}] Total run duration: {_fmt_dur(total_elapsed)} "
            f"for {episode_no} episode(s)"
            + (f"  (avg {_fmt_dur(total_elapsed / episode_no)}/episode)"
               if episode_no else "")
        )
        # Auto-generate the Markdown report (best-effort; even a partial/
        # interrupted run gets one since this is in `finally`).
        if episode_no and experiment_id and experiment_id > 0:
            try:
                from multi_agent.experiments.report import generate
                out = generate(experiment_id, None)
                print(f"[{_clock()}] Markdown report → {out}/  (open README.md)")
            except Exception as exc:
                print(f"[{_clock()}] (report generation skipped: {exc})")


if __name__ == "__main__":
    spade.run(main())
