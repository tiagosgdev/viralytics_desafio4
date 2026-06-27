"""
multi_agent/experiments/plot_learning_curve.py
───────────────────────────────────────────────
Plot the RL learning curve captured by the ``curve`` experiment mode: each
episode's simulated 1–5 review against the policy's cumulative PPO update_count.

Run from repo root (latest experiment with curve points if no id given):

    python -m multi_agent.experiments.plot_learning_curve [experiment_id]

Read-only on ``results.db``. Matches the matplotlib style of
``stock_agent/sanity_plots.py`` (savefig dpi=110, bbox_inches="tight",
plt.close). The rolling mean is computed manually (numpy) so pandas is not
required. Output: ``multi_agent/experiments/plots/rl_learning_curve.png``.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / GPU box — no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "results.db"
PLOTS_DIR = BASE_DIR / "plots"
ROLL_WINDOW = 15


def _latest_experiment_id(con: sqlite3.Connection) -> int | None:
    """Most recent experiment_id that actually has curve_points, else None."""
    row = con.execute(
        "SELECT experiment_id FROM curve_points "
        "GROUP BY experiment_id ORDER BY MAX(rowid) DESC LIMIT 1"
    ).fetchone()
    return int(row[0]) if row else None


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling mean with a shrinking window at the start (no NaNs)."""
    out = np.empty(len(values), dtype=float)
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out[i] = float(np.mean(values[lo : i + 1]))
    return out


def main() -> None:
    experiment_id = int(sys.argv[1]) if len(sys.argv) > 1 else None

    if not DB_PATH.exists():
        print(f"No results DB at {DB_PATH}; run the curve experiment first.")
        return

    con = sqlite3.connect(str(DB_PATH))
    try:
        if experiment_id is None:
            experiment_id = _latest_experiment_id(con)
        if experiment_id is None:
            print("No curve_points found in results.db; run EXPERIMENT_MODE=curve first.")
            return

        rows = con.execute(
            "SELECT episode_index, update_count, rating, rewards_landed, "
            "rewards_dropped, mean_return FROM curve_points "
            "WHERE experiment_id = ? ORDER BY episode_index",
            (experiment_id,),
        ).fetchall()
    finally:
        con.close()

    if not rows:
        print(f"Experiment {experiment_id} has no curve_points.")
        return

    episode_index = np.array([r[0] for r in rows], dtype=float)
    update_count = np.array([r[1] for r in rows], dtype=float)
    rating = np.array([float(r[2]) if r[2] is not None else np.nan for r in rows], dtype=float)
    total_dropped = sum(int(r[4] or 0) for r in rows)

    # Drop episodes without a numeric rating (abandoned / NULL) from the curve.
    keep = ~np.isnan(rating)
    episode_index, update_count, rating = episode_index[keep], update_count[keep], rating[keep]
    if len(rating) == 0:
        print(f"Experiment {experiment_id}: no rated episodes to plot.")
        return

    # X axis: cumulative PPO updates if the policy actually updated, else episodes.
    if float(update_count.max()) > 0:
        x = update_count
        x_label = "cumulative PPO updates"
    else:
        x = episode_index
        x_label = "episode index (no PPO updates recorded)"

    smooth = _rolling_mean(rating, ROLL_WINDOW)

    # Early (first 25%) vs late (last 25%) mean review delta.
    n = len(rating)
    q = max(1, n // 4)
    early_mean = float(np.mean(rating[:q]))
    late_mean = float(np.mean(rating[-q:]))
    delta = late_mean - early_mean

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, rating, ".", color="lightsteelblue", alpha=0.6, label="review (raw)")
    ax.plot(x, smooth, "-", color="steelblue", linewidth=2.0,
            label=f"rolling mean (window={ROLL_WINDOW})")
    ax.axhline(early_mean, color="darkorange", linestyle="--", linewidth=1.0,
               label=f"early mean = {early_mean:.2f}")
    ax.axhline(late_mean, color="seagreen", linestyle="--", linewidth=1.0,
               label=f"late mean = {late_mean:.2f}")
    ax.set_title(
        f"RL learning curve (experiment #{experiment_id}) — "
        f"Δreview(late−early) = {delta:+.2f}"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("simulated review (1–5)")
    ax.set_ylim(0.8, 5.2)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "rl_learning_curve.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"Experiment #{experiment_id}: {n} rated episodes, {len(rows)} curve points.")
    print(f"  early(first 25%) mean review = {early_mean:.3f}")
    print(f"  late (last 25%)  mean review = {late_mean:.3f}")
    print(f"  Δ (late − early)             = {delta:+.3f}")
    print(f"  total rewards_dropped        = {total_dropped}"
          + ("  ⚠ (should be 0)" if total_dropped else ""))
    print(f"  plot → {out}")


if __name__ == "__main__":
    main()
