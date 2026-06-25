"""
multi_agent/experiments/report.py
─────────────────────────────────
Turn a finished experiment (rows in ``results.db``) into a small set of
well-organised Markdown files.

    python -m multi_agent.experiments.report                # latest experiment
    python -m multi_agent.experiments.report 6              # a specific id
    python -m multi_agent.experiments.report --out docs/exp # custom out dir

Output (default ``multi_agent/experiments/reports/exp_<id>/``):
  * ``README.md``     — overview: metadata, totals, review distribution, the
                        headline winners (best/worst combo, best strategy per
                        agent), and links to the detail files.
  * ``combos.md``     — every combo ranked by mean review (mean / n / std / range).
  * ``by_agent.md``   — for each scorer agent, the marginal mean review per
                        strategy (averaged over all combos using it) + Δ vs that
                        agent's baseline strategy. This is the "which personality
                        wins" view.
  * ``by_persona.md`` — per-persona means and each persona's best/worst combo.

Read-only and stdlib-only (sqlite3 + statistics); safe to run any time after a
run, and re-runnable (overwrites the target dir's files).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional

from multi_agent.config import _DEFAULT_AGENT_STRATEGIES

_DB_PATH = Path(__file__).resolve().parent / "results.db"
_REPORTS_DIR = Path(__file__).resolve().parent / "reports"
_SCORERS = ("colour", "body", "clothing", "stock")


# ── data loading ──────────────────────────────────────────────────────────────

def _latest_experiment_id(db: sqlite3.Connection) -> Optional[int]:
    row = db.execute("SELECT MAX(experiment_id) FROM experiments").fetchone()
    return int(row[0]) if row and row[0] is not None else None


def _load(db: sqlite3.Connection, experiment_id: int) -> dict:
    exp = db.execute(
        "SELECT experiment_id, name, spec_json, git_sha, created_at "
        "FROM experiments WHERE experiment_id = ?",
        (experiment_id,),
    ).fetchone()
    if exp is None:
        raise SystemExit(f"No experiment with id={experiment_id} in {_DB_PATH}")

    rows = db.execute(
        "SELECT customer_id, combo_json, repeat_idx, n_turns, final_review, "
        "review_reason, abandoned FROM episodes WHERE experiment_id = ? "
        "ORDER BY episode_id",
        (experiment_id,),
    ).fetchall()

    episodes = []
    for customer_id, combo_json, repeat_idx, n_turns, review, reason, abandoned in rows:
        try:
            combo = json.loads(combo_json or "{}")
        except Exception:
            combo = {}
        episodes.append({
            "customer": customer_id,
            "combo": combo.get("name", "?"),
            "strategies": combo.get("strategies", {}),
            "repeat": repeat_idx,
            "n_turns": n_turns,
            "review": None if review is None else int(review),
            "reason": reason or "",
            "abandoned": bool(abandoned),
        })

    spec = {}
    try:
        spec = json.loads(exp[2] or "{}")
    except Exception:
        pass

    return {
        "id": exp[0], "name": exp[1], "spec": spec, "git_sha": exp[3] or "",
        "created_at": exp[4], "episodes": episodes,
    }


# ── stats helpers ─────────────────────────────────────────────────────────────

def _agg(reviews: list[int]) -> dict:
    """mean / n / std / min / max for a list of reviews (ignores None upstream)."""
    n = len(reviews)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": n,
        "mean": statistics.mean(reviews),
        "std": statistics.stdev(reviews) if n > 1 else 0.0,
        "min": min(reviews),
        "max": max(reviews),
    }


def _fmt(x, nd: int = 2) -> str:
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _combo_stats(episodes: list[dict]) -> list[tuple[str, dict]]:
    buckets: dict[str, list[int]] = {}
    for ep in episodes:
        if ep["review"] is not None:
            buckets.setdefault(ep["combo"], []).append(ep["review"])
    out = [(name, _agg(rs)) for name, rs in buckets.items()]
    out.sort(key=lambda r: (r[1]["mean"] if r[1]["mean"] is not None else -1), reverse=True)
    return out


def _agent_marginals(episodes: list[dict]) -> dict[str, list[tuple[str, dict]]]:
    """For each scorer agent → [(strategy, agg)] averaged over all combos using it."""
    out: dict[str, list[tuple[str, dict]]] = {}
    for agent in _SCORERS:
        buckets: dict[str, list[int]] = {}
        for ep in episodes:
            strat = ep["strategies"].get(agent)
            if strat is None or ep["review"] is None:
                continue
            buckets.setdefault(strat, []).append(ep["review"])
        ranked = [(s, _agg(rs)) for s, rs in buckets.items()]
        ranked.sort(key=lambda r: (r[1]["mean"] if r[1]["mean"] is not None else -1), reverse=True)
        out[agent] = ranked
    return out


def _persona_stats(episodes: list[dict]) -> list[tuple[str, dict]]:
    buckets: dict[str, list[int]] = {}
    for ep in episodes:
        if ep["review"] is not None:
            buckets.setdefault(ep["customer"], []).append(ep["review"])
    out = [(c, _agg(rs)) for c, rs in buckets.items()]
    out.sort(key=lambda r: (r[1]["mean"] if r[1]["mean"] is not None else -1), reverse=True)
    return out


# ── markdown rendering ────────────────────────────────────────────────────────

def _readme(data: dict, combos, marginals, personas) -> str:
    eps = data["episodes"]
    reviewed = [e["review"] for e in eps if e["review"] is not None]
    abandoned = sum(1 for e in eps if e["abandoned"])
    spec = data["spec"]
    when = datetime.fromtimestamp(data["created_at"]).strftime("%Y-%m-%d %H:%M:%S")

    dist = {k: sum(1 for r in reviewed if r == k) for k in range(1, 6)}
    dist_rows = "\n".join(
        f"| {k} | {dist[k]:>4} | {'█' * round(20 * dist[k] / max(len(reviewed), 1))} |"
        for k in range(5, 0, -1)
    )

    best_per_agent = "\n".join(
        f"| {agent} | **{rows[0][0]}** | {_fmt(rows[0][1]['mean'])} | "
        f"baseline=`{_DEFAULT_AGENT_STRATEGIES.get(agent, '?')}` |"
        for agent, rows in marginals.items() if rows
    )

    top5 = "\n".join(
        f"| {i+1} | `{name}` | {_fmt(s['mean'])} | {s['n']} |"
        for i, (name, s) in enumerate(combos[:5])
    )
    bottom5 = "\n".join(
        f"| {name} | {_fmt(s['mean'])} | {s['n']} |"
        for name, s in combos[-5:][::-1]
    )

    overall = _agg(reviewed)
    return f"""# Experiment #{data['id']} — {data['name']}

_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} from `results.db`._

## Metadata
| field | value |
|-------|-------|
| experiment_id | {data['id']} |
| name | {data['name']} |
| run mode | {spec.get('name', data['name'])} |
| repeats (K) | {spec.get('repeats', '?')} |
| personas | {', '.join(spec.get('customer_ids', [])) or '—'} |
| combos | {len(combos)} |
| git sha | `{data['git_sha'][:12] or '—'}` |
| started | {when} |

## Totals
| metric | value |
|--------|-------|
| episodes (rows) | {len(eps)} |
| reviewed | {len(reviewed)} |
| abandoned / NULL | {abandoned} / {len(eps) - len(reviewed) - abandoned} |
| overall mean review | **{_fmt(overall['mean'])}** (σ={_fmt(overall['std'])}) |
| review range | {_fmt(overall['min'])} … {_fmt(overall['max'])} |

### Review distribution
| review | count | |
|:------:|------:|:--|
{dist_rows}

## Headline — best strategy per agent (marginal)
Averaged over every combo in which that agent used the strategy. See
[`by_agent.md`](by_agent.md) for the full per-strategy tables.

| agent | best strategy | marginal mean | |
|-------|---------------|:------------:|--|
{best_per_agent}

## Top 5 combos
| rank | combo | mean | n |
|:----:|-------|:----:|:-:|
{top5}

## Bottom 5 combos
| combo | mean | n |
|-------|:----:|:-:|
{bottom5}

## Files
- [`combos.md`](combos.md) — all {len(combos)} combos ranked.
- [`by_agent.md`](by_agent.md) — marginal review per agent strategy (which personality wins).
- [`by_persona.md`](by_persona.md) — per-persona means and best/worst combo.

> ⚠️ Interpret with care: with K={spec.get('repeats', '?')} the per-combo n is small;
> small mean gaps may be noise. The marginal (`by_agent.md`) view has much larger
> n per cell and is the more reliable signal.
"""


def _combos_md(data: dict, combos) -> str:
    rows = "\n".join(
        f"| {i+1} | `{name}` | {_fmt(s['mean'])} | {s['n']} | {_fmt(s['std'])} | "
        f"{_fmt(s['min'])}–{_fmt(s['max'])} |"
        for i, (name, s) in enumerate(combos)
    )
    return f"""# Combos ranked — experiment #{data['id']}

All {len(combos)} combos, best mean review first. `std` is the sample standard
deviation across the combo's episodes; `range` is min–max review.

| rank | combo | mean | n | std | range |
|:----:|-------|:----:|:-:|:---:|:-----:|
{rows}

[← back to README](README.md)
"""


def _by_agent_md(data: dict, marginals) -> str:
    blocks = []
    for agent in _SCORERS:
        rows = marginals.get(agent, [])
        base = _DEFAULT_AGENT_STRATEGIES.get(agent)
        base_mean = next((s["mean"] for st, s in rows if st == base), None)
        lines = []
        for strat, s in rows:
            delta = (
                "—" if (base_mean is None or s["mean"] is None)
                else f"{s['mean'] - base_mean:+.2f}"
            )
            tag = " _(baseline)_" if strat == base else ""
            lines.append(
                f"| {strat}{tag} | {_fmt(s['mean'])} | {s['n']} | {_fmt(s['std'])} | {delta} |"
            )
        body = "\n".join(lines) or "| _(no data)_ | — | — | — | — |"
        blocks.append(
            f"### {agent}\nbaseline = `{base}`\n\n"
            f"| strategy | marginal mean | n | std | Δ vs baseline |\n"
            f"|----------|:------------:|:-:|:---:|:------------:|\n{body}\n"
        )
    return f"""# Marginal review per agent strategy — experiment #{data['id']}

For each scorer agent, the mean review across **every** combo in which the agent
used that strategy (the other three agents vary). Larger n per cell than the
per-combo view, so this is the cleaner "which personality helps" signal. Δ is
versus the agent's baseline strategy.

{"".join(blocks)}
[← back to README](README.md)
"""


def _by_persona_md(data: dict, personas) -> str:
    eps = data["episodes"]
    head = "\n".join(
        f"| {c} | {_fmt(s['mean'])} | {s['n']} | {_fmt(s['std'])} | {_fmt(s['min'])}–{_fmt(s['max'])} |"
        for c, s in personas
    )
    # best/worst combo per persona
    detail = []
    for c, _ in personas:
        c_eps = [e for e in eps if e["customer"] == c and e["review"] is not None]
        if not c_eps:
            continue
        best = max(c_eps, key=lambda e: e["review"])
        worst = min(c_eps, key=lambda e: e["review"])
        detail.append(
            f"- **{c}** — best `{best['combo']}` ({best['review']}), "
            f"worst `{worst['combo']}` ({worst['review']})"
        )
    return f"""# Per-persona results — experiment #{data['id']}

| persona | mean | n | std | range |
|---------|:----:|:-:|:---:|:-----:|
{head}

## Best / worst combo per persona
{chr(10).join(detail)}

[← back to README](README.md)
"""


# ── entry point ───────────────────────────────────────────────────────────────

def generate(experiment_id: Optional[int], out_dir: Optional[Path], db_path: Path = _DB_PATH) -> Path:
    db = sqlite3.connect(str(db_path))
    try:
        if experiment_id is None:
            experiment_id = _latest_experiment_id(db)
            if experiment_id is None:
                raise SystemExit(f"No experiments in {db_path}")
        data = _load(db, experiment_id)
    finally:
        db.close()

    combos = _combo_stats(data["episodes"])
    marginals = _agent_marginals(data["episodes"])
    personas = _persona_stats(data["episodes"])

    out = out_dir or (_REPORTS_DIR / f"exp_{experiment_id}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(_readme(data, combos, marginals, personas))
    (out / "combos.md").write_text(_combos_md(data, combos))
    (out / "by_agent.md").write_text(_by_agent_md(data, marginals))
    (out / "by_persona.md").write_text(_by_persona_md(data, personas))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Markdown report for an experiment.")
    ap.add_argument("experiment_id", nargs="?", type=int, default=None,
                    help="experiment id (default: latest)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: multi_agent/experiments/reports/exp_<id>/)")
    ap.add_argument("--db", type=Path, default=_DB_PATH, help="results.db path")
    args = ap.parse_args()
    out = generate(args.experiment_id, args.out, args.db)
    print(f"Report written to {out}/")
    for f in ("README.md", "combos.md", "by_agent.md", "by_persona.md"):
        print(f"  - {out / f}")


if __name__ == "__main__":
    main()
