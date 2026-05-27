from __future__ import annotations

import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

import ollama

# Sibling import without an __init__.py — same pattern stock_stats.py uses.
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from stock_stats import StockStats, PIVOT_KEYS  # noqa: E402


DEFAULT_OLLAMA_MODEL = (
    os.getenv("OLLAMA_STOCK_AGENT_MODEL")
    or os.getenv("OLLAMA_MODEL")
    or "qwen2.5:7b-instruct-q3_K_M"
)


# ─── season helper ──────────────────────────────────────────────────────

_MONTH_TO_SEASON_NORTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


def _current_season(today: datetime | None = None) -> str:
    today = today or datetime.now()
    return _MONTH_TO_SEASON_NORTH[today.month]


# ─── LLM prompt ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """You are the stock manager of a clothing store.

A customer query produced {n_candidates} candidate items. Some match the
customer's wishes perfectly, others match partially. Your job is to pick
the **top {k}** items the store should recommend.

Weigh these store-side priorities:

- **Clear old stock** — older items (high `age_days`) and high-stagnation
  items (large `days_since_last_sale`) are preferred for pushing.
- **Avoid pushing fast movers** — items with high `sales_velocity`
  (units sold per day) sell themselves; rank them lower.
- **Prefer current season** — items whose `season` matches "{current_season}"
  (or are "all-season") are slightly preferred, all else equal.
- **Reasonable stock** — don't push items with very low `stock_count`
  (risk running out before customer arrives).
- `push_score` is a precomputed combined signal (higher = stronger
  candidate to push). Treat as ONE input among the above, not the only one.

Reply with a JSON object using EXACTLY this schema. No prose, no markdown,
no extra fields:

{{"picks": [{{"item_id": <int>, "size": "<S>"}}, ...]}}

Constraints:
- Exactly {k} entries in `picks`, in priority order.
- Each entry has ONLY `item_id` (int) and `size` (string).
- Every (item_id, size) MUST be one of the candidates I sent you.
- Do not invent items, do not include any other fields, do not wrap in
  markdown fences."""


# ─── StockAgent ─────────────────────────────────────────────────────────


class StockAgent:
    """Phase 1 standalone StockAgent.

    Three responsibilities:
      1. get_candidates  — tier-based attribute relaxation
      2. rate            — push_score per candidate (private rating)
      3. pick_top        — LLM stock-manager picks top-k
    No SPADE / negotiate yet (Phase 2).
    """

    def __init__(
        self,
        stats: StockStats | None = None,
        *,
        model: str = DEFAULT_OLLAMA_MODEL,
    ) -> None:
        self.stats = stats or StockStats()
        self.model = model

    # ─── 1. Retrieve 40 candidates ──────────────────────────────────────

    def get_candidates(
        self, query: dict[str, str], n: int = 40
    ) -> list[tuple[int, str]]:
        """Tier-based attribute relaxation: items matching all params first,
        then matches-except-1, then matches-except-2, etc., until `n` collected.

        query keys: any subset of color/type/fit/size. Unknown keys raise.
        Filters: active=1 AND stock_count>0.
        Tiebreaker within a tier: item_id ASC (neutral, deterministic).
        """
        bad = [k for k in query if k not in PIVOT_KEYS]
        if bad:
            raise ValueError(f"unknown query keys: {bad}; allowed: {PIVOT_KEYS}")
        if not query:
            raise ValueError("query must contain at least one of color/type/fit/size")

        df = self.stats.df
        mask = (df["active"] == 1) & (df["stock_count"] > 0)
        df = df.loc[mask].copy()

        match_count = pd.Series(0, index=df.index)
        for k, v in query.items():
            match_count = match_count + (df[k] == v).astype(int)
        df["__match_count"] = match_count

        df = df.sort_values(
            ["__match_count", "item_id"], ascending=[False, True]
        ).head(n)
        return list(zip(df["item_id"].tolist(), df["size"].tolist()))

    # ─── 2. Rate (NOT a vote — phase 2 reserves "vote") ─────────────────

    def rate(
        self, candidates: Iterable[tuple[int, str]]
    ) -> dict[tuple[int, str], float]:
        """StockAgent's private rating per (item, size). Returns push_scores.
        Missing keys (not in stock) → 0.0. Not a vote.
        """
        return self.stats.get_push_scores(candidates, missing=0.0)

    # ─── 3. LLM picks top-k ────────────────────────────────────────────

    def pick_top(
        self,
        candidates: list[tuple[int, str]],
        k: int = 10,
        *,
        verbose: bool = False,
    ) -> list[tuple[int, str]]:
        """LLM stock-manager picks top-k items from `candidates`.

        Raises RuntimeError if Ollama is unreachable or returns invalid output.
        """
        if not candidates:
            raise ValueError("candidates is empty")
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if k > len(candidates):
            raise ValueError(
                f"k={k} larger than candidates={len(candidates)}"
            )

        candidate_set = set(candidates)
        table = self._candidate_table(candidates)
        system = _SYSTEM_PROMPT_TEMPLATE.format(
            n_candidates=len(candidates),
            k=k,
            current_season=_current_season(),
        )
        user = (
            "Candidate items (JSON list):\n"
            f"{table}\n\n"
            f"Pick top {k} as JSON per the schema above."
        )

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"temperature": 0},
                format="json",  # Ollama strict JSON mode
            )
        except Exception as exc:
            raise RuntimeError(
                f"Ollama call failed (model={self.model!r}): {exc}"
            ) from exc

        raw = response["message"]["content"].strip()
        if verbose:
            print(f"\n[LLM raw]\n{raw}\n")

        # Defensive fence strip — format="json" should prevent this but some
        # small models still leak markdown.
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"LLM returned invalid JSON: {exc}\nraw: {raw[:500]}"
            ) from exc

        # Accept either {"picks": [...]} or a bare [...] or any dict whose
        # value is a list — small models invent their own key name + drop
        # the schema. Extra fields per entry are ignored downstream.
        picks_raw = None
        if isinstance(parsed, list):
            picks_raw = parsed
        elif isinstance(parsed, dict):
            for candidate_key in ("picks", "items", "top", "top_items", "result"):
                v = parsed.get(candidate_key)
                if isinstance(v, list):
                    picks_raw = v
                    break
            if picks_raw is None:
                # last resort: take any list value from the dict
                list_values = [v for v in parsed.values() if isinstance(v, list)]
                if len(list_values) == 1:
                    picks_raw = list_values[0]

        if not isinstance(picks_raw, list):
            raise RuntimeError(
                f"LLM JSON has no picks list. Got: {parsed!r}"
            )

        # Validate, dedupe, restrict to candidate_set, cap at k
        seen: set[tuple[int, str]] = set()
        picks: list[tuple[int, str]] = []
        for entry in picks_raw:
            if not isinstance(entry, dict):
                continue
            iid = entry.get("item_id")
            sz = entry.get("size")
            if not isinstance(iid, int) or not isinstance(sz, str):
                continue
            key = (iid, sz)
            if key in seen or key not in candidate_set:
                continue
            seen.add(key)
            picks.append(key)
            if len(picks) == k:
                break

        if len(picks) < k:
            raise RuntimeError(
                f"LLM returned only {len(picks)}/{k} valid picks "
                f"(after dedup + candidate-set validation). raw: {raw[:500]}"
            )

        return picks

    # ─── candidate context table for the LLM ────────────────────────────

    def _candidate_table(self, candidates: list[tuple[int, str]]) -> str:
        """Compact JSON list of dicts, one per candidate, for the LLM."""
        rows = []
        for iid, sz in candidates:
            try:
                row = self.stats.get_row(iid, sz)
            except KeyError:
                continue
            rows.append({
                "item_id": int(iid),
                "size": sz,
                "color": row["color"],
                "type": row["type"],
                "fit": row["fit"],
                "season": row["season"],
                "stock_count": int(row["stock_count"]),
                "total_sold": int(row["total_sold"]),
                "age_days": round(float(row["age_days"]), 1),
                "days_since_last_sale": round(float(row["days_since_last_sale"]), 1),
                "sales_velocity": round(float(row["sales_velocity"]), 4),
                "push_score": round(float(row["push_score"]), 4),
            })
        return json.dumps(rows, indent=None)


# ─── REPL ──────────────────────────────────────────────────────────────


_HELP = """\
Commands:
  query k=v [k=v ...]   set the structured query (keys: color, type, fit, size)
  candidates [n]        fetch + show top-n (default 40) candidates with match_count
  rate                  show push_score for each cached candidate
  pick [k]              LLM picks top-k (default 10) from cached candidates
  state                 print current query + cached count
  help                  this help
  exit / quit           leave REPL"""


def _parse_query(tokens: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"bad token {tok!r}; expected k=v")
        k, v = tok.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _print_candidates(agent: StockAgent, cands: list[tuple[int, str]], query: dict) -> None:
    print(f"\n{len(cands)} candidates (sorted by match_count DESC, item_id ASC):")
    for iid, sz in cands[:20]:
        row = agent.stats.get_row(iid, sz)
        mc = sum(1 for k, v in query.items() if row[k] == v)
        print(
            f"  ({iid:>5}, {sz:<3}) match={mc}  "
            f"stock={int(row['stock_count']):>3} sold={int(row['total_sold']):>4} "
            f"age={row['age_days']:>6.1f}d  "
            f"color={row['color']:<10} type={row['type']:<22} fit={row['fit']}"
        )
    if len(cands) > 20:
        print(f"  ... ({len(cands) - 20} more)")


def _print_rate(agent: StockAgent, cands: list[tuple[int, str]]) -> None:
    ratings = agent.rate(cands)
    sorted_ratings = sorted(ratings.items(), key=lambda kv: -kv[1])
    print(f"\nrate() — push_scores for {len(cands)} candidates (top 20):")
    for (iid, sz), score in sorted_ratings[:20]:
        print(f"  ({iid:>5}, {sz:<3})  push={score:.4f}")
    if len(sorted_ratings) > 20:
        print(f"  ... ({len(sorted_ratings) - 20} more)")


def _print_pick(agent: StockAgent, cands: list[tuple[int, str]], k: int) -> None:
    print(f"\nLLM picking top {k} (model={agent.model})...")
    picks = agent.pick_top(cands, k=k)
    print(f"\nTop {k}:")
    for rank, (iid, sz) in enumerate(picks, 1):
        row = agent.stats.get_row(iid, sz)
        print(
            f"  {rank:>2}. ({iid:>5}, {sz:<3})  "
            f"push={row['push_score']:.3f}  stock={int(row['stock_count']):>3}  "
            f"sold={int(row['total_sold']):>4}  age={row['age_days']:>6.1f}d  "
            f"season={row['season']:<10} color={row['color']:<10} type={row['type']}"
        )


def repl(agent: StockAgent) -> None:
    print(f"StockAgent REPL — model={agent.model}, "
          f"loaded {len(agent.stats.df)} rows.")
    print(f"current_season = {_current_season()}")
    print("Type 'help' for commands.\n")

    query: dict[str, str] = {}
    cands: list[tuple[int, str]] = []

    while True:
        try:
            line = input("stock> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue

        try:
            tokens = shlex.split(line)
        except ValueError as exc:
            print(f"parse error: {exc}")
            continue
        cmd, args = tokens[0], tokens[1:]

        try:
            if cmd in ("exit", "quit"):
                return
            elif cmd == "help":
                print(_HELP)
            elif cmd == "state":
                print(f"query={query}  cached_candidates={len(cands)}")
            elif cmd == "query":
                query = _parse_query(args)
                print(f"query set: {query}")
                cands = []
            elif cmd == "candidates":
                if not query:
                    print("set a query first (e.g. query color=red size=M)")
                    continue
                n = int(args[0]) if args else 40
                cands = agent.get_candidates(query, n=n)
                _print_candidates(agent, cands, query)
            elif cmd == "rate":
                if not cands:
                    print("no cached candidates; run `candidates` first")
                    continue
                _print_rate(agent, cands)
            elif cmd == "pick":
                if not cands:
                    print("no cached candidates; run `candidates` first")
                    continue
                k = int(args[0]) if args else 10
                _print_pick(agent, cands, k)
            else:
                print(f"unknown command: {cmd}. type 'help'.")
        except Exception as exc:
            print(f"error: {exc}")


def main() -> int:
    try:
        agent = StockAgent()
    except FileNotFoundError as exc:
        print(f"Missing file: {exc}", file=sys.stderr)
        return 1
    repl(agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
