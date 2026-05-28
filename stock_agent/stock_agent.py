from __future__ import annotations

import json
import os
import re
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
from stock_stats import StockStats, PIVOT_KEYS, QUERY_KEYS, RANGE_KEYS  # noqa: E402


def _field_match(df: pd.DataFrame, key: str, values: list[str]) -> pd.Series:
    """Boolean mask over df: True where df[key] matches any value in `values`.

    age_group is special-cased to case-insensitive substring (the column
    stores comma-separated lists like 'adult, young adult'). Every other
    key uses exact equality (isin).
    """
    if key == "age_group":
        pattern = "|".join(re.escape(v) for v in values)
        return df["age_group"].str.contains(pattern, case=False, regex=True, na=False)
    return df[key].isin(values)


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
- Each candidate also carries `style`, `pattern`, `material`, `gender`,
  `age_group`, `occasion`, `brand`, `price`. Use them to break ties or
  to spot items that suit the customer's likely intent.

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
        self, query: dict, n: int = 40
    ) -> list[tuple[int, str]]:
        """Tier-based attribute relaxation w/ multi-value include + exclude.

        Canonical query schema (matches the LLM query-parser output in
        LNIAGIA/query_parsing/llm_query_parser.py):
            {
              "include": {field: [value, ...]},   # any-of, contributes 1 to match_count
              "exclude": {field: [value, ...]},   # hard drop
              "price_min": float,                  # hard filter
              "price_max": float,                  # hard filter
            }

        Shorthand (back-compat): a flat dict with QUERY_KEYS + RANGE_KEYS
        is auto-wrapped to {"include": {k: [v], ...}, ...}. Mixing
        shorthand with explicit include/exclude raises ValueError.

        Equality keys (QUERY_KEYS): color, type, fit, size, style, pattern,
          material, gender, age_group, season, occasion, brand.
        age_group uses case-insensitive substring match (the column stores
        comma-separated values like 'adult, young adult').

        Filters always applied: active=1 AND stock_count>0.
        Tiebreaker within a match_count tier: item_id ASC.
        """
        include, exclude, price_min, price_max = self._normalize_query(query)

        df = self.stats.df
        mask = (df["active"] == 1) & (df["stock_count"] > 0)
        df = df.loc[mask].copy()

        if price_min is not None:
            df = df.loc[df["price"] >= price_min]
        if price_max is not None:
            df = df.loc[df["price"] <= price_max]

        # Apply exclude as hard filter BEFORE scoring
        for k, vals in exclude.items():
            df = df.loc[~_field_match(df, k, vals)]

        if df.empty:
            return []

        # Include scoring → match_count (any-of within a key, 1 per key)
        match_count = pd.Series(0, index=df.index)
        for k, vals in include.items():
            match_count = match_count + _field_match(df, k, vals).astype(int)
        df["__match_count"] = match_count

        df = df.sort_values(
            ["__match_count", "item_id"], ascending=[False, True]
        ).head(n)
        return list(zip(df["item_id"].tolist(), df["size"].tolist()))

    # ─── query normalization ──────────────────────────────────────────

    @staticmethod
    def _normalize_query(
        q: dict,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], float | None, float | None]:
        """Accept canonical {include,exclude,price_*} OR flat shorthand.
        Returns validated (include, exclude, price_min, price_max).
        """
        if not isinstance(q, dict):
            raise ValueError(f"query must be a dict, got {type(q).__name__}")

        has_inc_exc = ("include" in q) or ("exclude" in q)
        bare_keys = [
            k for k in q
            if k not in ("include", "exclude") and k not in RANGE_KEYS
        ]
        if has_inc_exc and bare_keys:
            raise ValueError(
                f"mix shorthand and canonical: bare keys {bare_keys} "
                f"alongside include/exclude. Pick one form."
            )

        if has_inc_exc:
            include = dict(q.get("include") or {})
            exclude = dict(q.get("exclude") or {})
        else:
            include = {k: q[k] for k in bare_keys}
            exclude = {}

        # Validate keys + list-wrap str values
        for d, label in ((include, "include"), (exclude, "exclude")):
            for k, v in list(d.items()):
                if k not in QUERY_KEYS:
                    raise ValueError(
                        f"unknown key {k!r} in {label}; allowed: {QUERY_KEYS}"
                    )
                if isinstance(v, str):
                    d[k] = [v]
                elif isinstance(v, (list, tuple)):
                    d[k] = [str(x) for x in v]
                else:
                    raise ValueError(
                        f"{label}[{k!r}] must be str or list, got {type(v).__name__}"
                    )
                if not d[k]:
                    raise ValueError(f"{label}[{k!r}] is empty list")

        # Range keys
        price_min = price_max = None
        if "price_min" in q:
            try:
                price_min = float(q["price_min"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"price_min must be numeric, got {q['price_min']!r}"
                )
        if "price_max" in q:
            try:
                price_max = float(q["price_max"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"price_max must be numeric, got {q['price_max']!r}"
                )

        if not include and not exclude and price_min is None and price_max is None:
            raise ValueError(
                "query is empty; provide include / exclude / price_min / price_max"
            )

        return include, exclude, price_min, price_max

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
            price = row.get("price")
            try:
                price_val = round(float(price), 2)
            except (TypeError, ValueError):
                price_val = None
            rows.append({
                "item_id": int(iid),
                "size": sz,
                "color": row["color"],
                "type": row["type"],
                "fit": row["fit"],
                "season": row["season"],
                "style": row["style"],
                "pattern": row["pattern"],
                "material": row["material"],
                "gender": row["gender"],
                "age_group": row["age_group"],
                "occasion": row["occasion"],
                "brand": row["brand"],
                "price": price_val,
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


def _parse_query(tokens: list[str]) -> dict:
    """Parse REPL tokens into a canonical query dict.

    Token forms:
      key=v             -> include[key] = [v]
      key=v1,v2         -> include[key] = [v1, v2]
      +key=v[,v2]       -> explicit include (same as key=v)
      -key=v[,v2]       -> exclude[key] = [v, ...]
      price_min=20      -> top-level price_min (passed through)
      price_max=80      -> top-level price_max (passed through)

    Multiple tokens for the same key merge values. +key and -key may
    coexist (different sets), but the same value in both raises.
    """
    include: dict[str, list[str]] = {}
    exclude: dict[str, list[str]] = {}
    extras: dict = {}

    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"bad token {tok!r}; expected k=v")
        lhs, rhs = tok.split("=", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if not lhs or not rhs:
            raise ValueError(f"empty key or value in {tok!r}")

        # Prefix detection
        if lhs.startswith("+"):
            target, key = include, lhs[1:].strip()
        elif lhs.startswith("-"):
            target, key = exclude, lhs[1:].strip()
        elif lhs in RANGE_KEYS:
            extras[lhs] = rhs
            continue
        else:
            target, key = include, lhs

        if not key:
            raise ValueError(f"missing key after prefix in {tok!r}")

        values = [v.strip() for v in rhs.split(",") if v.strip()]
        if not values:
            raise ValueError(f"no values for {tok!r}")
        target.setdefault(key, [])
        for v in values:
            if v not in target[key]:
                target[key].append(v)

    # Catch contradictions: same value in include AND exclude for the same key
    for k, inc_vals in include.items():
        if k in exclude:
            clash = set(inc_vals) & set(exclude[k])
            if clash:
                raise ValueError(
                    f"value(s) {sorted(clash)} appear in BOTH +{k} and -{k}"
                )

    out: dict = {}
    if include:
        out["include"] = include
    if exclude:
        out["exclude"] = exclude
    out.update(extras)
    return out


def _print_candidates(agent: StockAgent, cands: list[tuple[int, str]], query: dict) -> None:
    # Recompute per-row match_count for the display (canonical query: count
    # include axes where any listed value matches the row).
    include = query.get("include") or {}
    print(f"\n{len(cands)} candidates (sorted by match_count DESC, item_id ASC):")
    for iid, sz in cands[:20]:
        row = agent.stats.get_row(iid, sz)
        mc = 0
        for k, vals in include.items():
            if k == "age_group":
                if any(str(v).lower() in str(row[k]).lower() for v in vals):
                    mc += 1
            elif row[k] in vals:
                mc += 1
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
