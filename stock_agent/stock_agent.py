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


_TOKEN_ALLOCATION_SCHEMA = {
    "type": "object",
    "properties": {
        "allocation": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "integer"},
                    "size": {"type": "string"},
                    "tokens": {"type": "integer", "minimum": 1},
                },
                "required": ["item_id", "size", "tokens"],
            },
        }
    },
    "required": ["allocation"],
}


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

A customer query produced {n_candidates} candidate items. You have a
budget of EXACTLY {total_tokens} TOKENS (not more, not less, total
across the whole allocation). Distribute these {total_tokens} tokens
across the candidates to express how strongly you want each item pushed.

Token allocation rules (STRICT):
- The SUM of `tokens` across ALL entries in your allocation MUST equal
  {total_tokens}. Example: if {total_tokens}=10 you might output one
  entry with tokens=10, or two entries with tokens=7 and tokens=3, or
  ten entries with tokens=1 each — but the sum is always {total_tokens}.
- Each listed entry needs at least 1 token. Don't list items with 0.
- Multiple tokens on the same (item_id, size) pair = stronger preference.
- The `item_id` and `size` MUST come VERBATIM from the candidate list
  I send you below. DO NOT invent item_ids. DO NOT list sizes the
  candidate row doesn't have.

Weigh these store-side priorities:

- **Clear old stock** — older items (high `age_days`) and high-stagnation
  items (large `days_since_last_sale`) deserve more tokens.
- **Avoid fast movers** — items with high `sales_velocity` sell themselves;
  spend fewer tokens on them.
- **Prefer current season** — items whose `season` matches "{current_season}"
  (or are "all-season") earn an edge.
- **Reasonable stock** — don't load up items with very low `stock_count`
  (runout risk before customer arrives).
- `push_score` is a precomputed combined signal — strong input but not
  the only one. Integrate with the other dimensions.
- Other attributes (style, pattern, material, gender, age_group, occasion,
  brand, price, body_type) for tie-breaking.

Reply with a JSON object using EXACTLY this schema. No prose, no markdown,
no extra fields:

{{"allocation": [{{"item_id": <int>, "size": "<S>", "tokens": <int>}}, ...]}}

Constraints (will be validated):
- sum of `tokens` across entries == {total_tokens}
- Each entry is a FLAT object with ONLY three keys: `item_id` (int),
  `size` (string), `tokens` (int ≥ 1). No nesting, no extra fields.
- Every (item_id, size) MUST be one of the candidates I sent you."""


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
        # LLM knobs — loaded from the same stock_config.json the stats use
        with open(self.stats.config_path) as fp:
            _cfg = json.load(fp)
        self.llm_temperature = float(_cfg.get("llm", {}).get("temperature", 0.5))

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

        # Defensive copy — earlier .loc[...] chains may return a view of
        # self.stats.df; the upcoming column assignment must not leak.
        df = df.copy()

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

        # First: reject typo'd top-level keys (anything not canonical AND
        # not a valid shorthand key). Catches `price_mn` etc. before the
        # mix-vs-canonical heuristic produces a misleading error.
        _CANONICAL_TOP = {"include", "exclude", *RANGE_KEYS}
        bad_top = [
            k for k in q
            if k not in _CANONICAL_TOP and k not in QUERY_KEYS
        ]
        if bad_top:
            raise ValueError(
                f"unknown top-level key(s): {bad_top}; "
                f"allowed canonical: {sorted(_CANONICAL_TOP)}; "
                f"allowed shorthand: {sorted(QUERY_KEYS)}"
            )

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

    # ─── 3. LLM allocates token budget ─────────────────────────────────

    def allocate_tokens(
        self,
        candidates: list[tuple[int, str]],
        total_tokens: int = 10,
        *,
        verbose: bool = False,
    ) -> list[dict]:
        """LLM stock-manager distributes `total_tokens` (default 10) across
        candidates per stock-manager directives.

        Returns:
            list of dicts: {"item_id": int, "size": str, "tokens": int}
            — only items receiving ≥1 token; sum of tokens == total_tokens.
            Items not appearing in the result implicitly receive 0 tokens.

        Multiple tokens per item allowed (concentration → stronger preference).

        Non-deterministic: temperature from stock_config.json
        (default 0.5) → allocations vary across calls.

        Phase 2 Coordinator multiplies each agent's tokens by the agent's
        vote-weight to determine final winners. Per-agent weight is owned
        by the Coordinator, not exposed here.

        Raises RuntimeError if Ollama is unreachable or returns no valid output.
        """
        if not candidates:
            raise ValueError("candidates is empty")
        if total_tokens <= 0:
            raise ValueError(f"total_tokens must be > 0, got {total_tokens}")

        candidate_set = set(candidates)
        table = self._candidate_table(candidates)
        system = _SYSTEM_PROMPT_TEMPLATE.format(
            n_candidates=len(candidates),
            total_tokens=total_tokens,
            current_season=_current_season(),
        )
        user = (
            "Candidate items (JSON list):\n"
            f"{table}\n\n"
            f"Distribute {total_tokens} tokens per the schema above."
        )

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"temperature": self.llm_temperature},
                format=_TOKEN_ALLOCATION_SCHEMA,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Ollama call failed (model={self.model!r}): {exc}"
            ) from exc

        raw = response["message"]["content"].strip()
        if verbose:
            print(f"\n[LLM raw]\n{raw}\n")

        # Defensive fence strip
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"LLM returned invalid JSON: {exc}\nraw: {raw[:500]}"
            ) from exc

        # Permissive: accept {"allocation": [...]}, bare list, or any
        # single-list-value dict.
        alloc_raw = None
        if isinstance(parsed, list):
            alloc_raw = parsed
        elif isinstance(parsed, dict):
            for key in ("allocation", "picks", "items", "tokens", "result"):
                v = parsed.get(key)
                if isinstance(v, list):
                    alloc_raw = v
                    break
            if alloc_raw is None:
                list_values = [v for v in parsed.values() if isinstance(v, list)]
                if len(list_values) == 1:
                    alloc_raw = list_values[0]
        if not isinstance(alloc_raw, list):
            raise RuntimeError(
                f"LLM JSON has no allocation list. Got: {parsed!r}"
            )

        # Validate + merge duplicates (sum tokens if same key appears twice)
        merged: dict[tuple[int, str], int] = {}
        for entry in alloc_raw:
            if not isinstance(entry, dict):
                continue
            if "item_id" not in entry and len(entry) == 1:
                inner = next(iter(entry.values()))
                if isinstance(inner, dict) and "item_id" in inner:
                    entry = inner
            iid = entry.get("item_id")
            sz = entry.get("size")
            tok = entry.get("tokens")
            if not isinstance(iid, int) or not isinstance(sz, str):
                continue
            try:
                tok = int(tok)
            except (TypeError, ValueError):
                continue
            if tok <= 0:
                continue
            key = (iid, sz)
            if key not in candidate_set:
                continue
            merged[key] = merged.get(key, 0) + tok

        total = sum(merged.values())
        if total == 0:
            # LLM returned no valid items (hallucinated IDs / wrong sizes).
            # Fall back to a deterministic push_score-weighted allocation
            # so the agent always returns something useful to the
            # Coordinator. Log to stderr.
            print(
                f"warn: LLM allocation had no in-set entries; "
                f"falling back to push_score-weighted distribution. "
                f"raw: {raw[:200]}",
                file=sys.stderr,
            )
            merged = self._push_score_allocation(candidates, total_tokens)
            total = sum(merged.values())

        # Renormalize to exactly total_tokens if the LLM over/undershot.
        if total != total_tokens:
            scaled = {k: v * total_tokens / total for k, v in merged.items()}
            merged = {k: max(1, int(round(v))) for k, v in scaled.items()}
            diff = total_tokens - sum(merged.values())
            if diff != 0:
                # Adjust the largest entry by the residual; clamp at 1
                top_key = max(merged, key=merged.get)
                merged[top_key] = max(1, merged[top_key] + diff)
                # If applying the diff still drifts (because of the clamp),
                # repeat with the next-largest until balanced or give up.
                # In practice with small total_tokens this rarely matters.
                if sum(merged.values()) != total_tokens:
                    # Fallback: trim/extend on the largest entry without clamping
                    top_key = max(merged, key=merged.get)
                    merged[top_key] += total_tokens - sum(merged.values())
                    if merged[top_key] < 1:
                        # Pathological — drop it
                        del merged[top_key]

        # Sort desc by tokens, then item_id asc
        return sorted(
            (
                {"item_id": iid, "size": sz, "tokens": int(t)}
                for (iid, sz), t in merged.items()
            ),
            key=lambda d: (-d["tokens"], d["item_id"]),
        )

    # ─── deterministic fallback allocation ───────────────────────────────

    def _push_score_allocation(
        self,
        candidates: list[tuple[int, str]],
        total_tokens: int,
    ) -> dict[tuple[int, str], int]:
        """Push-score weighted token distribution. Used as a deterministic
        fallback when the LLM returns nothing usable.

        Allocation is proportional to push_score, rounded to integers, with
        residual added to the highest-push entry. Items with push_score == 0
        receive 0 tokens.
        """
        scores = self.stats.get_push_scores(candidates, missing=0.0)
        # Sort desc and keep only positive scores
        positive = [(k, v) for k, v in scores.items() if v > 0]
        if not positive:
            # All zero (e.g. inactive rows) — give all tokens to the first candidate
            return {candidates[0]: total_tokens}
        positive.sort(key=lambda kv: (-kv[1], kv[0]))
        total_score = sum(v for _, v in positive)
        merged: dict[tuple[int, str], int] = {}
        for key, score in positive:
            tok = int(round(score / total_score * total_tokens))
            if tok > 0:
                merged[key] = tok
        # Patch sum drift on the highest-score entry
        diff = total_tokens - sum(merged.values())
        if diff != 0:
            top_key = positive[0][0]
            merged[top_key] = max(1, merged.get(top_key, 0) + diff)
        # If all entries rounded down to 0, slot 1 token onto the top one
        if not merged:
            merged[positive[0][0]] = total_tokens
        return merged

    # ─── candidate context table for the LLM ────────────────────────────

    def _candidate_table(self, candidates: list[tuple[int, str]]) -> str:
        """Compact JSON list of dicts, one per candidate, for the LLM."""
        rows = []
        for iid, sz in candidates:
            try:
                row = self.stats.get_row(iid, sz)
            except KeyError:
                print(
                    f"warn: candidate ({iid},{sz}) missing from stats; skipping",
                    file=sys.stderr,
                )
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
                "body_type": row.get("body_type"),
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
  query TOKEN [TOKEN ...]   set the structured query (tokens below)
  candidates [n]            fetch + show top-n (default 40) candidates with match_count
  rate                      show push_score for each cached candidate
  allocate [N]              LLM distributes N (default 10) tokens across cached candidates
  state                     print current query + cached count
  reload                    re-read DB into StockStats (after external sell/restock)
  help                      this help
  exit / quit               leave REPL

Query tokens:
  key=v                     include key=v
  key=v1,v2                 include key with multiple values (any-of)
  +key=v[,v2]               explicit include (same as bare key=v)
  -key=v[,v2]               exclude values (hard filter)
  price_min=N / price_max=N numeric range (no +/- prefix)

Equality keys: color, type, fit, size, style, pattern, material,
               gender, age_group, season, occasion, brand.
age_group uses case-insensitive substring (column stores "adult, young adult" etc.).
See README §5 for the full canonical schema + value enums."""


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

        # Range keys handled first — no +/- prefix allowed
        if lhs in RANGE_KEYS:
            extras[lhs] = rhs
            continue
        if lhs.startswith(("+-", "-+")):
            raise ValueError(f"double prefix in {tok!r}")
        if lhs.startswith("+"):
            target, key = include, lhs[1:].strip()
        elif lhs.startswith("-"):
            target, key = exclude, lhs[1:].strip()
        else:
            target, key = include, lhs

        if not key:
            raise ValueError(f"missing key after prefix in {tok!r}")
        if key in RANGE_KEYS:
            raise ValueError(
                f"range key {key!r} does not take a +/- prefix; "
                f"use bare {key}=N"
            )

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


def _print_allocation(agent: StockAgent, cands: list[tuple[int, str]], total: int) -> None:
    print(
        f"\nLLM allocating {total} tokens "
        f"(model={agent.model}, temp={agent.llm_temperature})..."
    )
    alloc = agent.allocate_tokens(cands, total_tokens=total)
    summed = sum(p["tokens"] for p in alloc)
    print(f"\nAllocation ({len(alloc)} items, total={summed}):")
    for rank, p in enumerate(alloc, 1):
        iid, sz, tok = p["item_id"], p["size"], p["tokens"]
        row = agent.stats.get_row(iid, sz)
        share = tok / total
        print(
            f"  {rank:>2}. ({iid:>5}, {sz:<3})  tokens={tok:>2}  "
            f"share={share:.0%}  push={row['push_score']:.3f}  "
            f"stock={int(row['stock_count']):>3}  age={row['age_days']:>6.1f}d  "
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
            elif cmd == "reload":
                agent.stats.reload()
                cands = []
                print("stats reloaded; cached candidates cleared")
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
            elif cmd == "allocate":
                if not cands:
                    print("no cached candidates; run `candidates` first")
                    continue
                total = int(args[0]) if args else 10
                _print_allocation(agent, cands, total)
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
