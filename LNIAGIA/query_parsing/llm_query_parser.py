# ════════════════════════════════════════════════════════════
# llm_query_parser.py
# Uses a local Ollama LLM to parse a natural language clothing
# query into structured include/exclude filters + price range.
# ════════════════════════════════════════════════════════════

import json
import os
import re
import sys
from pathlib import Path

import ollama

# Ensure the LNIAGIA package root is on sys.path so `DB` imports work
# (this file lives in LNIAGIA/query_parsing, and DB/ is a sibling folder).
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(_PROJECT_ROOT))

from DB.vector.nl_mappings import ALL_MAPPINGS
from DB.models import FILTERABLE_FIELDS, FREE_TEXT_FILTER_FIELDS

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

DEFAULT_OLLAMA_REFINER_MODEL = "qwen2.5:7b-instruct-q3_K_M"

_LEGACY_OLLAMA_MODEL = (os.getenv("OLLAMA_MODEL") or "").strip()

OLLAMA_REFINER_MODEL = (
  (os.getenv("OLLAMA_REFINER_MODEL") or "").strip()
  or _LEGACY_OLLAMA_MODEL
  or DEFAULT_OLLAMA_REFINER_MODEL
)
OLLAMA_ROUTER_MODEL = (
  (os.getenv("OLLAMA_ROUTER_MODEL") or "").strip()
  or OLLAMA_REFINER_MODEL
)

# Backward-compatible alias used by older imports and status endpoints.
OLLAMA_MODEL = OLLAMA_REFINER_MODEL

_MAX_REFINEMENT_CONTEXT_MESSAGES = 6

_REPLACE_MARKERS = (
  " instead ",
  "instead of",
  "replace",
  "swap",
  "change to",
  "rather than",
)

_SET_ONLY_MARKERS = (
  " only ",
  "only want",
  "only need",
  "just want",
  "nothing but",
  "except ",
)

_TYPE_KEYWORD_MAP = {
  "t-shirt": "short_sleeve_top",
  "tshirt": "short_sleeve_top",
  "t shirt": "short_sleeve_top",
  "tee": "short_sleeve_top",
  "shirt": "long_sleeve_top",
  "blouse": "long_sleeve_top",
  "sweater": "long_sleeve_top",
  "jumper": "long_sleeve_top",
  "pullover": "long_sleeve_top",
  "sweatshirt": "long_sleeve_top",
  "hoodie": "long_sleeve_outwear",
  "jacket": "long_sleeve_outwear",
  "coat": "long_sleeve_outwear",
  "blazer": "long_sleeve_outwear",
  "cardigan": "long_sleeve_outwear",
  "parka": "long_sleeve_outwear",
  "windbreaker": "long_sleeve_outwear",
  "tank top": "vest",
  "camisole": "vest",
  "sleeveless top": "vest",
  "pants": "trousers",
  "trouser": "trousers",
  "trousers": "trousers",
  "jeans": "trousers",
  "slacks": "trousers",
  "chinos": "trousers",
  "leggings": "trousers",
  "shorts": "shorts",
  "skirt": "skirt",
  "dress": "short_sleeve_dress",
}

# ══════════════════════════════════════════════════════════════
# PROMPT
# ══════════════════════════════════════════════════════════════

def _build_system_prompt() -> str:
    """Build the system prompt with all valid field values."""

    field_lines = []
    for field in FILTERABLE_FIELDS:
        keys = sorted(ALL_MAPPINGS[field].keys())
        field_lines.append(f'  "{field}": {json.dumps(keys)}')

    fields_block = ",\n".join(field_lines)

    free_text_block = "\n".join(
        f'  "{f}": <copy the exact name from the query, e.g. "Nike", "Gucci">'
        for f in FREE_TEXT_FILTER_FIELDS
    )

    return f"""\
You are a structured-data extractor for a clothing search engine.

Given a user's natural-language query, extract structured filters
that will be used to search a clothing database.

VALID FIELD VALUES (use ONLY these exact strings):
{{
{fields_block}
}}

FREE-TEXT FIELDS (copy the exact value the user mentions, any capitalisation):
{{
{free_text_block}
}}

Return ONLY a JSON object with this schema — no explanation, no markdown:
{{
  "include": {{
    "<field>": ["<value>", ...]
  }},
  "exclude": {{
    "<field>": ["<value>", ...]
  }}
}}

RULES:

1. INCLUDE vs EXCLUDE:
  - "include" lists the values the user WANTS (explicitly stated).
   - "exclude" lists the values the user does NOT want (negations).
  - Only infer include values when Rule 3 explicitly allows it.

2. NEGATION HANDLING:
   When the user expresses negation using words like "not", "no", "don't", 
   "avoid", "without", "hate", or "don't want":
   
   - Put the negated value in "exclude" for that field.
   - DO NOT add alternative or opposite values to "include" for that same field.
   - Leave the field out of "include" entirely unless the user also mentions 
     a positive preference for that field.
   
   Examples:
   - "not fitted" -> exclude: {{"fit": ["fitted"]}}
     Do NOT add "relaxed", "loose", or other fits to include.
   
   - "no floral pattern" -> exclude: {{"pattern": ["floral"]}}
     Do NOT add "plain" or other patterns to include.
   
   - "not too short" -> exclude: {{"length": ["short", "mini"]}}
     Do NOT add "long", "midi", or other lengths to include.
   
   - "I want a casual black t-shirt, not fitted" ->
     include: {{"style": ["casual"], "color": ["black"], "type": ["short_sleeve_top"]}}
     exclude: {{"fit": ["fitted"]}}

3. CONTEXTUAL INFERENCE (allowed for "include" only):
  You may infer reasonable values for "include" ONLY when the request is
  underspecified and high-level (occasion/context only), for example:
   
   - "meeting with a CEO" -> infer formal style, professional occasion
   - "beach vacation" -> infer summer season, casual or sporty style
   - "wedding guest" -> infer elegant or formal style, wedding occasion

  IMPORTANT inference gate:
  - If the user provides any concrete attribute constraints (for example specific
    color, type, fit, pattern, material, brand, season, or explicit style),
    DO NOT infer additional fields they did not mention.
  - In explicit requests, extract only what the user said plus valid mappings.

  Example (explicit, no inference):
  - "I want a black t-shirt that is not too fitted" ->
    include: {{"color": ["black"], "type": ["short_sleeve_top"]}}
    exclude: {{"fit": ["fitted"]}}
    Do NOT add style/occasion/season unless explicitly requested.
   
   However, do NOT infer values for "exclude" unless the user explicitly
   expresses negation.

4. TYPE MAPPINGS:
   When the user mentions a generic clothing term, map it to the appropriate 
   specific type(s) from the valid values. If the term is ambiguous or generic,
   include all relevant subtypes.
   
   Common mappings:
   - "t-shirt", "tee" -> "short_sleeve_top"
   - "shirt", "blouse" -> "long_sleeve_top" (or "short_sleeve_top" if context suggests short sleeves)
   - "sweater", "jumper", "pullover", "sweatshirt" -> "long_sleeve_top"
   - "hoodie" -> "long_sleeve_outwear"
   - "jacket", "coat", "blazer", "cardigan", "parka", "windbreaker" -> "long_sleeve_outwear"
   - "tank top", "camisole", "sleeveless top" -> "vest"
   - "pants", "jeans", "slacks", "chinos", "leggings" -> "trousers"
   - "dress" (generic) -> include ALL dress types: ["short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"]
   - "top" (generic) -> include ALL top types: ["short_sleeve_top", "long_sleeve_top", "vest"]
   - "slip dress", "strappy dress", "spaghetti strap dress" -> "sling_dress"
   - "sleeveless dress" -> "vest_dress"
   - "gown", "maxi dress", "evening dress" -> include ALL dress types
   - "outerwear", "outer layer" (generic) -> "long_sleeve_outwear"
   - "bottoms" (generic) -> include ALL bottom types: ["shorts", "trousers", "skirt"]

5. COLOR EXPANSIONS:
   - "dark colors" -> ["black", "navy", "burgundy", "olive", "brown"]
   - "vivid colors", "bold colors" -> ["white", "yellow", "orange", "pink", "red", "purple", "coral", "teal"]
   - "neutral colors" -> ["white", "gray", "beige", "cream", "brown"]
   - "light colors" -> ["white", "beige", "cream", "pink", "yellow"]
   
   If the user says "not dark colors" or "avoid dark colors", put the dark colors 
   in "exclude" and do NOT add light colors to "include".

6. PATTERN NEGATION:
   - If the user says they do not want patterns (e.g., "no patterns", "without patterns"),
     set "pattern" to ["plain"] in "include".
   - If the user says they do not want a specific pattern (e.g., "no floral"),
     put only that pattern in "exclude" and do NOT add anything to "include" for pattern.

7. AGE GROUP RANGES:
   - "baby": (0, 2)
   - "child": (3, 12)
   - "teenager": (13, 17)
   - "young adult": (18, 29)
   - "adult": (30, 59)
   - "senior": (60, 120)

8. EMPTY SECTIONS:
   - If there are no inclusions, set "include" to {{}}.
   - If there are no exclusions, set "exclude" to {{}}.
   - Omit any section that is empty.

EXAMPLES:

Example 1:
  User: "I want a black T-shirt that is not too fitted"
  Output:
  {{
    "include": {{
      "color": ["black"],
      "type": ["short_sleeve_top"],
    }},
    "exclude": {{
      "fit": ["fitted"]
    }}
  }}

Example 2:
  User: "I am going to have a very important meeting with a CEO of a company, what do you recommend"
  Output:
  {{
    "include": {{
      "style": ["formal", "smart casual"],
      "occasion": ["work"],
      "color": ["black", "navy", "gray"]
    }},
    "exclude": {{}}
  }}

Example 3:
  User: "Show me summer dresses, but no floral"
  Output:
  {{
    "include": {{
      "type": ["short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"],
      "season": ["summer"]
    }},
    "exclude": {{
      "pattern": ["floral"]
    }}
  }}

Example 4:
  User: "casual pants that are not skinny and not too long"
  Output:
  {{
    "include": {{
      "type": ["trousers"],
      "style": ["casual"]
    }},
    "exclude": {{
      "fit": ["slim fit"],
      "length": ["full length", "ankle"]
    }}
  }}

Example 5:
  User: "I need something for a beach vacation, avoid dark colors"
  Output:
  {{
    "include": {{
      "season": ["summer"],
      "occasion": ["beach"],
      "style": ["casual"]
    }},
    "exclude": {{
      "color": ["black", "navy", "burgundy", "olive", "brown"]
    }}
  }}

After creating the JSON, verify:
- Are negated values placed in "exclude" and not "include"?
- Did you avoid adding opposite values to "include" when the user only expressed a negation?
- Is the JSON valid and properly formatted?

If any check fails, fix the JSON before outputting.
"""

def _build_user_prompt(query: str) -> str:
    return f'User query: "{query}"'


def _build_refinement_system_prompt() -> str:
    """Build the system prompt for iterative query refinement."""

    field_lines = []
    for field in FILTERABLE_FIELDS:
        keys = sorted(ALL_MAPPINGS[field].keys())
        field_lines.append(f'  "{field}": {json.dumps(keys)}')

    fields_block = ",\n".join(field_lines)

    free_text_block = "\n".join(
        f'  "{f}": <free text, copy exact value>'
        for f in FREE_TEXT_FILTER_FIELDS
    )

    return f"""\
You update a clothing-search state from a follow-up message.

You receive:
1) previous semantic query text
2) previous filters JSON
3) user follow-up refinement

VALID FIELD VALUES (use ONLY these exact strings for closed-set fields):
{{
{fields_block}
}}

FREE-TEXT FIELDS:
{{
{free_text_block}
}}

Return ONLY JSON with this exact schema:
{{
  "query": "<updated semantic query text>",
  "filters": {{
    "include": {{"<field>": ["<value>"]}},
    "exclude": {{"<field>": ["<value>"]}}
  }}
}}

Rules:
- Keep previous intent unless the refinement changes it.
- If user says "also/add/include", add constraints.
- If user says "instead/change/swap/replace", replace old value with new value in the same field.
- If user says "only" for a field, treat it as set-only for that field (clear older values in that field).
- If user negates something ("not", "don't", "without"), place that value in "exclude".
- If user both negates and replaces (e.g., "not t-shirt, trousers instead"), keep the new value in include and the negated one in exclude.
- Ensure include/exclude never contain the same value for the same field.
- Keep query text consistent with the final filters and concise.
- If a section is empty, you may omit it.
- If follow-up is ambiguous, prefer minimal safe changes.

Type replacement examples:
- Previous include.type: ["short_sleeve_top"], user: "Now I want trousers instead of a t-shirt"
  -> include.type: ["trousers"]
- Previous include.type: ["short_sleeve_top", "trousers"], user: "No, I only want trousers"
  -> include.type: ["trousers"]
- Previous include.type: ["short_sleeve_top"], user: "Also include trousers"
  -> include.type: ["short_sleeve_top", "trousers"]

Always prefer explicit replacement intent over additive behavior.
"""


def _build_refinement_user_prompt(
  previous_query: str,
  previous_filters: dict,
  refinement: str,
  recent_messages: list[str] | None = None,
) -> str:
  lines = [
    f"Previous semantic query: {json.dumps(previous_query)}",
    f"Previous filters: {json.dumps(previous_filters, ensure_ascii=False)}",
    f"User refinement: {json.dumps(refinement)}",
  ]

  if isinstance(recent_messages, list) and recent_messages:
    cleaned_messages = [
      str(message).strip()
      for message in recent_messages
      if isinstance(message, str) and message.strip()
    ][-_MAX_REFINEMENT_CONTEXT_MESSAGES:]
    if cleaned_messages:
      lines.append(
        "Recent user messages (oldest to newest): "
        f"{json.dumps(cleaned_messages, ensure_ascii=False)}"
      )

  return "\n".join(lines)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
  seen = set()
  output = []
  for value in values:
    if value in seen:
      continue
    seen.add(value)
    output.append(value)
  return output


def _normalize_value_token(value: str) -> str:
  return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_refinement_text(refinement: str) -> str:
  cleaned = re.sub(r"\s+", " ", str(refinement or "").strip().lower())
  return f" {cleaned} "


def _normalize_filter_dict(filters: dict) -> dict:
  normalized = {"include": {}, "exclude": {}}
  if not isinstance(filters, dict):
    return normalized

  for section in ("include", "exclude"):
    block = filters.get(section, {})
    if not isinstance(block, dict):
      continue

    cleaned_block = {}
    for field, values in block.items():
      if not isinstance(field, str) or not isinstance(values, list):
        continue
      cleaned_values = [str(value).strip() for value in values if isinstance(value, str) and value.strip()]
      if cleaned_values:
        cleaned_block[field] = _dedupe_preserve_order(cleaned_values)

    if cleaned_block:
      normalized[section] = cleaned_block

  return normalized


def _find_type_mentions(refinement: str) -> list[str]:
  lowered = _normalize_refinement_text(refinement).strip()
  mentions = []
  occupied_spans: list[tuple[int, int]] = []

  def _overlaps_existing(start: int, end: int) -> bool:
    return any(start < existing_end and end > existing_start for existing_start, existing_end in occupied_spans)

  for keyword, mapped in sorted(_TYPE_KEYWORD_MAP.items(), key=lambda pair: len(pair[0]), reverse=True):
    pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
    for match in re.finditer(pattern, lowered):
      start, end = match.span()
      if _overlaps_existing(start, end):
        continue
      occupied_spans.append((start, end))
      mentions.append(mapped)
      break

  for canonical in sorted(ALL_MAPPINGS.get("type", {}).keys(), key=len, reverse=True):
    phrase = canonical.replace("_", " ")
    pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
    for match in re.finditer(pattern, lowered):
      start, end = match.span()
      if _overlaps_existing(start, end):
        continue
      occupied_spans.append((start, end))
      mentions.append(canonical)
      break

  return _dedupe_preserve_order(mentions)


def _choose_replacement_values(
  previous_values: list[str],
  candidate_values: list[str],
  mentioned_values: list[str],
  set_only_intent: bool,
) -> list[str]:
  previous_tokens = {_normalize_value_token(value) for value in previous_values}

  if mentioned_values:
    introduced_mentions = [
      value
      for value in mentioned_values
      if _normalize_value_token(value) not in previous_tokens
    ]
    if introduced_mentions:
      return _dedupe_preserve_order(introduced_mentions)
    return _dedupe_preserve_order(mentioned_values)

  introduced_candidates = [
    value
    for value in candidate_values
    if _normalize_value_token(value) not in previous_tokens
  ]
  if introduced_candidates:
    return _dedupe_preserve_order(introduced_candidates)

  if set_only_intent and candidate_values:
    return [candidate_values[0]]

  return _dedupe_preserve_order(candidate_values)


def _enforce_include_exclude_disjoint(filters: dict) -> dict:
  normalized = _normalize_filter_dict(filters)
  include = dict(normalized.get("include", {}))
  exclude = dict(normalized.get("exclude", {}))

  for field, ex_values in exclude.items():
    if field not in include:
      continue
    ex_tokens = {_normalize_value_token(value) for value in ex_values}
    kept = [value for value in include[field] if _normalize_value_token(value) not in ex_tokens]
    if kept:
      include[field] = kept
    else:
      del include[field]

  output = {}
  if include:
    output["include"] = include
  if exclude:
    output["exclude"] = exclude
  return output


def _apply_refinement_safety(previous_filters: dict, refinement: str, candidate_filters: dict) -> dict:
  previous = _normalize_filter_dict(previous_filters)
  candidate = _normalize_filter_dict(candidate_filters)

  include = dict(candidate.get("include", {}))
  exclude = dict(candidate.get("exclude", {}))
  previous_include = previous.get("include", {})

  lowered = _normalize_refinement_text(refinement)
  replace_intent = any(marker in lowered for marker in _REPLACE_MARKERS)
  set_only_intent = any(marker in lowered for marker in _SET_ONLY_MARKERS)

  if replace_intent or set_only_intent:
    for field, candidate_values in list(include.items()):
      if not isinstance(candidate_values, list) or not candidate_values:
        continue

      previous_values = previous_include.get(field, [])
      if field == "type":
        mentioned_types = _find_type_mentions(refinement)
        include[field] = _choose_replacement_values(
          previous_values=previous_values,
          candidate_values=candidate_values,
          mentioned_values=mentioned_types,
          set_only_intent=set_only_intent,
        )
        continue

      if not previous_values:
        continue

      include[field] = _choose_replacement_values(
        previous_values=previous_values,
        candidate_values=candidate_values,
        mentioned_values=[],
        set_only_intent=set_only_intent,
      )

  safe_filters = {"include": include, "exclude": exclude}
  return _enforce_include_exclude_disjoint(safe_filters)


# ══════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════

def parse_query(query: str, model: str | None = None, verbose: bool = False) -> dict:
    """
    Send a natural-language query to Ollama and return structured filters.

    Args:
        query:   The user's free-text search query.
        model:   Ollama model name (defaults to OLLAMA_MODEL).
        verbose: Print the raw LLM response for debugging.

    Returns:
        Dict with keys "include", "exclude" (either may be absent).
    """
    model = model or OLLAMA_REFINER_MODEL

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": _build_user_prompt(query)},
        ],
        # Latency knobs (safe, output unchanged):
        # - format="json" constrains decoding to a JSON object (no fences).
        # - keep_alive keeps the model resident so repeated calls skip reload.
        # - num_ctx=4096 ensures the large system prompt isn't silently
        #   truncated by the ~2048 default; num_predict caps output length.
        format="json",
        keep_alive="30m",
        options={"temperature": 0, "num_ctx": 4096, "num_predict": 512},
    )

    raw = response.message.content.strip()

    if verbose:
        print(f"\n  [LLM raw response]\n  {raw}\n")

    # Strip markdown fences if the model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]  # drop first ```json line
        raw = raw.rsplit("```", 1)[0]  # drop trailing ```
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  WARNING: LLM returned invalid JSON. Using empty filters.")
        parsed = {}

    # Validate values against known keys
    parsed = _validate(parsed)
    return parsed


def refine_query(
    previous_query: str,
    previous_filters: dict,
    refinement: str,
    model: str | None = None,
  recent_messages: list[str] | None = None,
    verbose: bool = False,
) -> dict:
    """
    Update semantic query text + filters using a follow-up refinement.

    Args:
        previous_query: Existing semantic query string.
        previous_filters: Existing parsed filters.
        refinement: User follow-up text (e.g., "also plain", "white instead").
        model: Ollama model name (defaults to OLLAMA_MODEL).
        recent_messages: Optional bounded user-message context for better refinement.
        verbose: Print raw LLM response.

    Returns:
        Dict with keys:
            "query": updated semantic query string
            "filters": updated validated filter dict
    """
    model = model or OLLAMA_REFINER_MODEL

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _build_refinement_system_prompt()},
            {
                "role": "user",
                "content": _build_refinement_user_prompt(
                    previous_query=previous_query,
                    previous_filters=previous_filters,
                    refinement=refinement,
                  recent_messages=recent_messages,
                ),
            },
        ],
        # Same latency knobs as parse_query (see note there).
        format="json",
        keep_alive="30m",
        options={"temperature": 0, "num_ctx": 4096, "num_predict": 512},
    )

    raw = response.message.content.strip()

    if verbose:
        print(f"\n  [LLM raw refinement response]\n  {raw}\n")

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print("  WARNING: LLM returned invalid refinement JSON. Keeping previous state.")
        return {
            "query": f"{previous_query} {refinement}".strip(),
            "filters": _validate(dict(previous_filters)),
        }

    updated_query = parsed.get("query")
    if not isinstance(updated_query, str) or not updated_query.strip():
        updated_query = f"{previous_query} {refinement}".strip()

    updated_filters = parsed.get("filters", previous_filters)
    if not isinstance(updated_filters, dict):
        updated_filters = previous_filters

    safe_filters = _apply_refinement_safety(
        previous_filters=previous_filters,
        refinement=refinement,
        candidate_filters=updated_filters,
    )

    return {
        "query": updated_query.strip(),
        "filters": _validate(safe_filters),
    }


def _validate(parsed: dict) -> dict:
    """Remove any field values that are not in the valid set."""
    for section in ("include", "exclude"):
        block = parsed.get(section, {})
        if not isinstance(block, dict):
            if section in parsed:
                del parsed[section]
            continue

        cleaned = {}
        for field, values in block.items():
            if not isinstance(values, list):
                continue

            # Free-text fields (e.g. brand) — just ensure non-empty strings,
            # no closed-set validation needed.
            if field in FREE_TEXT_FILTER_FIELDS:
                filtered = [v for v in values if isinstance(v, str) and v.strip()]
                if filtered:
                    cleaned[field] = filtered
                continue
            if field not in ALL_MAPPINGS:
                continue
            valid = set(ALL_MAPPINGS[field].keys())
            filtered = [v for v in values if v in valid]
            if filtered:
                cleaned[field] = filtered
        if cleaned:
            parsed[section] = cleaned
        elif section in parsed:
            del parsed[section]

    return parsed


# ══════════════════════════════════════════════════════════════
# CLI — quick manual test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  LLM Query Parser  (Ollama)")
    print("=" * 60)
    print(f"  Refiner model: {OLLAMA_REFINER_MODEL}")
    print(f"  Interaction model: {OLLAMA_ROUTER_MODEL}")
    print("  Type a clothing query, or 'exit' to quit.\n")

    # Warm up the Ollama model once before prompting the user so the
    # first interactive query isn't delayed by model startup.
    try:
      print(" Loading LLM model (warming up)...")
      _ = ollama.chat(
        model=OLLAMA_REFINER_MODEL,
        messages=[{"role": "system", "content": "warmup"}, {"role": "user", "content": "ping"}],
        options={"temperature": 0},
      )
      print(" Model ready.\n")
    except Exception as e:
      print(f" Warning: could not warm up Ollama model: {e}\n")

    print(" System prompt:")
    print(_build_system_prompt())
    while True:
        q = input("  Query > ").strip()
        if not q or q.lower() == "exit":
            break

        result = parse_query(q, verbose=True)
        print("  Parsed filters:")
        print(f"  {json.dumps(result, indent=2)}\n")
