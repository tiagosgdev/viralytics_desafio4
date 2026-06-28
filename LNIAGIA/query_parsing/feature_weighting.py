# ════════════════════════════════════════════════════════════
# feature_weighting.py
# Asks the user what they are looking for (in a persona voice),
# then analyses the answer against the 3 detected features
# (color, type, body type) and returns filters plus importance
# weights that sum to 100.
# ════════════════════════════════════════════════════════════

import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import ollama

_SCRIPT_DIR = Path(__file__).resolve().parent           # LNIAGIA/query_parsing/
_LNIAGIA_DIR = _SCRIPT_DIR.parent                       # LNIAGIA/
for _path in (_LNIAGIA_DIR, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from DB.models import TYPE, COLOR, BODY_TYPE, STYLE, OCCASION
from query_parsing.llm_query_parser import (
    OLLAMA_REFINER_MODEL,
    _build_system_prompt,
    _validate,
)

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

DEFAULT_PERSONA = "cruella"
SUPPORTED_PERSONAS = {"cruella", "edna"}

_TYPE_DISPLAY_MAP = {
    "short_sleeve_top": "t-shirt",
    "long_sleeve_top": "long-sleeve top",
    "long_sleeve_outwear": "jacket",
    "vest": "vest",
    "shorts": "shorts",
    "trousers": "trousers",
    "skirt": "skirt",
    "short_sleeve_dress": "short-sleeve dress",
    "long_sleeve_dress": "long-sleeve dress",
    "vest_dress": "sleeveless dress",
    "sling_dress": "slip dress",
}

# Templates use {color}, {type} and {body_type} placeholders.
# generate_intent_question picks one at random per call so the user does
# not see the same opener on every scan.
_QUESTION_VARIANTS: dict[str, tuple[str, ...]] = {
    "cruella": (
        "Darling, I see a {color} {type} made for a {body_type} shape. Tell me what you are hunting for.",
        "Oh darling, a {color} {type} flattering a {body_type} figure — bold start. What direction are we taking this?",
        "Well well, a {color} {type} cut for a {body_type} shape. Spill it, darling — what look are you after?",
        "A {color} {type} for a {body_type} silhouette — promising, darling. What do you want next?",
        "Darling, that {color} {type} made for a {body_type} shape is only the opening act. What are you really craving?",
    ),
    "edna": (
        "I see a {color} {type} made for a {body_type} shape. Tell me exactly what you are looking for.",
        "Detected: {color} {type}, {body_type} shape. State what you want. Be specific.",
        "A {color} {type} for a {body_type} figure. Now — what are we changing, keeping, or replacing?",
        "{color} {type}, {body_type} shape. Tell me your requirements. No vagueness.",
        "I have a {color} {type} for a {body_type} shape on screen. Describe what you need.",
    ),
}

_DEFAULT_QUESTION_VARIANTS: tuple[str, ...] = (
    "I see a {color} {type} made for a {body_type} shape. What are you looking for?",
    "Looks like a {color} {type} for a {body_type} figure. What would you like next?",
    "I detected a {color} {type} made for a {body_type} shape. Tell me what you want.",
)


# ══════════════════════════════════════════════════════════════
# PERSONA
# ══════════════════════════════════════════════════════════════

def _normalize_persona(value: Any) -> str:
    key = str(value or DEFAULT_PERSONA).strip().lower()
    if key == "cruela":
        key = "cruella"
    if key in SUPPORTED_PERSONAS:
        return key
    return DEFAULT_PERSONA


def _humanize_type(value: str) -> str:
    return _TYPE_DISPLAY_MAP.get(value, str(value or "").replace("_", " "))


def _humanize_body_type(value: str) -> str:
    return str(value or "").replace("_", " ")


# ══════════════════════════════════════════════════════════════
# STEP 1 — ASK THE USER
# ══════════════════════════════════════════════════════════════

def generate_intent_question(
    detected_color: str,
    detected_type: str,
    detected_body_type: str,
    persona: str = DEFAULT_PERSONA,
) -> str:
    """
    Build a short, persona-styled question that asks the user what they
    are looking for, given the three detected features.

    Picks a random phrase from the persona's variant list so the question
    feels fresh across multiple scans. No LLM call — pure string formatting.
    """
    persona_key = _normalize_persona(persona)
    variants = _QUESTION_VARIANTS.get(persona_key, _DEFAULT_QUESTION_VARIANTS)
    template = random.choice(variants)
    return template.format(
        color=detected_color,
        type=_humanize_type(detected_type),
        body_type=_humanize_body_type(detected_body_type),
    )

# ══════════════════════════════════════════════════════════════
# STEP 2 — ANALYSE THE ANSWER
# ══════════════════════════════════════════════════════════════

def _build_intent_system_prompt() -> str:
    color_values = json.dumps(sorted(COLOR))
    type_values = json.dumps(sorted(TYPE))
    body_type_values = json.dumps(sorted(BODY_TYPE))
    style_values = json.dumps(sorted(STYLE))
    occasion_values = json.dumps(sorted(OCCASION))
    base_prompt = _build_system_prompt()

    return f"""\
{base_prompt}

═══════════════════════════════════════════════════════════════
IN-STORE SCAN MODE — THIS SUPERSEDES THE OUTPUT SCHEMA ABOVE
═══════════════════════════════════════════════════════════════

You weigh clothing features by user intent and produce scan filters. The
filters drive what the store retrieves and how the recommender scores items,
so capture EVERY feature the shopper signals — including style and occasion.

Inputs:
- detected_color, detected_type, detected_body_type (from the vision scan)
- user_answer: free text replying to "what are you looking for?"

SCAN FILTER RULES (in addition to ALL rules above):
- detected_type is the shopper's starting garment. Put it in
    include.type UNLESS the answer names a different type — then use the
    new type only (do NOT exclude the old one).
- detected_color: keep it in include.color UNLESS the answer names a
    different color or negates it (then follow the normal negation rules).
- detected_body_type is a HARD constraint from the vision system. ALWAYS
    put detected_body_type in include.body_type, UNLESS the shopper
    explicitly names a different body shape — then use the shopper's shape.
    Use ONLY the valid values listed below.
- STYLE: when the answer signals a look or aesthetic (e.g. "smart",
    "minimalist", "elegant", "sporty", "streetwear", "casual"), put the
    matching value(s) in include.style. Map the shopper's words to the
    CLOSEST valid style value(s) below. Omit style only when the answer
    gives no style signal at all. Do NOT invent a style the shopper didn't
    imply.
- OCCASION: when the answer signals a context or event (e.g. "for work",
    "for a party", "date night", "everyday", "the beach", "a wedding"),
    put the matching value(s) in include.occasion. Map to the CLOSEST valid
    occasion value(s) below. Omit occasion only when no occasion is implied.
- Only add something to exclude when the shopper explicitly says they do
    NOT want it (direct negation like "not", "don't want", "avoid"). Do not
    infer excludes from preferences alone.
- All other fields follow the normal include/exclude rules above.

WEIGHTS — also score the importance of FOUR scan features:
- color, type, bodyType (body shape), and stock (inventory / popularity intent).
- Each importance is a number greater than 0; the FOUR MUST sum to 100.
- Higher importance = the user emphasised that feature more.
- Explicit mentions outweigh implicit / contextual hints.
- A contradiction (user wants a different value) scores HIGH because the
    user clearly cares about that feature.
- A feature the user never mentions still gets some importance — never 0.
- If the user talks about fit ("fit", "fits", "fitting", "fits me well"),
    give EXTRA importance to bodyType.
- stock: how much the shopper cares about inventory / popularity rather than a
    specific exact item. RAISE it when they say things like "popular", "what's
    trending", "best sellers", "on sale", "clearance", "whatever's in stock".
    LOWER it when the shopper is very specific about an exact item (precise
    color + type + shape) — they want THAT piece, not whatever is trending.
    When the answer gives no signal either way, keep stock small but > 0.

VALID VALUES (use ONLY these exact strings):
- color:    {color_values}
- type:     {type_values}
- bodyType: {body_type_values}
- style:    {style_values}
- occasion: {occasion_values}

EXAMPLES (detected_color=red, detected_type=short_sleeve_top, detected_body_type=hourglass).
These examples show ONLY the weights block for clarity; still return the
full scan output schema below.

Example A — user_answer: "I am looking for a t-shirt"
    Mentions type only. Color and body type are not mentioned. No stock cue.
    {{
        "color":    {{"value": "red", "importance": 10}},
        "type":     {{"value": "short_sleeve_top", "importance": 70}},
        "bodyType": {{"value": "hourglass", "importance": 10}},
        "stock":    {{"importance": 10}}
    }}

Example B — user_answer: "I want a red casual t-shirt"
    Mentions color + type + style ("casual" → include.style=["casual"]).
    Body type not mentioned. (Style/occasion go in include only — the FOUR
    weighted features below are still color/type/bodyType/stock.)
    {{
        "color":    {{"value": "red", "importance": 43}},
        "type":     {{"value": "short_sleeve_top", "importance": 43}},
        "bodyType": {{"value": "hourglass", "importance": 6}},
        "stock":    {{"importance": 8}}
    }}

Example C — user_answer: "I want something for the summer"
    Summer softly hints at short-sleeve tops. No explicit feature mention.
    {{
        "color":    {{"value": "red", "importance": 25}},
        "type":     {{"value": "short_sleeve_top", "importance": 40}},
        "bodyType": {{"value": "hourglass", "importance": 25}},
        "stock":    {{"importance": 10}}
    }}

Example D — user_answer: "I am looking for a green t-shirt"
    Contradicts color (red -> green) AND confirms type. Body type not mentioned.
    {{
        "color":    {{"value": "green", "importance": 43}},
        "type":     {{"value": "short_sleeve_top", "importance": 43}},
        "bodyType": {{"value": "hourglass", "importance": 6}},
        "stock":    {{"importance": 8}}
    }}

Example E — user_answer: "Something for a pear shape in blue"
    Contradicts color (red -> blue) AND body type (hourglass -> pear). Type not mentioned.
    {{
        "color":    {{"value": "blue", "importance": 42}},
        "type":     {{"value": "short_sleeve_top", "importance": 10}},
        "bodyType": {{"value": "pear", "importance": 42}},
        "stock":    {{"importance": 6}}
    }}

Example F — user_answer: "Something that fits me well"
    Mentions fit, which hints at body type being more important. No explicit feature mention.
    {{
        "color":    {{"value": "blue", "importance": 18}},
        "type":     {{"value": "short_sleeve_top", "importance": 18}},
        "bodyType": {{"value": "pear", "importance": 56}},
        "stock":    {{"importance": 8}}
    }}

Example G — user_answer: "Just show me what's popular and on sale"
    Strong stock / popularity intent; no specific feature emphasised.
    {{
        "color":    {{"value": "blue", "importance": 12}},
        "type":     {{"value": "short_sleeve_top", "importance": 12}},
        "bodyType": {{"value": "pear", "importance": 12}},
        "stock":    {{"importance": 64}}
    }}

Example H — user_answer: "a smart minimalist top for work"
    Signals style ("smart"→smart casual, "minimalist") AND occasion
    ("for work"→work) in addition to type. This example shows the FILTERS
    block (not just weights) to make the include capture explicit:
    "filters": {{
        "include": {{
            "color":    ["red"],
            "type":     ["long_sleeve_top"],
            "body_type":["hourglass"],
            "style":    ["smart casual", "minimalist"],
            "occasion": ["work"]
        }},
        "exclude": {{}}
    }}
    (Note "smart top" → a top garment, e.g. long_sleeve_top; do NOT collapse
    it to short_sleeve_top just because the detected garment was one.)

FINAL SCAN OUTPUT — return ONLY this JSON object (no markdown, no prose):
{{
    "query": "<short semantic search text consistent with the filters>",
    "filters": {{
        "include": {{ "<field>": ["<value>", ...] }},
        "exclude": {{ "<field>": ["<value>", ...] }}
    }},
    "weights": {{
        "color":    {{"importance": <number>}},
        "type":     {{"importance": <number>}},
        "bodyType": {{"importance": <number>}},
        "stock":    {{"importance": <number>}}
    }}
}}
"""


def _build_intent_user_prompt(
    detected_color: str,
    detected_type: str,
    detected_body_type: str,
    user_answer: str,
) -> str:
    return (
        f"detected_color: {detected_color}\n"
        f"detected_type: {detected_type}\n"
        f"detected_body_type: {detected_body_type}\n"
        f"user_answer: {json.dumps(user_answer)}"
    )


# ══════════════════════════════════════════════════════════════
# STEP 2 — ONE CALL: FILTERS + WEIGHTS FOR THE SCAN FLOW
# ══════════════════════════════════════════════════════════════
#
# The in-store flow is: vision detects (color, type, body_type) ->
# we ask ONE question -> the shopper answers -> from that single answer
# we need BOTH:
#   (a) the full include/exclude filter schema to query the DB, and
#   (b) the importance weights for the 3 features used by the
#       bidding/voting multi-agent system.
#
# These are produced in ONE LLM call so the filter values and the
# weight values cannot diverge (consistency by construction):
# the LLM returns the filters + 3 importance numbers, and the code
# reads each weight's VALUE back FROM the final filters (color/type)
# and from vision (body_type). This also keeps us within the latency
# budget (one round-trip instead of two).

# ── Deterministic color-negation handling ──────────────────────
# The small quantized LLM is unreliable at (a) moving a negated color
# category to "exclude" and (b) scoring how confident we are in a color
# the shopper just rejected. We therefore reconcile color in code, from
# the raw answer, so filters and the confidence number always agree.

# Mirrors the COLOR EXPANSIONS in llm_query_parser's system prompt.
_COLOR_CATEGORY_MAP: dict[str, tuple[str, ...]] = {
    "dark": ("black", "navy", "burgundy", "olive", "brown"),
    "light": ("white", "beige", "cream", "pink", "yellow"),
    "neutral": ("white", "gray", "beige", "cream", "brown"),
    "vivid": ("white", "yellow", "orange", "pink", "red", "purple", "coral", "teal"),
    "bold": ("white", "yellow", "orange", "pink", "red", "purple", "coral", "teal"),
    "bright": ("white", "yellow", "orange", "pink", "red", "purple", "coral", "teal"),
}

_NEGATION_WORDS = {
    "not", "no", "dont", "never", "avoid", "without", "hate", "cannot",
    "cant", "nope", "neither", "nor", "exclude", "skip",
}

_FIT_TERMS_RE = re.compile(r"\bfit(s|ting)?\b")
_BODYTYPE_FIT_BOOST = 1.6

# Confidence (=importance %) the color feature is pinned to when the
# shopper rejects the detected color. Both are very low, by request:
#   - direct  ("I don't want red", detected red) -> between 0 and 1
#   - category ("not too dark", detected black)  -> between 1 and 2
_COLOR_DIRECT_NEGATION_PCT = 0.5
_COLOR_CATEGORY_NEGATION_PCT = 1.5
# When the shopper negates a *different* color, the detected one survived
# and gains a little confidence — applied as a multiplier pre-normalization.
_COLOR_SURVIVOR_BOOST = 1.3


def _is_negation_token(token: str) -> bool:
    return token in _NEGATION_WORDS or token.endswith("n't")


def _mentions_fit(answer: str) -> bool:
    return bool(_FIT_TERMS_RE.search(str(answer or "").lower()))


def _split_terms_by_negation(
    answer: str,
    candidate_terms: set[str],
) -> tuple[set[str], set[str]]:
    """
    Return (negated_terms, positive_terms) found in `answer`.

    A candidate term is "negated" when a negation cue appears before it in
    the SAME clause (covers "not red", "I don't want red", "not too dark").
    Punctuation (",", ".", ";") ends a clause, so a negation does not leak
    into the next one — e.g. "not red, I want blue" negates only red.
    Otherwise the term is "positive".
    """
    tokens = re.findall(r"[a-z']+|[,.;]", str(answer or "").lower())
    boundaries = {",", ".", ";"}
    neg_positions = [i for i, tok in enumerate(tokens) if _is_negation_token(tok)]

    negated: set[str] = set()
    positive: set[str] = set()
    for j, tok in enumerate(tokens):
        if tok not in candidate_terms:
            continue
        in_negation = any(
            i < j
            and (j - i) <= 6
            and not any(tokens[k] in boundaries for k in range(i + 1, j))
            for i in neg_positions
        )
        (negated if in_negation else positive).add(tok)
    return negated, positive


# ── Deterministic price / budget extraction ────────────────────────
# The small quantized LLM is unreliable at pulling a numeric budget out of
# chat, so we resolve it in code from the raw answer (same philosophy as the
# color-negation reconciliation above). Two hard thresholds partition the
# catalogue into cheap / medium / expensive (catalogue spread: p25≈34,
# median≈68, p75≈153), used for vague terms like "cheap" or "premium".
PRICE_CHEAP_MAX = 50.0
PRICE_EXPENSIVE_MIN = 150.0

_PRICE_AMOUNT = r"\$?\s*(\d+(?:\.\d+)?)"
_PRICE_BETWEEN_RE = re.compile(
    r"\b(?:between|from)\b\s*" + _PRICE_AMOUNT + r"\s*(?:and|to|-|–|—)\s*" + _PRICE_AMOUNT,
    re.I,
)
_PRICE_RANGE_RE = re.compile(_PRICE_AMOUNT + r"\s*(?:to|-|–|—)\s*" + _PRICE_AMOUNT, re.I)
_PRICE_UNDER_RE = re.compile(
    r"\b(?:under|below|less than|cheaper than|up to|no more than|at most|within|max(?:imum)?)\b\s*"
    + _PRICE_AMOUNT,
    re.I,
)
# A negated floor is really a ceiling: "nothing over 40", "not above 75",
# "no more than 60". Checked BEFORE the floor pattern so it wins.
_PRICE_NEG_OVER_RE = re.compile(
    r"\b(?:no|not|nothing|none)\b[\w\s]{0,10}?\b(?:over|above|more than|more)\b\s*" + _PRICE_AMOUNT,
    re.I,
)
_PRICE_OVER_RE = re.compile(
    r"\b(?:over|above|more than|at least|min(?:imum)?|starting (?:from|at))\b\s*" + _PRICE_AMOUNT,
    re.I,
)
# "not/too/less expensive" means they want it CHEAPER, not pricier.
_PRICE_NEG_EXPENSIVE_RE = re.compile(
    r"\b(?:not|no|nothing|isn'?t|too|less|avoid)\b[\w\s]{0,12}\bexpensive\b", re.I
)
_PRICE_CHEAP_WORDS = (
    "cheap", "cheapest", "affordable", "inexpensive", "budget", "low price",
    "low-priced", "economical", "bargain",
)
_PRICE_EXPENSIVE_WORDS = (
    "expensive", "premium", "luxury", "luxurious", "high-end", "high end",
    "pricey", "designer", "splurge",
)
_PRICE_MEDIUM_WORDS = (
    "mid-range", "mid range", "midrange", "medium price", "moderate",
    "moderately priced", "reasonably priced", "reasonable price",
)


def _extract_price_range(user_answer: str) -> tuple[float | None, float | None]:
    """Return (price_min, price_max) inferred from the shopper's words.

    Numeric phrases win over vague terms; vague terms fall back to the two hard
    thresholds. Either bound may be None (open). Returns (None, None) when no
    budget signal is present.
    """
    text = str(user_answer or "").lower()
    if not text:
        return None, None

    # 1) Explicit numeric ranges / bounds take priority.
    m = _PRICE_BETWEEN_RE.search(text) or _PRICE_RANGE_RE.search(text)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (min(lo, hi), max(lo, hi))
    # A negated floor ("nothing over 40") is a ceiling — resolve it first so it
    # is not misread as a floor by _PRICE_OVER_RE.
    neg_over = _PRICE_NEG_OVER_RE.search(text)
    under = _PRICE_UNDER_RE.search(text)
    over = None if neg_over else _PRICE_OVER_RE.search(text)
    ceiling = float(neg_over.group(1)) if neg_over else (float(under.group(1)) if under else None)
    floor = float(over.group(1)) if over else None
    if ceiling is not None or floor is not None:
        return floor, ceiling

    # 2) Vague terms → the two hard thresholds. Check the "not expensive" =
    #    cheaper case before the plain "expensive" word so negation wins.
    if _PRICE_NEG_EXPENSIVE_RE.search(text):
        return None, PRICE_CHEAP_MAX
    if any(w in text for w in _PRICE_MEDIUM_WORDS):
        return PRICE_CHEAP_MAX, PRICE_EXPENSIVE_MIN
    if any(w in text for w in _PRICE_CHEAP_WORDS):
        return None, PRICE_CHEAP_MAX
    if any(w in text for w in _PRICE_EXPENSIVE_WORDS):
        return PRICE_EXPENSIVE_MIN, None
    return None, None


def _reconcile_color_negation(
    answer: str,
    detected_color: str,
    include: dict[str, list[str]],
    exclude: dict[str, list[str]],
) -> str:
    """
    Make the color filters agree with what the shopper actually said and
    return a state describing the detected color's standing:

        "confirmed_positive" | "replaced_positive" |
        "direct_negation" | "category_negation" |
        "other_negation" | "neutral"

    Mutates `include`/`exclude` so negated colors (named or via category)
    are removed from include and added to exclude.
    """
    color_set = set(COLOR)
    candidate_terms = color_set | set(_COLOR_CATEGORY_MAP)
    negated, positive = _split_terms_by_negation(answer, candidate_terms)

    negated_named = {t for t in negated if t in color_set}
    negated_category_colors: set[str] = set()
    for term in negated:
        if term in _COLOR_CATEGORY_MAP:
            negated_category_colors.update(_COLOR_CATEGORY_MAP[term])
    negated_colors = negated_named | negated_category_colors

    positive_named = {t for t in positive if t in color_set}
    positive_category_colors: set[str] = set()
    for term in positive:
        if term in _COLOR_CATEGORY_MAP:
            positive_category_colors.update(_COLOR_CATEGORY_MAP[term])

    # Apply negations to the filters (move out of include, into exclude).
    if negated_colors:
        kept_include = [c for c in include.get("color", []) if c not in negated_colors]
        if kept_include:
            include["color"] = kept_include
        else:
            include.pop("color", None)

        current_exclude = list(exclude.get("color", []))
        for color in sorted(negated_colors):
            if color not in current_exclude:
                current_exclude.append(color)
        exclude["color"] = current_exclude

    # Decide the detected color's standing (positive intent wins).
    if detected_color in positive_named or detected_color in positive_category_colors:
        return "confirmed_positive"
    if positive_named:
        return "replaced_positive"
    if detected_color in negated_named:
        return "direct_negation"
    if detected_color in negated_category_colors:
        return "category_negation"
    if negated_colors:
        return "other_negation"
    return "neutral"


def _first(values: Any) -> Any:
    if isinstance(values, list) and values:
        return values[0]
    return None


def _inject_detected_filters(
    include: dict[str, list[str]],
    detected_type: str,
    detected_body_type: str,
) -> None:
    """
    Guarantee the scan's hard features are present in include.

    - type: inject detected_type only if the LLM produced no type at all
      (i.e. the shopper did not name a different garment).
    - body_type: inject detected_body_type only if the LLM produced none
      (the override case keeps whatever valid body shape the LLM set).

    Only valid closed-set values are injected, so the result stays clean
    even if vision emits an unexpected token.
    """
    if not include.get("type") and detected_type in set(TYPE):
        include["type"] = [detected_type]
    if not include.get("body_type") and detected_body_type in set(BODY_TYPE):
        include["body_type"] = [detected_body_type]


def _normalize_scan_weights(
    raw_weights: Any,
    values: dict[str, str],
    user_answer: str,
    color_state: str = "neutral",
) -> dict[str, dict[str, Any]]:
    """
    Build the {color/type/bodyType/stock: {value, importance}} block.

    Importance numbers come from the LLM; the color/type/bodyType VALUES come
    from `values` (derived from the final filters + vision) so they cannot
    diverge. `stock` has no closed-set value, so its value is None.

    `color_state` (from _reconcile_color_negation) overrides the color
    importance deterministically:
      - direct_negation   -> pinned very low (~0.5)
      - category_negation -> pinned low      (~1.5)
      - other_negation    -> small boost (the detected color survived)
      - confirmed/replaced/neutral -> LLM number kept as-is

    Pinned color is held fixed; the remaining budget (100 - pinned) is
    split across the other features in proportion to their LLM numbers.
    Guarantees: every importance > 0 and the FOUR sum to 100.0.
    """
    if not isinstance(raw_weights, dict):
        raw_weights = {}

    features = ("color", "type", "bodyType", "stock")
    raw_importances: dict[str, float] = {}
    for feature in features:
        block = raw_weights.get(feature)
        candidate = block.get("importance") if isinstance(block, dict) else block
        try:
            importance = float(candidate)
        except (TypeError, ValueError):
            importance = 0.0
        if importance <= 0:
            importance = 1.0
        raw_importances[feature] = importance

    if _mentions_fit(user_answer):
        raw_importances["bodyType"] *= _BODYTYPE_FIT_BOOST

    # Apply the deterministic color override.
    pinned: dict[str, float] = {}
    if color_state == "direct_negation":
        pinned["color"] = _COLOR_DIRECT_NEGATION_PCT
    elif color_state == "category_negation":
        pinned["color"] = _COLOR_CATEGORY_NEGATION_PCT
    elif color_state == "other_negation":
        raw_importances["color"] *= _COLOR_SURVIVOR_BOOST

    unpinned = [f for f in features if f not in pinned]
    remaining = max(0.0, 100.0 - sum(pinned.values()))
    unpinned_total = sum(raw_importances[f] for f in unpinned)

    result: dict[str, dict[str, Any]] = {}
    for feature in features:
        if feature in pinned:
            pct = round(pinned[feature], 2)
        elif unpinned_total <= 0:
            pct = round(remaining / len(unpinned), 2)
        else:
            pct = round((raw_importances[feature] / unpinned_total) * remaining, 2)
        result[feature] = {"value": values.get(feature), "importance": pct}

    # Absorb rounding drift into the dominant UNPINNED feature (never the
    # pinned color, whose low value is intentional).
    drift = round(100.0 - sum(r["importance"] for r in result.values()), 2)
    if drift != 0 and unpinned:
        top = max(unpinned, key=lambda k: result[k]["importance"])
        result[top]["importance"] = round(result[top]["importance"] + drift, 2)

    return result


def analyze_intent(
    detected_color: str,
    detected_type: str,
    detected_body_type: str,
    user_answer: str,
    model: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    ONE LLM call that turns a single shopper answer into BOTH the DB
    filters and the multi-agent importance weights for the scan flow.

    Returns:
        {
          "query":   "<semantic search text>",
          "filters": {"include": {...}, "exclude": {...}},
          "weights": {
            "color":    {"value": "<valid color>",     "importance": <float>},
            "type":     {"value": "<valid type>",       "importance": <float>},
            "bodyType": {"value": "<valid body type>",  "importance": <float>},
            "stock":    {"value": None,                 "importance": <float>}
          }
        }

    Guarantees:
        - filters only contain valid closed-set values (via _validate)
        - detected type + body_type are always present in include
          (body_type is a HARD Qdrant filter) unless the shopper
          explicitly replaced them
        - every weight importance > 0 and the FOUR sum to 100.0
        - each color/type/bodyType VALUE is read back from the final filters /
          vision, so weights and filters are always consistent; stock has no
          closed-set value so its value is None
    """
    model = model or OLLAMA_REFINER_MODEL
    system_prompt = _build_intent_system_prompt()

    parsed: Any = {}
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": _build_intent_user_prompt(
                        detected_color, detected_type, detected_body_type, user_answer
                    ),
                },
            ],
            format="json",
            keep_alive="30m",
            options={"temperature": 0.5, "num_predict": 512, "num_ctx": 4096},
        )

        raw = ""
        message_obj = getattr(response, "message", None)
        if message_obj is not None:
            raw = str(getattr(message_obj, "content", "")).strip()
        elif isinstance(response, dict):
            raw = str(response.get("message", {}).get("content", "")).strip()

        if verbose:
            print(f"\n  [LLM raw scan response]\n  {raw}\n")

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0].strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            if verbose:
                print("  WARNING: invalid JSON from LLM, using detected-only fallback.")
            parsed = {}
    except Exception as exc:
        if verbose:
            print(f"  WARNING: LLM call failed ({exc}). Using detected-only fallback.")
        parsed = {}

    if not isinstance(parsed, dict):
        parsed = {}

    # ── Filters: validate against the closed sets, then guarantee the
    #    hard scan features are present. ──────────────────────────────
    raw_filters = parsed.get("filters")
    if not isinstance(raw_filters, dict):
        raw_filters = {}
    validated = _validate(raw_filters)
    include = validated.get("include") if isinstance(validated.get("include"), dict) else {}
    exclude = validated.get("exclude") if isinstance(validated.get("exclude"), dict) else {}

    # Deterministically reconcile color negation (the small LLM is
    # unreliable here): moves negated colors / categories out of include
    # and into exclude, and reports the detected color's standing so the
    # confidence number can be set accordingly.
    color_state = _reconcile_color_negation(user_answer, detected_color, include, exclude)

    _inject_detected_filters(include, detected_type, detected_body_type)
    filters = {"include": include, "exclude": exclude}

    # Budget, resolved deterministically from the raw answer (the small LLM is
    # unreliable here). Rides in `filters` as a SOFT signal: get_candidates adds
    # 1 to match_count for in-budget items, so it never dead-ends the pool.
    price_min, price_max = _extract_price_range(user_answer)
    if price_min is not None:
        filters["price_min"] = price_min
    if price_max is not None:
        filters["price_max"] = price_max

    if verbose:
        print(f"  [color state] {color_state}")

    # ── Weight VALUES are read back from the final filters / vision so
    #    they always match what the DB will actually be queried with. ──
    values = {
        "color": _first(include.get("color")) or detected_color,
        "type": _first(include.get("type")) or detected_type,
        "bodyType": _first(include.get("body_type")) or detected_body_type,
        "stock": None,   # inventory/popularity emphasis has no closed-set value
    }
    weights = _normalize_scan_weights(
        parsed.get("weights"), values, user_answer, color_state
    )

    # ── Semantic query text. ────────────────────────────────────────
    query = parsed.get("query")
    if not isinstance(query, str) or not query.strip():
        query = " ".join(
            part for part in (values["color"], _humanize_type(values["type"])) if part
        ).strip() or "clothing"

    return {"query": query.strip(), "filters": filters, "weights": weights}


# ══════════════════════════════════════════════════════════════
# CLI — quick manual test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Feature Weighting — Intent Analysis")
    print("=" * 60)
    print(f"  Refiner model: {OLLAMA_REFINER_MODEL}")
    print()

    detected_color = input("  Detected color     (default red)              > ").strip() or "red"
    detected_type = input("  Detected type      (default short_sleeve_top) > ").strip() or "short_sleeve_top"
    detected_body_type = input("  Detected body type (default hourglass)        > ").strip() or "hourglass"
    persona = input("  Persona            (default cruella)          > ").strip() or DEFAULT_PERSONA

    print()
    question = generate_intent_question(
        detected_color, detected_type, detected_body_type, persona=persona
    )
    print(f"  Assistant > {question}")

    user_answer = input("  You       > ").strip()
    if not user_answer:
        print("  (no answer, exiting)")
        sys.exit(0)

    import time

    start = time.perf_counter()
    scan = analyze_intent(
        detected_color, detected_type, detected_body_type, user_answer, verbose=True
    )
    elapsed = time.perf_counter() - start

    weights = scan["weights"]

    print("\n  Query:")
    print(f"    {scan['query']}")

    print("\n  Filters:")
    print(f"    {json.dumps(scan['filters'], ensure_ascii=False)}")

    print("\n  Weights:")
    for feature in ("color", "type", "bodyType", "stock"):
        data = weights[feature]
        print(f"    {feature:9s}: {str(data['value']):25s} - {data['importance']}%")
    print(f"\n  Sum: {sum(r['importance'] for r in weights.values())}%")
    print(f"  Elapsed (one merged call): {elapsed:.2f}s")
