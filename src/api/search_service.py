from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.api.custom_text_parser import parse_custom_query
from src.api.personas import normalize_persona
from src.recommendations.engine import RecommendationEngine

# Ensure legacy LNIAGIA modules that use absolute imports like "DB.*" are importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LNIAGIA_DIR = _PROJECT_ROOT / "LNIAGIA"
if _LNIAGIA_DIR.exists() and str(_LNIAGIA_DIR) not in sys.path:
    sys.path.insert(0, str(_LNIAGIA_DIR))

try:
    from LNIAGIA.query_parsing.llm_query_parser import (
        OLLAMA_MODEL,
        OLLAMA_REFINER_MODEL,
        OLLAMA_ROUTER_MODEL,
        parse_query,
    )
except Exception as exc:  # pragma: no cover - optional parser stack
    parse_query = None
    _PARSER_IMPORT_ERROR = exc
    OLLAMA_MODEL = "unavailable"
    OLLAMA_REFINER_MODEL = "unavailable"
    OLLAMA_ROUTER_MODEL = "unavailable"
else:
    _PARSER_IMPORT_ERROR = None

try:
    from LNIAGIA.DB.vector.VectorDBManager import (
        EMBEDDING_MODEL_NAME,
        _collection_exists,
        _get_client,
        _load_model,
        filtered_search,
    )
except Exception as exc:  # pragma: no cover - optional vector stack
    filtered_search = None
    _load_model = None
    _get_client = None
    _collection_exists = None
    EMBEDDING_MODEL_NAME = "unavailable"
    _VECTOR_IMPORT_ERROR = exc
else:
    _VECTOR_IMPORT_ERROR = None

_IMPORT_ERROR = _VECTOR_IMPORT_ERROR or _PARSER_IMPORT_ERROR


OVERRIDE_MARKERS = (
    "ignore the scan",
    "ignore my outfit",
    "ignore what i'm wearing",
    "forget the scan",
    "forget my outfit",
    "replace the vision result",
    "replace the scan",
    "start over",
    "something completely different",
    "not based on my outfit",
    "not based on the scan",
)

RESTORE_MARKERS = (
    "use my outfit again",
    "use the scan again",
    "go back to my outfit",
    "based on my outfit again",
    "use the vision result again",
)


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _merge_filters(base: dict, user: dict) -> dict:
    merged: Dict[str, Dict[str, List[str]]] = {}
    for section in ("include", "exclude"):
        target: Dict[str, List[str]] = {}
        for source in (base.get(section, {}), user.get(section, {})):
            for f, values in source.items():
                target[f] = _dedupe(target.get(f, []) + list(values))
        if target:
            merged[section] = target

    include = merged.get("include", {})
    exclude = merged.get("exclude", {})
    for f, ex_values in exclude.items():
        if f in include:
            include[f] = [v for v in include[f] if v not in ex_values]
            if not include[f]:
                del include[f]
    if include:
        merged["include"] = include
    elif "include" in merged:
        del merged["include"]
    return merged


# ── Unchanged helpers ──────────────────────────────────────────────────────

def _build_search_cards(hits: List) -> List[dict]:
    cards = []
    for hit in hits:
        payload = getattr(hit, "payload", {}) or {}
        item_type = payload.get("type", "item")
        brand = payload.get("brand")
        display_name = (
            f"{brand} {item_type.replace('_', ' ')}".title()
            if brand
            else item_type.replace("_", " ").title()
        )
        price_val = payload.get("price")
        if isinstance(price_val, (int, float)):
            price = f"EUR {price_val:.2f}"
        else:
            price = str(price_val) if price_val is not None else "N/A"

        cards.append(
            {
                "id": str(payload.get("item_id", getattr(hit, "id", uuid.uuid4().hex))),
                "name": display_name,
                "category": item_type,
                "price": price,
                "image_url": None,
                "reason": "Refined via semantic search",
                "score": round(float(getattr(hit, "score", 0.0)), 4),
                "brand": brand,
                "description": payload.get("description"),
                "metadata": {
                    key: payload.get(key)
                    for key in (
                        "color",
                        "style",
                        "pattern",
                        "material",
                        "fit",
                        "season",
                        "occasion",
                        "gender",
                        "age_group",
                    )
                    if payload.get(key) is not None
                },
            }
        )
    return cards


def _determine_mode(message: str, replace_vision: Optional[bool], current_mode: str) -> str:
    text = message.lower()
    if replace_vision is True:
        return "override"
    if replace_vision is False:
        return "vision"
    if any(marker in text for marker in OVERRIDE_MARKERS):
        return "override"
    if any(marker in text for marker in RESTORE_MARKERS):
        return "vision"
    return current_mode


def _strict_for_persona(persona: str) -> bool:
    return normalize_persona(persona) == "cruella"


def _build_contextual_query(
    state: "SearchSession",
    message: str,
    mode: str,
) -> str:
    """
    Build the semantic search query for chat/refinement turns.

    NOTE: user profile is intentionally NOT included here.  It is applied
    once — at session-creation time — to seed the base_filters and the
    initial recommendations.  Follow-up chat turns use only the detected
    outfit and the user's own words so the conversation stays responsive.
    """
    if mode == "override":
        return message

    worn = ", ".join(c.replace("_", " ") for c in state.detected_categories) or "an outfit"
    seed = ", ".join(c.replace("_", " ") for c in state.seed_categories)

    if seed:
        return (
            f"The user is wearing {worn}. Look for items that complement that outfit, "
            f"especially within these recommendation categories: {seed}. "
            f"User refinement: {message}"
        )
    return f"The user is wearing {worn}. Find complementary fashion items. User refinement: {message}"


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class SearchSession:
    id: str
    persona: str
    detected_categories: List[str]
    seed_categories: List[str]
    base_filters: dict
    initial_recommendations: List[dict] = field(default_factory=list)
    active_filters: dict = field(default_factory=dict)
    last_results: List[dict] = field(default_factory=list)
    mode: str = "vision"
    history: List[dict] = field(default_factory=list)
    # user_profile is NOT stored here - it is consumed once in create_session()
    # to seed base_filters, then discarded so chat/refine turns stay clean.


# ── Service ────────────────────────────────────────────────────────────────

class UnifiedSearchService:
    def __init__(self, recommender: RecommendationEngine):
        self.recommender = recommender
        self._sessions: Dict[str, SearchSession] = {}
        self._embedding_model = None

    def set_embedding_model(self, model) -> None:
        self._embedding_model = model

    def availability(self) -> dict:
        parser_ready = parse_query is not None
        vector_ready = filtered_search is not None and self._vector_collection_ready()
        return {
            "parser_ready": parser_ready,
            "vector_ready": vector_ready,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "parser_model": OLLAMA_MODEL,
            "parser_refiner_model": OLLAMA_REFINER_MODEL,
            "interaction_model": OLLAMA_ROUTER_MODEL,
            "import_error": str(_IMPORT_ERROR) if _IMPORT_ERROR else None,
        }

    def warmup(self) -> dict:
        info = self.availability()
        embedding_loaded = self._get_embedding_model() is not None
        parser_warmed = False
        parser_warning = None

        if parse_query is not None:
            try:
                parse_query("show me neutral smart casual options", verbose=False)
                parser_warmed = True
            except Exception as exc:  # pragma: no cover
                parser_warning = str(exc)

        return {
            **info,
            "embedding_loaded": embedding_loaded,
            "parser_warmed": parser_warmed,
            "parser_warning": parser_warning,
        }

    def create_session(
        self,
        detected_categories: List[str],
        recommendations: Optional[List[dict]] = None,
        persona: str = "cruella",
        user_profile: Optional[dict] = None,
    ) -> SearchSession:
        persona = normalize_persona(persona)
        detected_categories = _dedupe(detected_categories)

        if recommendations is not None:
            recs = recommendations
        elif detected_categories:
            recs = self.recommender.recommend(detected_categories)
        else:
            recs = []

        seed_categories = _dedupe(
            [rec.get("category") for rec in recs if rec.get("category")]
        )

        # Base filters contain only seed types. Profile is applied once in
        # search_detected_items() (the initial CV query) and not persisted here.
        base_filters = {"include": {"type": seed_categories}} if seed_categories else {}

        # ── DEBUG ──────────────────────────────────────────────────────────────
        print("\n" + "="*60)
        print("🔍  SESSION CREATED")
        print(f"    persona          : {persona}")
        print(f"    detected         : {detected_categories}")
        print(f"    seed_categories  : {seed_categories}")
        print(f"    user_profile     : {user_profile}")
        print(f"    base_filters     : {base_filters}")
        print("="*60 + "\n")
        # ── END DEBUG ──────────────────────────────────────────────────────────

        session = SearchSession(
            id=uuid.uuid4().hex,
            persona=persona,
            detected_categories=detected_categories,
            seed_categories=seed_categories,
            base_filters=base_filters,
            initial_recommendations=recs,
            last_results=recs,
            active_filters=base_filters,
            mode="vision" if detected_categories else "override",
        )
        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SearchSession]:
        return self._sessions.get(session_id)

    def refine(
        self,
        session_id: str,
        message: str,
        history: Optional[List[dict]] = None,
        replace_vision: Optional[bool] = None,
        parser_backend: str = "llm",
        state: Optional[dict] = None,
    ) -> dict:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(session_id)

        mode = _determine_mode(message, replace_vision, session.mode)
        strict = _strict_for_persona(session.persona)
        parsed_filters = self._safe_parse(message, parser_backend=parser_backend)
        effective_filters = (
            parsed_filters
            if mode == "override"
            else _merge_filters(session.base_filters, parsed_filters)
        )
        # Pass the session's cached user_profile into the query builder
        search_query = _build_contextual_query(session, message, mode)

        results, warning = self._run_search(search_query, effective_filters, strict)
        if not results:
            results = session.initial_recommendations if mode == "vision" else []

        session.mode = mode
        session.active_filters = effective_filters
        session.last_results = results
        session.history = list(history or []) + [{"role": "user", "content": message}]

        reply = self._build_reply(mode, warning, results)
        return {
            "reply": reply,
            "mode": mode,
            "persona": session.persona,
            "session_id": session.id,
            "active_filters": effective_filters,
            "results": results,
            "strict": strict,
            "warning": warning,
            "state": {
                "query": search_query,
                "filters": effective_filters,
                "persona": session.persona,
                "parser_backend": parser_backend,
            },
        }

    def _build_reply(self, mode: str, warning: Optional[str], results: List[dict]) -> str:
        if warning:
            base = f"I kept the conversation going, but search is degraded: {warning}."
        elif mode == "override":
            base = "I replaced the scan context and searched from your message alone."
        else:
            base = "I refined the results using your scan as context."

        if results:
            return f"{base} I found {len(results)} option(s) to explore next."
        return (
            f"{base} I found 0 results with the current filters, "
            "so you may want to try a more specific refinement."
        )

    def _safe_parse(self, message: str, parser_backend: str = "llm") -> dict:
        if parser_backend == "custom":
            return parse_custom_query(message)
        if parse_query is None:
            return {}
        try:
            return parse_query(message, verbose=False)
        except Exception as exc:  # pragma: no cover
            print(f"Warning: query parser failed: {exc}")
            return {}

    def _run_search(self, query: str, filters: dict, strict: bool) -> tuple[List[dict], Optional[str]]:
        if filtered_search is None:
            return [], "the integrated search modules are not available"
        if not self._vector_collection_ready():
            return [], "the vector collection has not been built yet"

        model = self._get_embedding_model()
        if model is None:
            return [], "the embedding model could not be loaded"

        try:
            hits = filtered_search(query, filters, model, strict=strict)

            # ── DEBUG ──────────────────────────────────────────────────────────
            print("\n" + "="*60)
            print("🔍  RAW HITS FROM VECTOR SEARCH")
            print(f"    query   : {query}")
            print(f"    filters : {filters}")
            print(f"    strict  : {strict}")
            print(f"    hits    : {len(hits)}")
            for i, hit in enumerate(hits[:5]):   # print first 5 only
                payload = getattr(hit, "payload", {}) or {}
                print(f"\n    [{i}] score={getattr(hit, 'score', '?'):.4f}")
                for key in ("type", "color", "style", "material", "season", "occasion", "gender", "age_group"):
                    print(f"         {key:12}: {payload.get(key, '—')}")
            print("="*60 + "\n")
            # ── END DEBUG ──────────────────────────────────────────────────────

            return _build_search_cards(hits), None
        except Exception as exc:
            return [], str(exc)

    def _get_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model
        if _load_model is None:
            return None
        try:
            self._embedding_model = _load_model()
        except Exception as exc:  # pragma: no cover
            print(f"Warning: embedding model load failed: {exc}")
            return None
        return self._embedding_model

    def _vector_collection_ready(self) -> bool:
        if _get_client is None or _collection_exists is None:
            return False
        try:
            client = _get_client()
            try:
                return _collection_exists(client)
            finally:
                client.close()
        except Exception as exc:  # pragma: no cover
            print(f"Warning: vector collection check failed: {exc}")
            return False
