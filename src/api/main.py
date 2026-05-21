"""
src/api/main.py
───────────────
FastAPI application.

Endpoints:
  GET  /                    → HTML dashboard (serves frontend/index.html)
  GET  /health              → health check
  POST /api/detect/image    → single image upload detection
  POST /api/mobile/scan     → mobile alias for image scan
  WS   /ws/camera           → real-time WebSocket camera stream

Run:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import base64
import hashlib
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import cv2
import httpx
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    AuthResponse,
    BodyAnalysisResponse,
    ChatRequest,
    ChatResponse,
    ConversationRequest,
    DetectionResponse,
    HealthResponse,
    LoginRequest,
    RegisterRequest,
    SessionResponse,
    SessionStartRequest,
)
from src.api.auth import create_access_token, get_optional_user_id
from src.api.personas import PERSONA_CONFIGS, normalize_persona
from src.api.search_service import UnifiedSearchService
from src.detection.camera import CameraStream
from src.detection.detector import BaseDetector, FashionDetector
from src.detection.fashionnet_detector import FashionNetDetector
from src.pose_analyzer import PoseAnalyzer
from src.recommendations.engine import RecommendationEngine

# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="FashionSense API",
    description="Real-time clothing detection & recommendation system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ──────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Shared singletons ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DB_PATH = str(PROJECT_ROOT / "LNIAGIA" / "DB" / "SQLLite" / "clothing.db")


def _find_weights() -> str:
    env = os.getenv("MODEL_WEIGHTS")
    if env:
        full = PROJECT_ROOT / "models" / "weights" / env / "weights" / "best.pt"
        return str(full)
    candidates = [
        "yolov8l_fashion/weights/best.pt",
        "yolov8m_fashion/weights/best.pt",
        "yolov8s_fashion/weights/best.pt",
        "yolov8n_fashion/weights/best.pt",
    ]
    weights_dir = PROJECT_ROOT / "models" / "weights"
    for c in candidates:
        full = weights_dir / c
        if full.exists():
            print(f"Found trained weights: {full}")
            return str(full)
    print("⚠️  No fine-tuned weights found, using base yolov8n.pt")
    return "yolov8n.pt"


WEIGHTS_PATH = _find_weights()

detector: BaseDetector = None
recommender: RecommendationEngine = None
camera: CameraStream = None
whisper_model = None
search_service: UnifiedSearchService = None
pose_analyzer: PoseAnalyzer | None = None
detectors_by_persona: dict[str, BaseDetector] = {}
cameras_by_persona: dict[str, CameraStream] = {}


def _find_ffmpeg_exe() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))
    candidates = [
        local_appdata / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
        local_appdata / "Programs" / "ffmpeg" / "bin" / "ffmpeg.exe",
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/ffmpeg/ffmpeg.exe"),
        Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _find_fashionnet_weights() -> Optional[str]:
    env = os.getenv("FASHIONNET_WEIGHTS")
    if env:
        full = PROJECT_ROOT / "models" / "weights" / env / "best.pt"
        if full.exists():
            return str(full)
        print(f"⚠️  FASHIONNET_WEIGHTS={env!r} not found at {full}, falling back")
    candidate = PROJECT_ROOT / "models" / "weights" / "fashionnet" / "best.pt"
    return str(candidate) if candidate.exists() else None


def _resolve_detector(persona: str) -> BaseDetector:
    return detectors_by_persona.get(normalize_persona(persona)) or detector


def _resolve_camera(persona: str) -> CameraStream:
    return cameras_by_persona.get(normalize_persona(persona)) or camera


def _persona_text_backend(persona: str) -> str:
    key = normalize_persona(persona)
    return PERSONA_CONFIGS[key].text_backend


def _strict_for_persona(persona: str) -> bool:
    return normalize_persona(persona) == "cruella"


def _resolve_assistant_mode(payload: ConversationRequest) -> str:
    if isinstance(payload.assistant_mode, str) and payload.assistant_mode.strip():
        return normalize_persona(payload.assistant_mode)
    if isinstance(payload.state, dict):
        state_mode = payload.state.get("assistant_mode")
        if isinstance(state_mode, str) and state_mode.strip():
            return normalize_persona(state_mode)
    return "cruella"


# ── User-profile helpers ───────────────────────────────────────────────────

def _get_user_profile_from_db(user_id: int) -> Optional[dict]:
    """
    Fetch a user's preference profile from the SQLite DB.
    Returns None if the user doesn't exist or the DB is unreachable.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT age_group, gender,
                   favorite_colors, favorite_styles, favorite_materials,
                   preferred_seasons, preferred_occasions
            FROM   users
            WHERE  id = ?
            """,
            (int(user_id),),
        )
        row = cur.fetchone()
        conn.close()
    except Exception as exc:
        print(f"⚠️  Could not fetch user profile for user_id={user_id}: {exc}")
        return None

    if row is None:
        return None

    def _split(value: Optional[str]) -> list[str]:
        if not value:
            return []
        return [v.strip() for v in value.split(",") if v.strip()]

    return {
        "age_group":           row["age_group"] or "",
        "gender":              row["gender"] or "",
        "favorite_colors":     _split(row["favorite_colors"]),
        "favorite_styles":     _split(row["favorite_styles"]),
        "favorite_materials":  _split(row["favorite_materials"]),
        "preferred_seasons":   _split(row["preferred_seasons"]),
        "preferred_occasions": _split(row["preferred_occasions"]),
    }


def _get_user_profile(user_id: Optional[int]) -> Optional[dict]:
    """
    Resolve a user_id (already extracted + validated by JWT) to a DB profile.
    Returns None for guests or users with no profile row.
    """
    if user_id is None:
        return None
    return _get_user_profile_from_db(user_id)


# ── Existing DB-backed recommendation helper (unchanged) ───────────────────

# def _db_backed_recommendations_sync(detected_categories: list[str]) -> list[dict]:
#     try:
#         from LNIAGIA.search_app import search_detected_items
#     except Exception as exc:
#         print(f"⚠️  Could not import DB search helper: {exc}")
#         return []

#     print(f"\n🔍  Fetching DB recommendations for detected categories: {detected_categories}")
#     ranked_results, warning = search_detected_items(detected_categories, strict=False)
#     if warning:
#         print(f"⚠️  DB recommendation warning: {warning}")

#     return _format_conversation_results(ranked_results)


def _preload_search_embeddings_sync() -> Optional[str]:
    global search_service

    if search_service is None:
        return "search service is not initialized"

    try:
        from LNIAGIA.DB.vector.VectorDBManager import _load_model
        from LNIAGIA.search_app import set_conversation_embedding_model
    except Exception as exc:
        return f"could not import embedding preload helpers ({exc})"

    try:
        shared_model = _load_model()
    except Exception as exc:
        return f"embedding model load failed ({exc})"

    try:
        search_service.set_embedding_model(shared_model)
        set_conversation_embedding_model(shared_model)
    except Exception as exc:
        return f"could not share embedding model across search paths ({exc})"

    return None


@app.on_event("startup")
async def startup():
    global detector, recommender, camera, whisper_model, search_service
    global detectors_by_persona, cameras_by_persona, pose_analyzer

    weights = WEIGHTS_PATH
    print(f"\n🚀  Loading model from: {weights}")
    cruella_detector = FashionDetector(weights=weights, conf_thres=0.60)

    fashionnet_weights = _find_fashionnet_weights()
    if fashionnet_weights:
        print(f"\n🧵  Loading FashionNet from: {fashionnet_weights}")
        try:
            edna_detector = FashionNetDetector(weights=fashionnet_weights, conf_thres=0.35)
        except Exception as exc:
            print(f"⚠️  FashionNet failed to load, falling back to Cruella detector: {exc}")
            edna_detector = cruella_detector
    else:
        print("⚠️  No FashionNet checkpoint found, Edna will reuse Cruella vision backend")
        edna_detector = cruella_detector

    detector = cruella_detector
    detectors_by_persona = {"cruella": cruella_detector, "edna": edna_detector}

    recommender = RecommendationEngine(top_k=5)
    search_service = UnifiedSearchService(recommender)
    pose_analyzer = PoseAnalyzer()
    if pose_analyzer.is_available():
        print(f"🧍  Pose analyzer ready ({pose_analyzer.model_path})")
    else:
        print("⚠️  Pose analyzer model not available; body analysis endpoints will be limited")

    preload_warning = await run_in_threadpool(_preload_search_embeddings_sync)
    if preload_warning:
        print(f"⚠️  Embedding preload skipped: {preload_warning}")
    else:
        print("🧠  Embedding model preloaded for API and conversation search")

    cameras_by_persona = {
        key: CameraStream(
            detector_instance,
            recommender,
            recommendation_resolver=_db_backed_recommendations_with_profile_sync,
            body_analysis_resolver=_run_body_analysis if pose_analyzer.is_available() else None,
            source=0,
        )
        for key, detector_instance in detectors_by_persona.items()
    }
    camera = cameras_by_persona["cruella"]
    for key, camera_instance in cameras_by_persona.items():
        try:
            camera_instance.warmup()
            print(f"📷  Camera warmed and ready for {key}")
        except Exception as e:
            print(f"⚠️  Camera warmup failed for {key}: {e}")

    async def _load_whisper():
        global whisper_model
        try:
            from faster_whisper import WhisperModel

            loop = asyncio.get_event_loop()
            print("🎙️  Loading Whisper model (may download on first run)...")
            whisper_model = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: WhisperModel("base", device="cpu", compute_type="int8"),
                ),
                timeout=60,
            )
            print("🎙️  Whisper speech-to-text loaded")
        except asyncio.TimeoutError:
            print("⚠️  Whisper loading timed out after 60s — voice input disabled")
        except Exception as e:
            print(f"⚠️  Whisper not available: {e}")

    asyncio.create_task(_load_whisper())
    print("✅  API ready\n")


@app.on_event("shutdown")
async def shutdown():
    global cameras_by_persona
    for camera_instance in cameras_by_persona.values():
        camera_instance.shutdown()


# ── Static / proxy routes ───────────────────────────────────────────────────

_GOOGLE_IMAGE_HOSTS = {"drive.google.com", "drive.usercontent.google.com"}


def _extract_google_drive_file_id(raw_url: str) -> Optional[str]:
    parsed = urlparse(raw_url)
    host = parsed.netloc.lower()
    if host not in _GOOGLE_IMAGE_HOSTS:
        return None
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) >= 3 and path_parts[0] == "file" and path_parts[1] == "d":
        return path_parts[2]
    query_id = parse_qs(parsed.query).get("id")
    if query_id and query_id[0].strip():
        return query_id[0].strip()
    return None


def _normalize_google_drive_image_url(raw_url: str) -> str:
    file_id = _extract_google_drive_file_id(raw_url)
    if not file_id:
        return raw_url
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=view"


@app.get("/", response_class=HTMLResponse)
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return HTMLResponse("<h1>FashionSense API</h1><p>Visit <a href='/docs'>/docs</a></p>")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model_loaded=detector is not None)


def _run_body_analysis(frame: np.ndarray) -> tuple[dict | None, str | None]:
    """Run pose analysis and return structured JSON plus an optional annotated frame."""
    if pose_analyzer is None:
        return None, None

    analysis = pose_analyzer.analyze(frame, draw_overlay=True, include_landmarks=True)
    if analysis.landmarks_detected == 0:
        return analysis.to_dict(include_landmarks=True), None

    body_b64 = None
    if analysis.annotated_image is not None:
        ok, body_buf = cv2.imencode(".jpg", analysis.annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            body_b64 = base64.b64encode(body_buf).decode("utf-8")
    return analysis.to_dict(include_landmarks=True), body_b64


@app.get("/api/image-proxy")
async def image_proxy(url: str):
    raw_url = (url or "").strip()
    if not raw_url:
        raise HTTPException(status_code=400, detail="Missing 'url' query parameter")

    normalized_url = _normalize_google_drive_image_url(raw_url)
    parsed = urlparse(normalized_url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Only http/https image URLs are allowed")
    if parsed.netloc.lower() not in _GOOGLE_IMAGE_HOSTS:
        raise HTTPException(status_code=400, detail="Only Google Drive image hosts are allowed")

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            upstream = await client.get(normalized_url, headers={"User-Agent": "Mozilla/5.0"})
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch image: {exc}") from exc

    if upstream.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Image host returned status {upstream.status_code}")

    content_type = (upstream.headers.get("content-type") or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=502, detail="Image URL did not return image content")

    media_type = content_type.split(";", 1)[0]
    return Response(
        content=upstream.content,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ── Session endpoints ────────────────────────────────────────────────────────

@app.post("/api/session/start", response_model=SessionResponse)
async def start_session(
    payload: SessionStartRequest,
    user_id: Optional[int] = Depends(get_optional_user_id),
):
    persona = normalize_persona(payload.persona)
    user_profile = await run_in_threadpool(_get_user_profile, user_id)

    recs = await run_in_threadpool(
        _db_backed_recommendations_with_profile_sync,
        payload.detected_categories or [],
        user_profile,                           # ← always pass, even if categories empty
    )

    # Fall back to whatever the frontend sent if the fetch returned nothing
    if not recs:
        recs = payload.recommendations or []

    session = search_service.create_session(
        detected_categories=payload.detected_categories,
        recommendations=recs,
        persona=persona,
        user_profile=user_profile,
    )

    return SessionResponse(
        session_id=session.id,
        mode=session.mode,
        persona=session.persona,
        detected_categories=session.detected_categories,
        seed_categories=session.seed_categories,
        active_filters=session.active_filters,
        results=session.last_results,
    )

@app.get("/api/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    session = search_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(
        session_id=session.id,
        mode=session.mode,
        persona=session.persona,
        detected_categories=session.detected_categories,
        seed_categories=session.seed_categories,
        active_filters=session.active_filters,
        results=session.last_results,
    )


@app.post("/api/chat/warmup")
async def warmup_chat():
    return search_service.warmup()

# ── NEW: profile-aware DB recommendation fetch ─────────────────────────────

def _db_backed_recommendations_with_profile_sync(
    detected_categories: list[str],
    user_profile: Optional[dict] = None,
) -> list[dict]:
    print(f"DEBUG user_profile received: {user_profile}")  # ← add this

    try:
        from LNIAGIA.search_app import search_detected_items
    except Exception as exc:
        print(f"⚠️  Could not import DB search helper: {exc}")
        return []

    # Build filter kwargs from profile if available
    profile_filters = {}
    if user_profile:
        if user_profile.get("favorite_colors"):
            profile_filters["colors"] = user_profile["favorite_colors"]
        if user_profile.get("favorite_styles"):
            profile_filters["styles"] = user_profile["favorite_styles"]
        if user_profile.get("favorite_materials"):
            profile_filters["materials"] = user_profile["favorite_materials"]
        if user_profile.get("preferred_seasons"):
            profile_filters["seasons"] = user_profile["preferred_seasons"]
        if user_profile.get("preferred_occasions"):
            profile_filters["occasions"] = user_profile["preferred_occasions"]
        if user_profile.get("gender"):
            profile_filters["gender"] = user_profile["gender"]
        if user_profile.get("age_group"):
            profile_filters["age_group"] = user_profile["age_group"]

    # ── DEBUG ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("🔍  INITIAL REC QUERY")
    print(f"    detected_categories : {detected_categories}")
    print(f"    profile_filters     : {profile_filters}")
    print("="*60 + "\n")
    # ── END DEBUG ──────────────────────────────────────────────────────────

    ranked_results, warning = search_detected_items(
        detected_categories,
        strict=False,
        colors=profile_filters.get("colors"),
        styles=profile_filters.get("styles"),
        materials=profile_filters.get("materials"),
        seasons=profile_filters.get("seasons"),
        occasions=profile_filters.get("occasions"),
        gender=profile_filters.get("gender"),
        age_group=profile_filters.get("age_group")
    )
    if warning:
        print(f"⚠️  DB recommendation warning: {warning}")

    return _format_conversation_results(ranked_results)

# ── Core detection implementation ─────────────────────────────────────────────

async def _detect_image_impl(
    file: UploadFile,
    persona: str = "cruella",
    user_profile: Optional[dict] = None,       # ← NEW
) -> DetectionResponse:
    """
    Shared implementation for image/mobile scan uploads.

    When `user_profile` is provided (logged-in user), it is:
      1. Stored on the search session so every subsequent refinement query
         is enriched with the user's style preferences.
      2. Used to pre-filter recommendations toward the user's preferred
         colors, styles, materials, seasons and occasions.
    """
    persona = normalize_persona(persona)
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    active_detector = _resolve_detector(persona)
    result = active_detector.detect(frame)
    cats = list({d.class_name for d in result.detections})
    body_analysis: dict | None = None
    body_annotated_frame: str | None = None

    if pose_analyzer is not None and pose_analyzer.is_available():
        try:
            body_analysis, body_annotated_frame = await run_in_threadpool(
                _run_body_analysis,
                frame.copy(),
            )
        except Exception as exc:
            print(f"⚠️  Body analysis failed: {exc}")
            body_analysis = {
                "measurements": {
                    "shoulder_width": 0.0,
                    "hip_width": 0.0,
                    "waist_width": 0.0,
                    "torso_length": 0.0,
                    "leg_length": 0.0,
                    "shoulder_hip_ratio": 0.0,
                    "waist_hip_ratio": 0.0,
                },
                "body_shape": "unknown",
                "landmarks_detected": 0,
                "confidence": 0.0,
                "warnings": [f"Body analysis failed: {exc}"],
                "landmarks": [],
            }

    # ── CHANGED: pass user_profile into the recommendation fetch ──────────
    recs = await run_in_threadpool(
        _db_backed_recommendations_with_profile_sync,
        cats,
        user_profile,   # ← was just cats before
    )
    
    if not isinstance(recs, list):
        recs = []

    # Pass user_profile into create_session so it enriches base_filters
    # and is persisted on the session for follow-up chat turns.
    session = search_service.create_session(
        cats,
        recs,
        persona=persona,
        user_profile=user_profile,          # ← NEW
    )

    # Encode annotated frame
    annotated = active_detector.draw(frame, result)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_frame = base64.b64encode(buf).decode("utf-8")

    return DetectionResponse(
        detections=[
            {
                "class_id":   d.class_id,
                "class_name": d.class_name,
                "confidence": round(d.confidence, 3),
                "bbox":       d.bbox,
                "color":      d.color,            # RGB tuple
                "color_name": d.color_name,
            }
            for d in result.detections
        ],
        recommendations=recs,
        inference_ms=round(result.inference_ms, 1),
        annotated_frame=b64_frame,
        body_analysis=body_analysis,
        body_annotated_frame=body_annotated_frame,
        session_id=session.id,
        persona=persona,
    )


@app.post("/api/detect/image", response_model=DetectionResponse)
async def detect_image(
    persona: str = "cruella",
    file: UploadFile = File(...),
    user_id: Optional[int] = Depends(get_optional_user_id),
):
    """
    Accepts an uploaded image, runs detection, returns detections + recommendations.

    If the request carries a valid JWT in the Authorization header, the user's
    saved style preferences are loaded from the DB and used to personalise the
    initial recommendations only.  Chat/refinement turns are not affected.
    """
    user_profile = await run_in_threadpool(_get_user_profile, user_id)
    if user_profile:
        print(f"👤  Personalising detection for user_id={user_id}")
    return await _detect_image_impl(file, persona=persona, user_profile=user_profile)


@app.post("/api/mobile/scan", response_model=DetectionResponse)
async def mobile_scan(
    persona: str = "cruella",
    file: UploadFile = File(...),
    user_id: Optional[int] = Depends(get_optional_user_id),
):
    """
    Mobile-friendly alias for image scan uploads from native clients.
    Supports the same user-profile personalisation as /api/detect/image.
    """
    user_profile = await run_in_threadpool(_get_user_profile, user_id)
    if user_profile:
        print(f"👤  Personalising mobile scan for user_id={user_id}")
    return await _detect_image_impl(file, persona=persona, user_profile=user_profile)


# ── Conversation / chat endpoints ─────────────────────────────────────────────

@app.post("/api/analyze/body", response_model=BodyAnalysisResponse)
async def analyze_body(file: UploadFile = File(...)):
    """
    Accepts an uploaded image and returns pose landmarks, normalized body
    measurements, and a heuristic fashion body-shape classification.
    """
    if pose_analyzer is None or not pose_analyzer.is_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "Pose analyzer is not available. Add a MediaPipe Pose Landmarker "
                "model at models/weights/mediapipe/pose_landmarker_heavy.task "
                "or set POSE_LANDMARKER_MODEL_PATH."
            ),
        )

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    analysis_dict, annotated_frame = await run_in_threadpool(_run_body_analysis, frame)
    if analysis_dict is None:
        raise HTTPException(status_code=500, detail="Body analysis returned no result")

    return BodyAnalysisResponse(
        measurements=analysis_dict["measurements"],
        body_shape=analysis_dict["body_shape"],
        landmarks_detected=analysis_dict["landmarks_detected"],
        confidence=analysis_dict["confidence"],
        warnings=analysis_dict.get("warnings", []),
        landmarks=analysis_dict.get("landmarks", []),
        annotated_frame=annotated_frame,
    )


def _get_detected_type(session_id: Optional[str], detected_categories: list[str]) -> Optional[str]:
    if detected_categories:
        return detected_categories[0]
    if session_id:
        session = search_service.get_session(session_id)
        if session and session.detected_categories:
            return session.detected_categories[0]
    return None


def _format_conversation_results(raw_results: list[dict]) -> list[dict]:
    formatted: list[dict] = []
    for entry in raw_results:
        item = entry.get("item") or {}
        item_type = str(item.get("type") or item.get("category") or "item")
        brand = item.get("brand")
        name = (
            item.get("name")
            or item.get("title")
            or item.get("product_name")
            or (
                f"{brand} {item_type.replace('_', ' ')}".title()
                if brand
                else item_type.replace("_", " ").title()
            )
        )
        price_value = item.get("price")
        if isinstance(price_value, (int, float)):
            price = f"EUR {price_value:.2f}"
        else:
            price = str(price_value) if price_value is not None else "N/A"

        reason_parts = []
        for key in ("color", "style", "occasion", "season", "material"):
            value = item.get(key)
            if value:
                reason_parts.append(str(value))
        reason = " | ".join(reason_parts) if reason_parts else "Refined via semantic search"

        formatted.append(
            {
                "id": str(entry.get("item_id") or item.get("id") or item.get("item_id") or ""),
                "name": str(name),
                "category": item_type,
                "price": price,
                "image_url": item.get("image_url") or item.get("image"),
                "reason": reason,
                "score": round(float(entry.get("score", 0.0)), 4),
                "brand": brand,
                "description": (
                    item.get("description")
                    or item.get("short_description")
                    or item.get("summary")
                ),
                "metadata": {
                    key: item.get(key)
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
                    if item.get(key) is not None
                },
            }
        )
    return formatted


async def _run_conversation_search(payload: ConversationRequest) -> dict:
    try:
        from LNIAGIA.search_app import run_conversation_model
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not import conversation search module: {exc}",
        ) from exc

    assistant_mode = _resolve_assistant_mode(payload)
    strict = _strict_for_persona(assistant_mode)

    result = await run_in_threadpool(
        run_conversation_model,
        detected_type=payload.detected_type,
        user_input=payload.message,
        conversation_state=payload.state,
        strict=strict,
        assistant_mode=assistant_mode,
    )

    if not result.get("ok", False):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Conversation search failed."),
        )

    return result


@app.post("/api/search/conversation")
async def search_conversation(payload: ConversationRequest, persona: str = "cruella"):
    assistant_mode = payload.assistant_mode or normalize_persona(persona)
    return await _run_conversation_search(
        ConversationRequest(
            detected_type=payload.detected_type,
            message=payload.message,
            strict=_strict_for_persona(assistant_mode),
            assistant_mode=assistant_mode,
            state=payload.state,
        )
    )


# ── Confidence threshold endpoints ─────────────────────────────────────────────

@app.get("/api/conf")
async def get_conf(persona: str = "cruella"):
    active_detector = _resolve_detector(persona)
    return {"conf_thres": round(active_detector.conf_thres, 2), "persona": normalize_persona(persona)}


@app.post("/api/conf/{value}")
async def set_conf(value: float, persona: str = "cruella"):
    """
    Update the detection confidence threshold live (0.01 – 0.99).
    Affects both the live camera stream and result filtering.
    """
    if not (0.01 <= value <= 0.99):
        raise HTTPException(status_code=400, detail="conf must be between 0.01 and 0.99")
    active_detector = _resolve_detector(persona)
    active_detector.conf_thres = value
    return {"conf_thres": round(active_detector.conf_thres, 2), "persona": normalize_persona(persona)}


# ── Voice transcription ─────────────────────────────────────────────────────────

_WHISPER_JUNK = {
    "you", "thank you", "thanks", "thanks for watching",
    "thank you for watching", "bye", "goodbye", "hello",
    "the end", "subtitle", "subtitles", "music",
    "applause", "laughter", "", "...",
}


@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Accept an audio file (webm/opus from browser) and return transcribed text."""
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")

    data = await audio.read()
    print(f"🎙️  Received audio: {len(data)} bytes, type: {audio.content_type}")
    if not data:
        raise HTTPException(status_code=400, detail="No audio data received")

    ffmpeg_path = _find_ffmpeg_exe()
    if ffmpeg_path is None:
        raise HTTPException(status_code=503, detail="ffmpeg is not installed or not available on PATH")

    suffix = Path(audio.filename or "voice.webm").suffix.lower() or ".webm"
    if suffix not in {".webm", ".wav", ".ogg", ".mp3", ".m4a", ".mp4"}:
        suffix = ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        source_path = tmp.name

    wav_path = source_path.rsplit(".", 1)[0] + ".wav"
    import subprocess

    result = subprocess.run(
        [ffmpeg_path, "-y", "-i", source_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        print(f"🎙️  ffmpeg stderr: {result.stderr}")
        if os.path.exists(source_path):
            os.unlink(source_path)
        raise HTTPException(status_code=400, detail="Could not decode uploaded audio")

    wav_size = os.path.getsize(wav_path) if os.path.exists(wav_path) else 0
    print(f"🎙️  Converted to WAV: {wav_size} bytes")
    os.unlink(source_path)

    try:
        segments, info = whisper_model.transcribe(wav_path, beam_size=5, language="en")
        print(f"🎙️  Language: {info.language} ({info.language_probability:.0%}), duration: {info.duration:.1f}s")

        parts = []
        for seg in segments:
            print(f"🎙️  Segment [{seg.start:.1f}s-{seg.end:.1f}s] no_speech={seg.no_speech_prob:.2f}: {seg.text!r}")
            parts.append(seg.text.strip())

        text = " ".join(parts).strip()
        if text.lower().rstrip(".!?, ") in _WHISPER_JUNK:
            print(f"🎙️  Filtered as hallucination: {text!r}")
            text = ""

        print(f"🎙️  Final text: {text!r}")
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    return {"text": text}


# ── Chat endpoint ────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    persona = normalize_persona(payload.persona)
    session_id = payload.session_id
    if not session_id:
        session = search_service.create_session(
            detected_categories=payload.detected_categories,
            recommendations=payload.recommendations,
            persona=persona,
        )
        session_id = session.id
    else:
        session = search_service.get_session(session_id)
        if session is not None:
            persona = session.persona

    detected_type = _get_detected_type(session_id, payload.detected_categories)
    assistant_mode = payload.assistant_mode or persona
    conversation_result = await _run_conversation_search(
        ConversationRequest(
            detected_type=detected_type,
            message=payload.message,
            strict=_strict_for_persona(assistant_mode),
            assistant_mode=assistant_mode,
            state=payload.state,
        )
    )

    results = _format_conversation_results(conversation_result.get("results", []))
    state = conversation_result.get("state") or {}
    active_filters = state.get("filters") if isinstance(state.get("filters"), dict) else {}
    mode = "override" if payload.replace_vision else "vision"

    reply = str(conversation_result.get("reply") or "I processed your request.")
    warning = conversation_result.get("warning")

    model_mode = conversation_result.get("mode")
    if isinstance(model_mode, str) and model_mode.strip():
        persona = normalize_persona(model_mode)

    strict_value = _strict_for_persona(persona)
    if isinstance(state, dict) and isinstance(state.get("strict"), bool):
        strict_value = state.get("strict")

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        mode=mode,
        persona=persona,
        active_filters=active_filters,
        results=results,
        state=state,
        strict=strict_value,
        warning=warning,
    )


# ── Auth endpoints ───────────────────────────────────────────────────────────────

@app.post("/api/auth/login", response_model=AuthResponse)
async def login(payload: LoginRequest):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, password FROM users WHERE email = ?", (payload.email,))
    user = cur.fetchone()
    conn.close()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user_id, stored_password = user
    if not _verify_password(payload.password, stored_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return AuthResponse(
        success=True,
        message="Login successful",
        user_id=str(user_id),
        token=create_access_token(user_id),
    )


@app.post("/api/auth/register", response_model=AuthResponse)
async def register(payload: RegisterRequest):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email = ?", (payload.email,))
    if cur.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")

    password = _hash_password(payload.password)
    cur.execute(
        "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
        (payload.name, payload.email, password),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()

    return AuthResponse(
        success=True,
        message="Registration successful",
        user_id=str(user_id),
        token=create_access_token(user_id),
    )


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _verify_password(password: str, hashed: str) -> bool:
    return _hash_password(password) == hashed


# ── WebSocket camera endpoint ─────────────────────────────────────────────────

@app.websocket("/ws/camera")
async def websocket_camera(ws: WebSocket):
    await ws.accept()
    persona = normalize_persona(ws.query_params.get("persona"))

    token = ws.query_params.get("token")
    user_profile = None
    if token:
        try:
            from src.api.auth import _decode_token
            user_id = _decode_token(token)
            user_profile = await run_in_threadpool(_get_user_profile, user_id)
            if user_profile:
                print(f"👤  WebSocket personalised for user_id={user_id}")
        except Exception as e:
            print(f"⚠️  WebSocket token decode failed: {e}")

    active_camera = _resolve_camera(persona)
    print(f"🔌  WebSocket client connected ({persona})")

    async def send(text: str):
        try:
            await ws.send_text(text)
        except Exception:
            pass

    async def receive():
        try:
            return await ws.receive_text()
        except WebSocketDisconnect:
            return None

    try:
        await active_camera.run_session(send, receive, user_profile=user_profile)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"⚠️  WebSocket error: {e}")
    finally:
        print(f"🔌  WebSocket client disconnected ({persona})")
