"""
src/api/schemas.py
──────────────────
Pydantic request / response models.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool


class DetectionResponse(BaseModel):
    detections:      List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    inference_ms:    float
    annotated_frame: Optional[str] = None   # base64 JPEG
    body_analysis:   Optional[Dict[str, Any]] = None
    body_annotated_frame: Optional[str] = None
    session_id:      Optional[str] = None
    persona:         Optional[str] = None


class BodyAnalysisResponse(BaseModel):
    body_shape: str
    measurements: Dict[str, float]
    confidence: float
    pose_validation: Dict[str, Any]
    landmarks_detected: int
    silhouette: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    landmarks: List[Dict[str, Any]] = Field(default_factory=list)
    annotated_frame: Optional[str] = None


class SessionStartRequest(BaseModel):
    detected_categories: List[str]
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    persona: str = "cruella"


class SessionResponse(BaseModel):
    session_id: str
    mode: str
    persona: str
    detected_categories: List[str]
    seed_categories: List[str]
    active_filters: Dict[str, Any]
    results: List[Dict[str, Any]]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    session_id: Optional[str] = None
    persona: str = "cruella"
    replace_vision: Optional[bool] = None
    strict: bool = Field(
        default=False,
        description="Deprecated. Strictness is derived from assistant_mode/persona.",
    )
    assistant_mode: Optional[str] = None
    state: Optional[Dict[str, Any]] = None
    detected_categories: List[str] = Field(default_factory=list)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)


class ConversationRequest(BaseModel):
    detected_type: Optional[str] = None
    message: Optional[str] = None
    strict: bool = Field(
        default=False,
        description="Deprecated. Strictness is derived from assistant_mode/persona.",
    )
    assistant_mode: Optional[str] = None
    state: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    mode: str
    persona: str
    active_filters: Dict[str, Any]
    results: List[Dict[str, Any]]
    state: Optional[Dict[str, Any]] = None
    strict: bool = False
    warning: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class AuthResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    user_id: Optional[str] = None
    token: Optional[str] = None


# ── Robot navigation ─────────────────────────────────────────────────────────

class RobotNavigateRequest(BaseModel):
    target: str = Field(..., description="Named location from coordinates.json, e.g. 't-shirt stand'")


class RobotNavigateResponse(BaseModel):
    status: str                       # "sent" | "error"
    target: str
    x: Optional[float] = None
    y: Optional[float] = None
    theta: Optional[float] = None
    message: Optional[str] = None


class RobotNavigateByCategoryRequest(BaseModel):
    category: str = Field(..., description="Clothing detection category, e.g. 'short_sleeve_top'")


class RobotStatusResponse(BaseModel):
    connected: bool
    known_locations: List[str]
    category_map: Dict[str, str] = Field(default_factory=dict)


# ── Multi-agent recommendation ────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    detected_color:     str = ""
    detected_type:      str = ""
    detected_body_type: str = ""
    user_answer:        str = ""
    user_gender:        str = ""
    user_height_cm:     Optional[float] = None


class RecommendResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    round_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    round_id: str
    item_id:  int
    size:     str = ""
    rating:   int = Field(..., ge=1, le=5, description="1–5 emoji satisfaction; 3 = neutral")


class FeedbackResponse(BaseModel):
    ok:      bool
    applied: bool = False          # False when the rating was neutral (no weight change)
    reason:  Optional[str] = None  # populated when ok is False
    policy:  Optional[Dict[str, Any]] = None
