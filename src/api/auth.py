"""
src/api/auth.py
───────────────
JWT authentication helpers.

Usage
-----
• Call `create_access_token(user_id)` after successful login/register.
• Protect endpoints with `get_current_user_id` as a FastAPI dependency.
• For optional auth (e.g. detection endpoints) use `get_optional_user_id`.

Environment variables
---------------------
JWT_SECRET_KEY  – signing secret (REQUIRED in production; has an insecure default
                  for local dev so the server still starts without configuration).
JWT_ALGORITHM   – default "HS256"
JWT_EXPIRE_MIN  – token lifetime in minutes, default 60 * 24 (24 h)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

try:
    from jose import JWTError, jwt
except SyntaxError as exc:
    raise RuntimeError(
        "Detected incompatible 'jose' package. "
        "Install 'python-jose[cryptography]' and remove 'jose'."
    ) from exc
except ImportError as exc:
    raise RuntimeError(
        "Missing dependency 'python-jose[cryptography]'. "
        "Install it in the active virtual environment."
    ) from exc

# ── Configuration ──────────────────────────────────────────────────────────────

_SECRET_KEY: str = os.getenv(
    "JWT_SECRET_KEY",
    "CHANGE_ME_in_production_use_a_long_random_string",   # dev-only fallback
)
_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MIN", str(60 * 24)))   # 24 h

# Warn loudly if still using the dev default in what looks like production.
if _SECRET_KEY == "CHANGE_ME_in_production_use_a_long_random_string":
    import warnings
    warnings.warn(
        "JWT_SECRET_KEY is not set – using insecure default. "
        "Set the JWT_SECRET_KEY environment variable before deploying.",
        stacklevel=1,
    )


# ── Token creation ─────────────────────────────────────────────────────────────

def create_access_token(user_id: int) -> str:
    """
    Return a signed JWT that encodes `user_id` in the `sub` claim.

    The token expires after JWT_EXPIRE_MIN minutes (default 24 h).
    """
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)


# ── Token verification ─────────────────────────────────────────────────────────

def _decode_token(token: str) -> int:
    """
    Decode and validate a JWT.  Returns the integer user_id or raises
    HTTPException(401) on any failure (expired, bad signature, missing claim …).
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
    except JWTError:
        raise credentials_exc

    sub = payload.get("sub")
    if not sub:
        raise credentials_exc

    try:
        return int(sub)
    except (TypeError, ValueError):
        raise credentials_exc


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    """Pull the raw token string out of 'Authorization: Bearer <token>'."""
    if not authorization:
        return None
    parts = authorization.strip().split(maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


# ── FastAPI dependencies ───────────────────────────────────────────────────────

def get_current_user_id(
    authorization: Optional[str] = Header(default=None),
) -> int:
    """
    **Required** auth dependency.  Raises 401 if the header is absent or invalid.

    Use this for endpoints that must be authenticated.
    """
    token = _extract_bearer(authorization)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing or malformed",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _decode_token(token)


def get_optional_user_id(
    authorization: Optional[str] = Header(default=None),
) -> Optional[int]:
    """
    **Optional** auth dependency.  Returns the user_id if a valid token is
    present, or None for unauthenticated requests (no exception raised).

    Use this for endpoints that work for both guests and logged-in users
    (e.g. /api/detect/image).
    """
    token = _extract_bearer(authorization)
    if token is None:
        return None
    try:
        return _decode_token(token)
    except HTTPException:
        return None   # treat invalid token as anonymous rather than hard-failing
