"""
Supabase Auth — JWT verification for protected API routes.

Öncelik 1: Yerel JWT (HS256) — legacy secret, anında.
Öncelik 2: JWKS (ES256/RS256) — yeni signing keys, cache'li yerel doğrulama.
Öncelik 3: Supabase Auth API — son çare, ağ bağımlı.
"""

import logging
import os
from typing import Optional

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)
SUPABASE_URL = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET")
security = HTTPBearer(auto_error=False)

_AUTH_API_TIMEOUT = 15.0
_JWKS_CACHE_TTL = 600  # 10 dakika


def _verify_via_jwt_legacy(token: str) -> Optional[dict]:
    """Legacy JWT secret (HS256) ile yerel doğrulama."""
    if not SUPABASE_JWT_SECRET:
        return None
    try:
        from jose import jwt, JWTError

        try:
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
            )
        except JWTError:
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                options={"verify_aud": False},
            )
        return payload
    except Exception:
        return None


def _verify_via_jwks(token: str) -> Optional[dict]:
    """JWKS ile yerel doğrulama (ES256/RS256) — ağ sadece ilk/key rotation'da."""
    if not SUPABASE_URL:
        return None
    jwks_url = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
    try:
        import jwt as pyjwt
        from jwt import PyJWKClient

        jwks_client = PyJWKClient(
            jwks_url,
            cache_keys=True,
            lifespan=_JWKS_CACHE_TTL,
        )
        header = pyjwt.get_unverified_header(token)
        kid = header.get("kid")
        if not kid:
            return None
        key = jwks_client.get_signing_key_from_jwt(token)
        try:
            payload = pyjwt.decode(
                token,
                key.key,
                algorithms=["ES256", "RS256"],
                audience="authenticated",
            )
        except pyjwt.PyJWTError:
            payload = pyjwt.decode(
                token,
                key.key,
                algorithms=["ES256", "RS256"],
                options={"verify_aud": False},
            )
        return payload
    except Exception as e:
        logger.debug("JWKS verification failed: %s", e)
        return None


def _verify_via_api(token: str) -> Optional[dict]:
    """Supabase Auth API — son çare, timeout/retry ile."""
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None
    url = f"{SUPABASE_URL}/auth/v1/user"
    headers = {"Authorization": f"Bearer {token}", "apikey": SUPABASE_ANON_KEY}
    for attempt in range(2):
        try:
            with httpx.Client(timeout=_AUTH_API_TIMEOUT) as client:
                resp = client.get(url, headers=headers)
        except Exception as e:
            logger.warning("Supabase auth attempt %d failed: %s", attempt + 1, e)
            continue
        if resp.status_code == 200:
            user = resp.json()
            return {"sub": user.get("id"), "email": user.get("email")}
        if resp.status_code == 401:
            return None
        logger.warning("Supabase auth returned %s", resp.status_code)
    return None


def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """
    Verify Supabase JWT. Returns payload with 'sub' (user_id) and 'email'.
    Auth disabled when neither API nor JWT secret is configured.
    """
    if not (SUPABASE_URL and SUPABASE_ANON_KEY) and not SUPABASE_JWT_SECRET:
        return None

    if not credentials:
        logger.warning("Auth 401: Missing Authorization header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # 1. Legacy JWT (HS256) — anında, ağ yok
    payload = _verify_via_jwt_legacy(token)
    if payload:
        return payload

    # 2. JWKS (ES256/RS256) — cache'li, ilk istekten sonra ağ yok
    payload = _verify_via_jwks(token)
    if payload:
        return payload

    # 3. Auth API — son çare
    payload = _verify_via_api(token)
    if payload:
        return payload

    logger.warning(
        "Auth 401: Token rejected (HS256/JWKS/API all failed). "
        "Check Fly.io: SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_JWT_SECRET"
    )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user_id(
    payload: Optional[dict] = Depends(verify_token),
) -> Optional[str]:
    """Extract user_id (UUID) from JWT payload."""
    if payload is None:
        return None
    return payload.get("sub")
