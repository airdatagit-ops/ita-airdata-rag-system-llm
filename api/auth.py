"""API authentication module."""

import time
from typing import Any

import httpx
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

from config import config

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

_TOKEN_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _auth_mode() -> str:
    return (config.AUTH_MODE or "api_key").strip().lower()


def _introspection_url() -> str:
    if config.DRUPAL_OAUTH_INTROSPECTION_URL:
        return config.DRUPAL_OAUTH_INTROSPECTION_URL
    if config.DRUPAL_OAUTH_BASE_URL:
        return f"{config.DRUPAL_OAUTH_BASE_URL}/oauth/introspect"
    return ""


def _userinfo_url() -> str:
    if config.DRUPAL_OAUTH_USERINFO_URL:
        return config.DRUPAL_OAUTH_USERINFO_URL
    if config.DRUPAL_OAUTH_BASE_URL:
        return f"{config.DRUPAL_OAUTH_BASE_URL}/oauth/userinfo"
    return ""


def _cache_ttl(payload: dict[str, Any]) -> int:
    now = int(time.time())
    exp = payload.get("exp")
    if isinstance(exp, int) and exp > now:
        return max(0, min(config.DRUPAL_OAUTH_CACHE_TTL, exp - now))
    expires_in = payload.get("expires_in")
    if isinstance(expires_in, int):
        return max(0, min(config.DRUPAL_OAUTH_CACHE_TTL, expires_in))
    return config.DRUPAL_OAUTH_CACHE_TTL


async def _introspect_token(token: str) -> dict[str, Any] | None:
    introspection_url = _introspection_url()
    if not introspection_url:
        return None

    auth: tuple[str, str] | None = None
    if config.DRUPAL_OAUTH_CLIENT_ID and config.DRUPAL_OAUTH_CLIENT_SECRET:
        auth = (config.DRUPAL_OAUTH_CLIENT_ID, config.DRUPAL_OAUTH_CLIENT_SECRET)

    async with httpx.AsyncClient(timeout=config.DRUPAL_OAUTH_TIMEOUT) as client:
        response = await client.post(
            introspection_url,
            data={"token": token},
            auth=auth,
        )
        response.raise_for_status()
        payload = response.json()

    if not payload.get("active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive OAuth2 token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


async def _fetch_userinfo(token: str) -> dict[str, Any] | None:
    userinfo_url = _userinfo_url()
    if not userinfo_url:
        return None

    async with httpx.AsyncClient(timeout=config.DRUPAL_OAUTH_TIMEOUT) as client:
        response = await client.get(
            userinfo_url,
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        return response.json()


async def _verify_drupal_oauth_token(token: str) -> dict[str, Any]:
    cached = _TOKEN_CACHE.get(token)
    if cached and cached[0] > time.time():
        return cached[1]

    try:
        payload = await _introspect_token(token)
        if payload is None:
            payload = await _fetch_userinfo(token)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        logger.warning(f"Drupal OAuth2 validation failed: {exc.response.status_code} {exc.response.text}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid OAuth2 token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except httpx.RequestError as exc:
        logger.error(f"Could not reach Drupal OAuth2 server: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OAuth2 validation service unavailable",
        ) from exc

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Drupal OAuth2 validation URL is not configured",
        )

    _TOKEN_CACHE[token] = (time.time() + _cache_ttl(payload), payload)
    return payload


async def verify_api_key(
    api_key: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> str | dict[str, Any]:
    """Verify API authentication.

    AUTH_MODE controls accepted credentials:
    - api_key: only X-API-Key
    - drupal_oauth2: only Authorization: Bearer <token>
    - api_key_or_drupal_oauth2: accepts either credential type
    """
    mode = _auth_mode()
    allow_api_key = mode in {"api_key", "api_key_or_drupal_oauth2", "api_key_or_oauth2"}
    allow_oauth = mode in {"drupal_oauth2", "oauth2", "api_key_or_drupal_oauth2", "api_key_or_oauth2"}

    if allow_api_key and api_key and api_key == config.API_KEY:
        return api_key

    if allow_oauth and bearer and bearer.scheme.lower() == "bearer":
        return await _verify_drupal_oauth_token(bearer.credentials)

    if mode not in {"api_key", "drupal_oauth2", "oauth2", "api_key_or_drupal_oauth2", "api_key_or_oauth2"}:
        logger.error(f"Unsupported AUTH_MODE={config.AUTH_MODE!r}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication mode is not supported",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
