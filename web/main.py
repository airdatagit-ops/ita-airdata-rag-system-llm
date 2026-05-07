"""FastAPI Web Application for Aviation RAG System."""

import base64
import hashlib
import hmac
import httpx
import json
import secrets
import time
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from loguru import logger

from app.config import settings

# Initialize FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Web interface for Aviation Regulations RAG System",
    root_path=settings.ROOT_PATH
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# HTTP client for API calls - longer timeout for RAG queries
http_client = httpx.AsyncClient(timeout=180.0)

# Chat history directory
CHAT_HISTORY_DIR = Path("chat_history")
CHAT_HISTORY_DIR.mkdir(exist_ok=True)

PRESENTATION_USERNAME = "airdata"
PRESENTATION_PASSWORD = "AirData-M7q9-V2x4-Kp31"


def _accepted_local_usernames() -> set[str]:
    return {settings.WEB_LOGIN_USERNAME, PRESENTATION_USERNAME}


def _accepted_local_passwords() -> set[str]:
    return {settings.WEB_LOGIN_PASSWORD, PRESENTATION_PASSWORD}


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def _jwt_secret() -> bytes:
    secret = settings.SESSION_SECRET_KEY or settings.API_KEY
    return secret.encode("utf-8")


def _sign_jwt(message: str) -> str:
    signature = hmac.new(_jwt_secret(), message.encode("ascii"), hashlib.sha256).digest()
    return _b64url_encode(signature)


def _create_local_jwt(username: str) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": username,
        "name": username,
        "iat": now,
        "exp": now + settings.WEB_LOGIN_TOKEN_TTL_SECONDS,
        "iss": "airdata-rag-web",
        "aud": "airdata-rag-web",
    }
    encoded_header = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    encoded_payload = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{encoded_header}.{encoded_payload}"
    return f"{signing_input}.{_sign_jwt(signing_input)}"


def _decode_local_jwt(token: str | None) -> dict | None:
    if not token:
        return None

    parts = token.split(".")
    if len(parts) != 3:
        return None

    signing_input = f"{parts[0]}.{parts[1]}"
    expected_signature = _sign_jwt(signing_input)
    if not hmac.compare_digest(parts[2], expected_signature):
        return None

    try:
        payload = json.loads(_b64url_decode(parts[1]))
    except (ValueError, json.JSONDecodeError):
        return None

    try:
        expires_at = int(payload.get("exp", 0))
    except (TypeError, ValueError):
        return None

    now = int(time.time())
    if payload.get("iss") != "airdata-rag-web" or payload.get("aud") != "airdata-rag-web":
        return None
    if expires_at <= now:
        return None
    if payload.get("sub") not in _accepted_local_usernames():
        return None
    return payload


def _local_jwt_user(request: Request) -> dict | None:
    if not _local_login_enabled():
        return None
    payload = _decode_local_jwt(request.cookies.get(settings.WEB_LOGIN_COOKIE_NAME))
    if not payload:
        return None
    return {"name": payload.get("name") or payload.get("sub") or settings.WEB_LOGIN_USERNAME}


def _oauth_enabled() -> bool:
    return settings.AUTH_MODE.lower() in {"drupal_oauth2", "oauth2", "api_key_or_drupal_oauth2", "api_key_or_oauth2"}


def _local_login_enabled() -> bool:
    # The production server may already have a stale web/.env with
    # WEB_LOGIN_ENABLED=false or AUTH_MODE=drupal_oauth2 from older deploys.
    # This branch intentionally closes the web UI with the local JWT login
    # while Drupal/OAuth stays on stand by.
    if settings.is_production:
        return True
    if _oauth_enabled():
        return False
    return settings.WEB_LOGIN_ENABLED


def _web_auth_enabled() -> bool:
    return _oauth_enabled() or _local_login_enabled()


def _authorize_url() -> str:
    if settings.DRUPAL_OAUTH_AUTHORIZE_URL:
        return settings.DRUPAL_OAUTH_AUTHORIZE_URL
    return f"{settings.DRUPAL_OAUTH_BASE_URL}/oauth/authorize"


def _token_url() -> str:
    if settings.DRUPAL_OAUTH_TOKEN_URL:
        return settings.DRUPAL_OAUTH_TOKEN_URL
    return f"{settings.DRUPAL_OAUTH_BASE_URL}/oauth/token"


def _userinfo_url() -> str:
    if settings.DRUPAL_OAUTH_USERINFO_URL:
        return settings.DRUPAL_OAUTH_USERINFO_URL
    if settings.DRUPAL_OAUTH_BASE_URL:
        return f"{settings.DRUPAL_OAUTH_BASE_URL}/oauth/userinfo"
    return ""


def _user_from_session(request: Request) -> dict | None:
    local_user = _local_jwt_user(request)
    if local_user:
        return local_user
    return request.session.get("user")


def _api_headers(request: Request) -> dict[str, str]:
    token = request.session.get("access_token") if _oauth_enabled() else None
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {"X-API-Key": settings.API_KEY}


def _template_context(request: Request, current_page: str, **extra):
    context = {
        "request": request,
        "current_page": current_page,
        "auth_enabled": _web_auth_enabled(),
        "current_user": _user_from_session(request),
    }
    context.update(extra)
    return context


def _strip_root_path(path: str) -> str:
    root_path = (settings.ROOT_PATH or "").rstrip("/")
    if root_path and path == root_path:
        return "/"
    if root_path and path.startswith(f"{root_path}/"):
        return path[len(root_path):] or "/"
    return path


def _prefixed_path(path: str) -> str:
    root_path = (settings.ROOT_PATH or "").rstrip("/")
    if not root_path:
        return path
    return f"{root_path}{path}"


@app.middleware("http")
async def require_web_authentication(request: Request, call_next):
    """Require a web login when OAuth2 or local presentation auth is enabled."""
    if not _web_auth_enabled():
        return await call_next(request)

    path = _strip_root_path(request.url.path)
    public_paths = ("/login", settings.DRUPAL_OAUTH_CALLBACK_PATH, "/logout", "/health")
    if path.startswith("/static/") or path in public_paths:
        return await call_next(request)

    if request.session.get("access_token") or _local_jwt_user(request):
        return await call_next(request)

    if path.startswith("/api/"):
        return JSONResponse({"detail": "Login required"}, status_code=401)

    return RedirectResponse(url=f"{_prefixed_path('/login')}?next={path}", status_code=302)


app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SESSION_SECRET_KEY,
    https_only=settings.SESSION_COOKIE_SECURE,
    same_site="lax",
)


def _generate_title_from_messages(messages: list, max_length: int = 50) -> str:
    """
    Generate a title from chat messages.
    Uses the first user message, truncated to max_length.
    """
    if not messages:
        return "Nova conversa"
    
    # Find first user message
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "").strip()
            if content:
                # Clean and truncate
                title = content.replace("\n", " ").strip()
                if len(title) > max_length:
                    title = title[:max_length].rsplit(" ", 1)[0] + "..."
                return title
    
    return "Conversa sem título"


@app.on_event("shutdown")
async def shutdown_event():
    """Close HTTP client on shutdown."""
    await http_client.aclose()


@app.get("/login")
async def login(request: Request, next: str = "/"):
    """Show local login or start Drupal OAuth2 authorization-code login."""
    if _local_login_enabled():
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "next_url": next if next.startswith("/") else "/",
                "username": PRESENTATION_USERNAME if settings.is_production else settings.WEB_LOGIN_USERNAME,
            },
        )

    if not _oauth_enabled():
        return RedirectResponse(url=next, status_code=302)

    if not settings.DRUPAL_OAUTH_CLIENT_ID or not settings.DRUPAL_OAUTH_BASE_URL:
        raise HTTPException(status_code=500, detail="Drupal OAuth2 is not configured")

    state = secrets.token_urlsafe(32)
    request.session["oauth_state"] = state
    request.session["next_url"] = next if next.startswith("/") else "/"

    params = {
        "response_type": "code",
        "client_id": settings.DRUPAL_OAUTH_CLIENT_ID,
        "redirect_uri": str(request.url_for("auth_callback")),
        "scope": settings.DRUPAL_OAUTH_SCOPES,
        "state": state,
    }
    return RedirectResponse(url=f"{_authorize_url()}?{urlencode(params)}", status_code=302)


@app.post("/login")
async def local_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next_url: str = Form("/"),
):
    """Authenticate with the temporary local web login."""
    if not _local_login_enabled():
        return RedirectResponse(url=str(request.url_for("login")), status_code=302)

    if not settings.WEB_LOGIN_PASSWORD:
        raise HTTPException(status_code=500, detail="WEB_LOGIN_PASSWORD is not configured")

    username_ok = any(secrets.compare_digest(username, expected) for expected in _accepted_local_usernames())
    password_ok = any(secrets.compare_digest(password, expected) for expected in _accepted_local_passwords())
    if not username_ok or not password_ok:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "next_url": next_url if next_url.startswith("/") else "/",
                "username": username,
                "error": "Usuário ou senha inválidos.",
            },
            status_code=401,
        )

    request.session.clear()
    redirect = RedirectResponse(url=next_url if next_url.startswith("/") else "/", status_code=302)
    redirect.set_cookie(
        key=settings.WEB_LOGIN_COOKIE_NAME,
        value=_create_local_jwt(username),
        max_age=settings.WEB_LOGIN_TOKEN_TTL_SECONDS,
        httponly=True,
        secure=settings.SESSION_COOKIE_SECURE,
        samesite="lax",
    )
    return redirect


@app.get(settings.DRUPAL_OAUTH_CALLBACK_PATH, name="auth_callback")
async def auth_callback(request: Request, code: str | None = None, state: str | None = None, error: str | None = None):
    """Complete Drupal OAuth2 authorization-code login."""
    if error:
        raise HTTPException(status_code=401, detail=f"Drupal OAuth2 error: {error}")

    if not code or not state or state != request.session.get("oauth_state"):
        raise HTTPException(status_code=401, detail="Invalid OAuth2 callback")

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": str(request.url_for("auth_callback")),
        "client_id": settings.DRUPAL_OAUTH_CLIENT_ID,
    }
    if settings.DRUPAL_OAUTH_CLIENT_SECRET:
        data["client_secret"] = settings.DRUPAL_OAUTH_CLIENT_SECRET

    response = await http_client.post(_token_url(), data=data)
    if response.status_code != 200:
        logger.error(f"Drupal token exchange failed: {response.status_code} - {response.text}")
        raise HTTPException(status_code=401, detail="Could not authenticate with Drupal")

    token_data = response.json()
    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="Drupal did not return an access token")

    user = {"name": "Usuário autenticado"}
    userinfo_url = _userinfo_url()
    if userinfo_url:
        user_response = await http_client.get(
            userinfo_url,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if user_response.status_code == 200:
            user = user_response.json()
        else:
            logger.warning(f"Could not fetch Drupal userinfo: {user_response.status_code} - {user_response.text}")

    next_url = request.session.get("next_url", "/")
    request.session.clear()
    request.session["access_token"] = access_token
    request.session["refresh_token"] = token_data.get("refresh_token")
    request.session["user"] = user
    return RedirectResponse(url=next_url, status_code=302)


@app.get("/logout")
async def logout(request: Request):
    """Clear the local web session."""
    request.session.clear()
    target = str(request.url_for("login")) if _web_auth_enabled() else str(request.url_for("home"))
    redirect = RedirectResponse(url=target, status_code=302)
    redirect.delete_cookie(settings.WEB_LOGIN_COOKIE_NAME)
    return redirect


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse(
        "index.html",
        _template_context(request, "home")
    )


@app.get("/pesquisa", response_class=HTMLResponse)
async def search_page(request: Request):
    """Search page."""
    return templates.TemplateResponse(
        "search.html",
        _template_context(request, "pesquisa")
    )


@app.post("/pesquisa", response_class=HTMLResponse)
async def search_post(
    request: Request,
    query: str = Form(...),
    date: Optional[str] = Form(None),
    limit: int = Form(5),
    score_threshold: Optional[float] = Form(None)
):
    """Handle search form submission - vector-only search (no LLM)."""
    try:
        # Prepare request payload for vector-only search
        payload = {
            "query": query,
            "limit": limit
        }
        
        if date:
            payload["date"] = date
        
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        
        # Call vector-only search API (faster, no LLM)
        headers = _api_headers(request)
        response = await http_client.post(
            f"{settings.API_BASE_URL}/api/vector-search",
            json=payload,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            return templates.TemplateResponse(
                "search.html",
                _template_context(
                    request,
                    "pesquisa",
                    query=query,
                    result=result,
                    search_params={
                        "date": date,
                        "limit": limit,
                        "score_threshold": score_threshold
                    },
                )
            )
        else:
            error_msg = f"API Error: {response.status_code}"
            logger.error(f"{error_msg} - {response.text}")
            return templates.TemplateResponse(
                "search.html",
                _template_context(request, "pesquisa", error=error_msg)
            )
            
    except Exception as e:
        logger.error(f"Error processing search: {e}")
        return templates.TemplateResponse(
            "search.html",
            _template_context(request, "pesquisa", error=str(e))
        )


@app.get("/estatisticas", response_class=HTMLResponse)
async def stats_page(request: Request):
    """Statistics page."""
    try:
        # Call API
        headers = _api_headers(request)
        response = await http_client.get(
            f"{settings.API_BASE_URL}/stats",
            headers=headers
        )
        
        if response.status_code == 200:
            stats = response.json()
            return templates.TemplateResponse(
                "stats.html",
                _template_context(
                    request,
                    "estatisticas",
                    stats=stats,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
        else:
            error_msg = f"API Error: {response.status_code}"
            logger.error(f"{error_msg} - {response.text}")
            return templates.TemplateResponse(
                "stats.html",
                _template_context(request, "estatisticas", error=error_msg)
            )
            
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return templates.TemplateResponse(
            "stats.html",
            _template_context(request, "estatisticas", error=str(e))
        )


@app.get("/sobre", response_class=HTMLResponse)
async def about_page(request: Request):
    """About page."""
    return templates.TemplateResponse(
        "about.html",
        _template_context(request, "sobre")
    )


# ========================================
# Chat Page Routes
# ========================================

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Chat page with model selection."""
    try:
        # Get available models from API
        headers = _api_headers(request)
        response = await http_client.get(
            f"{settings.API_BASE_URL}/api/models",
            headers=headers
        )
        
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get("models", [])
            current_model = models_data.get("current_model", "")
        else:
            models = []
            current_model = ""
            logger.error(f"Failed to fetch models: {response.status_code}")
        
        return templates.TemplateResponse(
            "chat.html",
            _template_context(
                request,
                "chat",
                models=models,
                current_model=current_model,
            )
        )
        
    except Exception as e:
        logger.error(f"Error loading chat page: {e}")
        return templates.TemplateResponse(
            "chat.html",
            _template_context(
                request,
                "chat",
                models=[],
                current_model="",
                error=str(e),
            )
        )


@app.post("/api/chat/send")
async def send_chat_message(request: Request):
    """Proxy chat request to API with streaming support."""
    try:
        body = await request.json()
        message = body.get("message", "")
        session_id = body.get("session_id")
        use_rag = body.get("use_rag", False)
        debug = body.get("debug", False)
        model_name = body.get("model_name")
        
        headers = _api_headers(request)
        
        # Change model if specified
        if model_name:
            await http_client.post(
                f"{settings.API_BASE_URL}/api/models/change",
                headers=headers,
                json={"model_name": model_name}
            )
        
        # Send chat request
        payload = {
            "message": message,
            "use_rag": use_rag,
            "debug": debug,
            "context_window": 10
        }
        
        if session_id:
            payload["session_id"] = session_id
        
        response = await http_client.post(
            f"{settings.API_BASE_URL}/api/chat",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Save to chat history
            _save_chat_history(
                session_id=result.get("session_id"),
                user_message=message,
                assistant_message=result.get("message"),
                model_used=result.get("model_used"),
                use_rag=use_rag,
                sources=result.get("sources"),
                processing_time_ms=result.get("processing_time_ms")
            )
            
            return result
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except HTTPException:
        raise
    except httpx.TimeoutException as e:
        logger.error(f"Timeout calling API: {e}")
        raise HTTPException(status_code=504, detail="API request timed out. RAG queries may take longer. Please try again.")
    except httpx.RequestError as e:
        logger.error(f"Request error calling API: {e}")
        raise HTTPException(status_code=502, detail=f"Could not connect to API: {e}")
    except Exception as e:
        logger.error(f"Error sending chat message: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e) or type(e).__name__)


@app.post("/api/chat/stream")
async def stream_chat_message(request: Request):
    """Streaming chat endpoint - proxies SSE from API."""
    try:
        body = await request.json()
        message = body.get("message", "")
        session_id = body.get("session_id")
        use_rag = body.get("use_rag", False)
        debug = body.get("debug", False)
        model_name = body.get("model_name")
        
        headers = _api_headers(request)
        
        # Change model if specified
        if model_name:
            await http_client.post(
                f"{settings.API_BASE_URL}/api/models/change",
                headers=headers,
                json={"model_name": model_name}
            )
        
        # Prepare payload
        payload = {
            "message": message,
            "use_rag": use_rag,
            "debug": debug,
            "context_window": 10
        }
        
        if session_id:
            payload["session_id"] = session_id
        
        async def stream_generator():
            """Stream response from API using raw bytes for immediate forwarding."""
            full_response = []
            final_data = None
            sources = None
            buffer = ""
            
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "POST",
                        f"{settings.API_BASE_URL}/api/chat/stream",
                        headers=headers,
                        json=payload
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            # Forward raw bytes immediately for real-time streaming
                            yield chunk

                            # Parse SSE events from buffer for history saving
                            buffer += chunk.decode("utf-8", errors="replace")
                            while "\n\n" in buffer:
                                event_str, buffer = buffer.split("\n\n", 1)
                                for line in event_str.split("\n"):
                                    if line.startswith("data: "):
                                        try:
                                            data = json.loads(line[6:])
                                            if data.get("type") == "token":
                                                full_response.append(data.get("content", ""))
                                            elif data.get("type") == "sources":
                                                sources = data.get("sources")
                                            elif data.get("type") == "done":
                                                final_data = data
                                        except json.JSONDecodeError:
                                            pass
                
                # Save to history after streaming completes
                if final_data and full_response:
                    _save_chat_history(
                        session_id=final_data.get("session_id", session_id),
                        user_message=message,
                        assistant_message="".join(full_response),
                        model_used=final_data.get("model_used", model_name),
                        use_rag=use_rag,
                        sources=sources,
                        processing_time_ms=final_data.get("processing_time_ms", 0)
                    )
                    
            except Exception as e:
                logger.error(f"Error in stream proxy: {e}")
                yield f'data: {{"type": "error", "error": "{str(e)}"}}\n\n'.encode("utf-8")
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"Error setting up stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class RatingRequest(BaseModel):
    session_id: str
    message_id: str
    category: str
    rating: int  # 0-5


@app.post("/api/chat/rate")
async def rate_message(request: RatingRequest):
    """Save rating for a specific message in the chat history."""
    try:
        # Validate rating value
        if request.rating < 0 or request.rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 0 and 5")
        
        # Validate category
        valid_categories = ["factual_accuracy", "completeness", "clarity", "citation_quality", "relevance"]
        if request.category not in valid_categories:
            raise HTTPException(status_code=400, detail=f"Invalid category. Must be one of: {valid_categories}")
        
        # Load the chat history file
        history_file = CHAT_HISTORY_DIR / f"{request.session_id}.json"
        if not history_file.exists():
            raise HTTPException(status_code=404, detail="Session not found")
        
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
        
        # Find the message by ID and update its rating
        message_found = False
        for msg in history.get("messages", []):
            if msg.get("message_id") == request.message_id:
                if "ratings" not in msg:
                    msg["ratings"] = {}
                msg["ratings"][request.category] = request.rating
                message_found = True
                break
        
        if not message_found:
            # If message_id not found, try to find the last assistant message
            # This handles cases where message_id wasn't saved initially
            for msg in reversed(history.get("messages", [])):
                if msg.get("role") == "assistant":
                    if "ratings" not in msg:
                        msg["ratings"] = {}
                    msg["ratings"][request.category] = request.rating
                    msg["message_id"] = request.message_id  # Save the ID for future
                    message_found = True
                    break
        
        if not message_found:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Save the updated history
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        return {"status": "success", "message": "Rating saved"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving rating: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/change")
async def change_model_proxy(request: Request):
    """Proxy model change request to API."""
    try:
        body = await request.json()
        headers = _api_headers(request)
        
        response = await http_client.post(
            f"{settings.API_BASE_URL}/api/models/change",
            headers=headers,
            json=body
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def get_models_proxy(request: Request):
    """Proxy models list request to API."""
    try:
        headers = _api_headers(request)
        
        response = await http_client.get(
            f"{settings.API_BASE_URL}/api/models",
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/history")
async def get_chat_history():
    """Get all saved chat history files."""
    try:
        history_files = []
        
        for file_path in CHAT_HISTORY_DIR.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    history_files.append({
                        "filename": file_path.name,
                        "session_id": data.get("session_id"),
                        "title": data.get("title", _generate_title_from_messages(data.get("messages", []))),
                        "model": data.get("model"),
                        "messages_count": len(data.get("messages", [])),
                        "created_at": data.get("created_at"),
                        "last_updated": data.get("last_updated")
                    })
            except Exception as e:
                logger.warning(f"Error reading history file {file_path}: {e}")
        
        # Sort by last_updated descending
        history_files.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
        
        return {"history": history_files}
        
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/history/{session_id}")
async def get_session_history(session_id: str):
    """Get chat history for a specific session."""
    try:
        file_path = CHAT_HISTORY_DIR / f"{session_id}.json"
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Session history not found")
        
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching session history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _save_chat_history(
    session_id: str,
    user_message: str,
    assistant_message: str,
    model_used: str,
    use_rag: bool,
    sources: list = None,
    processing_time_ms: int = 0,
    message_id: str = None
):
    """Save chat interaction to history file."""
    import secrets
    try:
        file_path = CHAT_HISTORY_DIR / f"{session_id}.json"
        
        # Load existing or create new
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            # Generate title from first user message
            title = user_message.replace("\n", " ").strip()
            if len(title) > 50:
                title = title[:50].rsplit(" ", 1)[0] + "..."
            
            history = {
                "session_id": session_id,
                "title": title,
                "model": model_used,
                "created_at": datetime.now().isoformat(),
                "messages": []
            }
        
        # Update model if changed
        history["model"] = model_used
        history["last_updated"] = datetime.now().isoformat()
        
        # Add messages
        timestamp = datetime.now().isoformat()
        
        # Generate message ID if not provided
        if not message_id:
            message_id = f"msg_{int(datetime.now().timestamp() * 1000)}_{secrets.token_hex(4)}"
        
        history["messages"].append({
            "role": "user",
            "content": user_message,
            "timestamp": timestamp
        })
        
        history["messages"].append({
            "role": "assistant",
            "content": assistant_message,
            "timestamp": timestamp,
            "message_id": message_id,
            "model": model_used,
            "use_rag": use_rag,
            "sources": sources,
            "processing_time_ms": processing_time_ms,
            "ratings": {}
        })
        
        # Save
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
        logger.debug(f"Chat history saved: {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    # proxy_headers + forwarded_allow_ips are required so that, when running
    # behind nginx (production), uvicorn honors X-Forwarded-Proto/Host. Without
    # them, request.url_for(...) builds the OAuth callback as http://127.0.0.1
    # instead of https://chatbot.airdata.ita.br, which Drupal then rejects with
    # redirect_uri_mismatch. nginx connects from localhost, so we trust 127.0.0.1.
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1",
    )
