"""FastAPI Web Application for Aviation RAG System."""

import httpx
import json
from pathlib import Path
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
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


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "current_page": "home"}
    )


@app.get("/pesquisa", response_class=HTMLResponse)
async def search_page(request: Request):
    """Search page."""
    return templates.TemplateResponse(
        "search.html",
        {"request": request, "current_page": "pesquisa"}
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
        headers = {"X-API-Key": settings.API_KEY}
        response = await http_client.post(
            f"{settings.API_BASE_URL}/api/vector-search",
            json=payload,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            return templates.TemplateResponse(
                "search.html",
                {
                    "request": request,
                    "current_page": "pesquisa",
                    "query": query,
                    "result": result,
                    "search_params": {
                        "date": date,
                        "limit": limit,
                        "score_threshold": score_threshold
                    }
                }
            )
        else:
            error_msg = f"API Error: {response.status_code}"
            logger.error(f"{error_msg} - {response.text}")
            return templates.TemplateResponse(
                "search.html",
                {
                    "request": request,
                    "current_page": "pesquisa",
                    "error": error_msg
                }
            )
            
    except Exception as e:
        logger.error(f"Error processing search: {e}")
        return templates.TemplateResponse(
            "search.html",
            {
                "request": request,
                "current_page": "pesquisa",
                "error": str(e)
            }
        )


@app.get("/estatisticas", response_class=HTMLResponse)
async def stats_page(request: Request):
    """Statistics page."""
    try:
        # Call API
        headers = {"X-API-Key": settings.API_KEY}
        response = await http_client.get(
            f"{settings.API_BASE_URL}/stats",
            headers=headers
        )
        
        if response.status_code == 200:
            stats = response.json()
            return templates.TemplateResponse(
                "stats.html",
                {
                    "request": request,
                    "current_page": "estatisticas",
                    "stats": stats,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            )
        else:
            error_msg = f"API Error: {response.status_code}"
            logger.error(f"{error_msg} - {response.text}")
            return templates.TemplateResponse(
                "stats.html",
                {
                    "request": request,
                    "current_page": "estatisticas",
                    "error": error_msg
                }
            )
            
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return templates.TemplateResponse(
            "stats.html",
            {
                "request": request,
                "current_page": "estatisticas",
                "error": str(e)
            }
        )


@app.get("/sobre", response_class=HTMLResponse)
async def about_page(request: Request):
    """About page."""
    return templates.TemplateResponse(
        "about.html",
        {"request": request, "current_page": "sobre"}
    )


# ========================================
# Chat Page Routes
# ========================================

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Chat page with model selection."""
    try:
        # Get available models from API
        headers = {"X-API-Key": settings.API_KEY}
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
            {
                "request": request,
                "current_page": "chat",
                "models": models,
                "current_model": current_model
            }
        )
        
    except Exception as e:
        logger.error(f"Error loading chat page: {e}")
        return templates.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "current_page": "chat",
                "models": [],
                "current_model": "",
                "error": str(e)
            }
        )


@app.post("/api/chat/send")
async def send_chat_message(request: Request):
    """Proxy chat request to API with streaming support."""
    try:
        body = await request.json()
        message = body.get("message", "")
        session_id = body.get("session_id")
        use_rag = body.get("use_rag", False)
        model_name = body.get("model_name")
        
        headers = {"X-API-Key": settings.API_KEY}
        
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
        model_name = body.get("model_name")
        
        headers = {"X-API-Key": settings.API_KEY}
        
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
        headers = {"X-API-Key": settings.API_KEY}
        
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
async def get_models_proxy():
    """Proxy models list request to API."""
    try:
        headers = {"X-API-Key": settings.API_KEY}
        
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
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD
    )
