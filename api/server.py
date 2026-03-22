"""FastAPI server for Aviation RAG System."""

import json
import time
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from loguru import logger

from config import config
from api.auth import verify_api_key
from api.schemas import (
    SearchRequest,
    SearchResponse,
    VectorSearchRequest,
    VectorSearchResponse,
    StatsResponse,
    ChatRequest,
    ChatResponse,
    SessionInfo,
    SessionStatsResponse,
    OllamaModel,
    ModelsListResponse,
    ChangeModelRequest,
    ChangeModelResponse
)
from search.rag import RAGPipeline
from search.vector_search import VectorSearch
from database.qdrant_manager import QdrantManager
from api.session_manager import session_manager
from models.llm import LlamaModel

# Initialize
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize services
rag = RAGPipeline()
db = QdrantManager()
vector_search = VectorSearch()  # For vector-only search (no LLM)
llm = LlamaModel()  # Dedicated LLM instance for chatbot


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": config.API_TITLE,
        "version": config.API_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check."""
    return_json = {"status": "healthy"}
    logger.info("Health check accessed.")
    logger.info(f"Json returned: {return_json}")
    return return_json


@app.post("/api/search-regulations", response_model=SearchResponse)
@limiter.limit(f"{config.RATE_LIMIT}/minute")
async def search_regulations(
    request: Request,
    search_request: SearchRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Search for regulations using RAG.

    Requires X-API-Key header for authentication.
    """
    try:
        result = rag.query(
            question=search_request.query,
            date=search_request.date,
            limit=search_request.limit,
            return_sources=True
        )

        return SearchResponse(**result)

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise


@app.post("/api/vector-search", response_model=VectorSearchResponse)
@limiter.limit(f"{config.RATE_LIMIT}/minute")
async def vector_search_endpoint(
    request: Request,
    search_request: VectorSearchRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Vector-only search for regulations (no LLM generation).
    
    Fast semantic search that returns similar documents without
    generating an LLM response. Useful for quick lookups.

    Requires X-API-Key header for authentication.
    """
    import time
    start_time = time.time()
    
    try:
        # Perform vector search
        if search_request.date:
            results = vector_search.search_temporal(
                query=search_request.query,
                date=search_request.date,
                limit=search_request.limit
            )
        else:
            results = vector_search.search(
                query=search_request.query,
                limit=search_request.limit,
                score_threshold=search_request.score_threshold
            )
        
        search_time_ms = int((time.time() - start_time) * 1000)
        
        return VectorSearchResponse(
            query=search_request.query,
            sources=results,
            total_results=len(results),
            search_time_ms=search_time_ms
        )

    except Exception as e:
        logger.error(f"Error processing vector search: {e}")
        raise


@app.get("/stats", response_model=StatsResponse)
async def get_stats(api_key: str = Depends(verify_api_key)):
    """Get system statistics including document counts."""
    try:
        info = db.get_collection_info()
        logger.info(f"Stats retrieved: {info}")
        
        try:
            from pipeline.document_store import DocumentStore
            info["documents"] = DocumentStore().stats()
        except Exception as e:
            logger.warning(f"Could not get document counts: {e}")
            info["documents"] = None
        
        return StatsResponse(**info)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise


# ========================================
# Chatbot Endpoints
# ========================================

@app.post("/api/chat/stream")
@limiter.limit(f"{config.RATE_LIMIT}/minute")
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Streaming chat endpoint with conversation history and optional RAG.
    
    Returns Server-Sent Events (SSE) with tokens as they are generated.
    This eliminates timeout issues for long-running queries.

    Requires X-API-Key header for authentication.
    """
    try:
        # Get or create session
        if chat_request.session_id:
            session = session_manager.get_session(chat_request.session_id)
            if session is None:
                logger.warning(f"Session {chat_request.session_id} not found, creating new")
                session_id = session_manager.create_session(chat_request.session_id)
            else:
                session_id = chat_request.session_id
        else:
            session_id = session_manager.create_session()

        # Add user message to session
        session_manager.add_message(session_id, "user", chat_request.message)

        # Get conversation context
        context_messages = session_manager.get_context_window(
            session_id,
            last_n=chat_request.context_window
        )

        def generate_stream():
            """Sync generator that yields SSE events.

            Must be a plain (non-async) generator so that Starlette runs it
            via iterate_in_threadpool, keeping the event loop free to flush
            each chunk to the client immediately.
            """
            start_time = time.time()
            full_response = []
            sources = None
            
            try:
                # Send session info first
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
                
                if chat_request.use_rag:
                    # Perform RAG search
                    if chat_request.rag_date:
                        search_results = rag.search.search_temporal(
                            chat_request.message,
                            chat_request.rag_date,
                            limit=5
                        )
                    else:
                        search_results = rag.search.search(
                            chat_request.message,
                            limit=5
                        )
                    
                    sources = search_results
                    
                    # Send sources info (including text for modal display)
                    sources_data = [
                        {
                            "regulation_id": s.get("regulation_id"),
                            "score": s.get("score"),
                            "text": s.get("text", "")[:2000]
                        }
                        for s in search_results[:5]
                    ] if search_results else []
                    yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data})}\n\n"
                    
                    if search_results:
                        # Build RAG prompt
                        context_str = rag.llm._build_context_string(search_results)
                        system_prompt = """Você é um assistente especializado em regulamentação de aviação civil brasileira.
Use as normas fornecidas para responder, mas também mantenha o contexto da conversa anterior.
Cite sempre as fontes quando mencionar informações das normas."""

                        prompt = "=== HISTÓRICO DA CONVERSA ===\n"
                        for msg in context_messages[:-1]:
                            prompt += f"{msg['role'].upper()}: {msg['content']}\n"
                        prompt += f"\n=== NORMAS REGULATÓRIAS ===\n{context_str}\n"
                        prompt += f"\n=== PERGUNTA ATUAL ===\n{chat_request.message}\n"
                        prompt += "\n=== RESPOSTA ===\nBaseado nas normas fornecidas e no contexto da conversa:\n"
                        
                        # Stream from LLM
                        for chunk in llm.generate(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            temperature=chat_request.temperature,
                            max_tokens=chat_request.max_tokens,
                            stream=True
                        ):
                            full_response.append(chunk)
                            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                    else:
                        for chunk in _stream_without_rag(chat_request, context_messages):
                            full_response.append(chunk)
                            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                else:
                    for chunk in _stream_without_rag(chat_request, context_messages):
                        full_response.append(chunk)
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                
                # Save assistant response to session
                complete_response = "".join(full_response)
                session_manager.add_message(session_id, "assistant", complete_response)
                
                # Get final stats
                session = session_manager.get_session(session_id)
                message_count = session.get_message_count() if session else 0
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                # Send completion event
                yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'message_count': message_count, 'processing_time_ms': processing_time_ms, 'model_used': llm.model_name})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in stream generation: {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    except ValueError as e:
        logger.error(f"Validation error in chat stream: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing chat stream request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def _stream_without_rag(request: ChatRequest, context_messages: list):
    """Generate streaming response without RAG."""
    system_prompt = """Você é um assistente especializado em aviação civil brasileira.
Responda de forma clara e precisa. Se a pergunta for sobre regulamentações específicas,
sugira ao usuário usar a funcionalidade de busca com RAG para obter informações precisas."""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(context_messages)
    
    for chunk in llm.chat(
        messages=messages,
        temperature=request.temperature or 0.7,
        max_tokens=request.max_tokens,
        stream=True
    ):
        yield chunk


@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit(f"{config.RATE_LIMIT}/minute")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Chat endpoint with conversation history and optional RAG.

    Features:
    - Maintains conversation history per session
    - Optional RAG integration for fact-based responses
    - Configurable context window
    - Session-based continuity

    Requires X-API-Key header for authentication.
    """
    start_time = time.time()

    try:
        # Get or create session
        if chat_request.session_id:
            session = session_manager.get_session(chat_request.session_id)
            if session is None:
                logger.warning(f"Session {chat_request.session_id} not found, creating new")
                session_id = session_manager.create_session(chat_request.session_id)
            else:
                session_id = chat_request.session_id
        else:
            session_id = session_manager.create_session()

        # Add user message to session
        session_manager.add_message(session_id, "user", chat_request.message)

        # Get conversation context
        context_messages = session_manager.get_context_window(
            session_id,
            last_n=chat_request.context_window
        )

        # Generate response
        if chat_request.use_rag:
            # Use RAG for fact-based response
            assistant_message = await _chat_with_rag(
                request=chat_request,
                session_id=session_id,
                context_messages=context_messages
            )
        else:
            # Use LLM only for general conversation
            assistant_message = await _chat_without_rag(
                request=chat_request,
                context_messages=context_messages
            )

        # Add assistant response to session
        session_manager.add_message(session_id, "assistant", assistant_message["content"])

        # Get session info
        session = session_manager.get_session(session_id)
        message_count = session.get_message_count() if session else 0

        processing_time_ms = int((time.time() - start_time) * 1000)

        return ChatResponse(
            message=assistant_message["content"],
            session_id=session_id,
            message_count=message_count,
            sources=assistant_message.get("sources"),
            processing_time_ms=processing_time_ms,
            model_used=llm.model_name
        )

    except ValueError as e:
        logger.error(f"Validation error in chat: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def _chat_with_rag(
    request: ChatRequest,
    session_id: str,
    context_messages: list
) -> dict:
    """
    Handle chat with RAG integration.

    Args:
        request: Chat request
        session_id: Session ID
        context_messages: Previous messages for context

    Returns:
        Dictionary with content and sources
    """
    # Perform RAG search
    if request.rag_date:
        search_results = rag.search.search_temporal(
            request.message,
            request.rag_date,
            limit=5
        )
    else:
        search_results = rag.search.search(
            request.message,
            limit=5
        )

    if not search_results:
        # No relevant documents found, use regular chat
        return await _chat_without_rag(request, context_messages)

    # Build context string from search results
    context_str = rag.llm._build_context_string(search_results)

    # Build prompt with conversation history
    system_prompt = """Você é um assistente especializado em regulamentação de aviação civil brasileira.
Use as normas fornecidas para responder, mas também mantenha o contexto da conversa anterior.
Cite sempre as fontes quando mencionar informações das normas."""

    # Combine conversation history with RAG context
    prompt = f"""=== HISTÓRICO DA CONVERSA ===
"""
    
    # Add previous messages (excluding the last user message)
    for msg in context_messages[:-1]:
        prompt += f"{msg['role'].upper()}: {msg['content']}\n"

    prompt += f"""
=== NORMAS REGULATÓRIAS ===
{context_str}

=== PERGUNTA ATUAL ===
{request.message}

=== RESPOSTA ===
Baseado nas normas fornecidas e no contexto da conversa:
"""

    # Generate response
    response = llm.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

    return {
        "content": response,
        "sources": search_results
    }


async def _chat_without_rag(
    request: ChatRequest,
    context_messages: list
) -> dict:
    """
    Handle chat without RAG (general conversation).

    Args:
        request: Chat request
        context_messages: Previous messages for context

    Returns:
        Dictionary with content
    """
    # Add system prompt for aviation context
    system_prompt = """Você é um assistente especializado em aviação civil brasileira.
Responda de forma clara e precisa. Se a pergunta for sobre regulamentações específicas,
sugira ao usuário usar a funcionalidade de busca com RAG para obter informações precisas."""

    # Prepare messages for chat
    messages = []
    
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # Add conversation history
    messages.extend(context_messages)

    # Generate response
    response = llm.chat(
        messages=messages,
        temperature=request.temperature or 0.7,
        max_tokens=request.max_tokens
    )

    return {
        "content": response,
        "sources": None
    }


@app.get("/api/chat/session/{session_id}", response_model=SessionInfo)
async def get_session_info(
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get information about a chat session."""
    session = session_manager.get_session(session_id)
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    return SessionInfo(**session.to_dict())


@app.delete("/api/chat/session/{session_id}")
async def delete_session(
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Delete a chat session."""
    success = session_manager.delete_session(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session deleted successfully", "session_id": session_id}


@app.post("/api/chat/session/{session_id}/clear")
async def clear_session(
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Clear chat history in a session without deleting it."""
    success = session_manager.clear_session(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session history cleared", "session_id": session_id}


@app.get("/api/chat/sessions/stats", response_model=SessionStatsResponse)
async def get_session_stats(api_key: str = Depends(verify_api_key)):
    """Get statistics about active chat sessions."""
    stats = session_manager.get_stats()
    return SessionStatsResponse(**stats)


# ========================================
# Model Management Endpoints
# ========================================

@app.get("/api/models", response_model=ModelsListResponse)
async def list_models(api_key: str = Depends(verify_api_key)):
    """
    List all available Ollama models.
    
    Returns the list of models installed on the Ollama server
    and the currently selected model.
    """
    try:
        # Get models from Ollama
        models_response = llm.client.list()
        
        # Parse response
        if isinstance(models_response, dict) and "models" in models_response:
            models_list = models_response["models"]
        elif isinstance(models_response, (list, tuple)):
            models_list = models_response
        else:
            models_list = getattr(models_response, "models", [])
        
        # Convert to schema
        models = []
        for m in models_list:
            if isinstance(m, dict):
                models.append(OllamaModel(
                    name=m.get("name", m.get("model", "unknown")),
                    size=_format_size(m.get("size", 0)),
                    modified_at=m.get("modified_at", ""),
                    digest=m.get("digest", "")[:12] if m.get("digest") else None
                ))
            else:
                models.append(OllamaModel(
                    name=getattr(m, "name", getattr(m, "model", "unknown")),
                    size=_format_size(getattr(m, "size", 0)),
                    modified_at=str(getattr(m, "modified_at", "")),
                    digest=str(getattr(m, "digest", ""))[:12] if getattr(m, "digest", None) else None
                ))
        logger.info(f"Available models: {[model.name for model in models]}")
        logger.info(f"Current model: {llm.model_name}")
        sorted_models = sorted(models, key=lambda x: x.name.lower())
        return ModelsListResponse(
            models=sorted_models,
            current_model=llm.model_name
        )
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/change", response_model=ChangeModelResponse)
async def change_model(
    request: Request,
    change_request: ChangeModelRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Change the current LLM model.
    
    The new model must be available on the Ollama server.
    """
    global llm
    
    try:
        previous_model = llm.model_name
        new_model = change_request.model_name
        
        # Verify the model exists
        models_response = llm.client.list()
        
        if isinstance(models_response, dict) and "models" in models_response:
            models_list = models_response["models"]
        elif isinstance(models_response, (list, tuple)):
            models_list = models_response
        else:
            models_list = getattr(models_response, "models", [])
        
        available_models = []
        for m in models_list:
            if isinstance(m, dict):
                available_models.append(m.get("name", m.get("model", "")))
            else:
                available_models.append(getattr(m, "name", getattr(m, "model", "")))
        
        if new_model not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{new_model}' not found. Available models: {available_models}"
            )
        
        # Change the model
        llm.model_name = new_model
        
        logger.info(f"Model changed from '{previous_model}' to '{new_model}'")
        
        return ChangeModelResponse(
            success=True,
            previous_model=previous_model,
            current_model=new_model,
            message=f"Successfully changed model from '{previous_model}' to '{new_model}'"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    if not size_bytes:
        return "N/A"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.RELOAD
    )
