"""FastAPI server for Aviation RAG System."""

import asyncio
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
    DocumentStats,
    DocumentTypeCount,
    RelationStats,
    ModelInfo,
    ChatRequest,
    ChatResponse,
    SessionInfo,
    SessionStatsResponse,
    OllamaModel,
    ModelsListResponse,
    ChangeModelRequest,
    ChangeModelResponse,
)
from search.rag import RAGPipeline
from search.vector_search import VectorSearch
from search.cache import InMemoryCache
from search.exceptions import SearchBackendError
from search.prompts import SYSTEM_PROMPT
from database.qdrant_manager import QdrantManager
from api.session_manager import session_manager
from models.llm import LlamaModel
from models.embeddings import EmbeddingModel, SparseEncoder

# ========================================
# Application
# ========================================

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ========================================
# Shared service instances (DI)
# ========================================
# Each component is created once and injected into dependents.
# ``rag.llm`` and the module-level ``llm`` reference the same object,
# so ``change_model`` affects all code paths automatically.

db = QdrantManager()
dense_model = EmbeddingModel() if config.SEARCH_DENSE_ENABLED else None
sparse_model = SparseEncoder() if config.SEARCH_SPARSE_ENABLED else None
llm = LlamaModel()
embedding_cache = InMemoryCache(maxsize=1024, default_ttl=3600)
response_cache = InMemoryCache(maxsize=256, default_ttl=1800)

vector_search = VectorSearch(
    dense_model=dense_model, sparse_model=sparse_model, db=db,
    embedding_cache=embedding_cache,
)
rag = RAGPipeline(search=vector_search, llm=llm, response_cache=response_cache)


# ========================================
# General Endpoints
# ========================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": config.API_TITLE,
        "version": config.API_VERSION,
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check."""
    return_json = {"status": "healthy"}
    logger.info("Health check accessed.")
    logger.info(f"Json returned: {return_json}")
    return return_json


# ========================================
# Search Endpoints
# ========================================

@app.post("/api/search-regulations", response_model=SearchResponse)
@limiter.limit(f"{config.RATE_LIMIT}/minute")
async def search_regulations(
    request: Request,
    search_request: SearchRequest,
    api_key: str = Depends(verify_api_key),
):
    """Search for regulations using RAG.

    Requires X-API-Key header for authentication.
    """
    try:
        result = await asyncio.to_thread(
            rag.query,
            question=search_request.query,
            date=search_request.date,
            limit=search_request.limit,
            return_sources=True,
        )
        return SearchResponse(**result)
    except SearchBackendError:
        raise HTTPException(status_code=503, detail="Search service unavailable")
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise


@app.post("/api/vector-search", response_model=VectorSearchResponse)
@limiter.limit(f"{config.RATE_LIMIT}/minute")
async def vector_search_endpoint(
    request: Request,
    search_request: VectorSearchRequest,
    api_key: str = Depends(verify_api_key),
):
    """Vector-only search for regulations (no LLM generation).

    Fast semantic search that returns similar documents without
    generating an LLM response. Useful for quick lookups.

    Requires X-API-Key header for authentication.
    """
    start_time = time.time()

    try:
        if search_request.date:
            results = await asyncio.to_thread(
                vector_search.search_temporal,
                query=search_request.query,
                date=search_request.date,
                limit=search_request.limit,
            )
        else:
            results = await asyncio.to_thread(
                vector_search.search,
                query=search_request.query,
                limit=search_request.limit,
                score_threshold=search_request.score_threshold,
            )

        search_time_ms = int((time.time() - start_time) * 1000)

        return VectorSearchResponse(
            query=search_request.query,
            sources=results,
            total_results=len(results),
            search_time_ms=search_time_ms,
        )
    except SearchBackendError:
        raise HTTPException(status_code=503, detail="Search service unavailable")
    except Exception as e:
        logger.error(f"Error processing vector search: {e}")
        raise


# ========================================
# Stats Endpoint
# ========================================

@app.get("/stats", response_model=StatsResponse)
async def get_stats(api_key: str = Depends(verify_api_key)):
    """Get system statistics including document counts."""
    try:
        info = db.get_collection_info()
        logger.info(f"Stats retrieved: {info}")

        doc_stats = None
        try:
            from pipeline.document_store import DocumentStore
            raw = DocumentStore().stats()

            _TYPE_LABELS = {
                "ICA": "ICA", "DCA": "DCA", "MCA": "MCA", "FCA": "FCA",
                "NSCA": "NSCA", "PCA": "PCA", "RCA": "RCA", "TCA": "TCA",
                "OCA": "OCA", "NPA": "NPA", "PTA": "PTA", "BCA": "BCA",
                "BMA": "BMA", "IMA": "IMA", "ROCA": "ROCA", "RICA": "RICA",
                "RIMA": "RIMA", "RMA": "RMA", "NOPREP": "NOPREP",
                "Lei": "Leis", "Lei Complementar": "Leis Complementares",
                "Decreto": "Decretos", "Decreto-Lei": "Decretos-Lei",
                "Medida Provisória": "Medidas Provisórias",
                "Portaria": "Portarias", "Portaria Conjunta": "Portarias Conjuntas",
                "Resolução": "Resoluções",
                "Instrução Normativa": "Instruções Normativas",
                "Constituição Federal": "Constituição Federal",
                "Ordem técnica": "Ordens Técnicas",
                "Orientação normativa": "Orientações Normativas",
                "Manual Eletrônico": "Manuais Eletrônicos",
            }

            by_type = {
                key: DocumentTypeCount(
                    count=count,
                    label=_TYPE_LABELS.get(key, key),
                )
                for key, count in raw.get("by_type", {}).items()
            }

            relations_raw = raw.get("relations", {})
            relations = RelationStats(
                total=relations_raw.get("total", 0),
                by_type=relations_raw.get("by_type", {}),
                resolved=relations_raw.get("resolved", 0),
                unresolved=relations_raw.get("unresolved", 0),
            )

            doc_stats = DocumentStats(
                by_type=by_type,
                by_status=raw.get("by_status", {}),
                sources=raw.get("by_source", {}),
                total_originals=raw.get("total_originals", 0),
                total_processed=raw.get("total_processed", 0),
                embedded_documents=raw.get("embedded_documents", 0),
                total_chunks=raw.get("total_chunks", 0),
                embedding_models=raw.get("embedding_models", []),
                relations=relations,
                last_updated=raw.get("last_updated"),
                last_embedded=raw.get("last_embedded"),
            )
        except Exception as e:
            logger.warning(f"Could not get document counts: {e}")

        model_info = ModelInfo(
            llm_model=config.OLLAMA_MODEL or "",
            embedding_model=config.EMBEDDING_MODEL or "",
            embedding_dimension=config.EMBEDDING_DIMENSION or 0,
            sparse_model=config.SPARSE_EMBEDDING_MODEL or "",
        )

        return StatsResponse(
            vectors_count=info.get("vectors_count", 0),
            points_count=info.get("points_count", 0),
            status=info.get("status", "unknown"),
            documents=doc_stats,
            model_info=model_info,
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise


# ========================================
# Chatbot Endpoints
# ========================================

def _resolve_session(chat_request: ChatRequest) -> str:
    """Return existing or newly-created session id."""
    if chat_request.session_id:
        session = session_manager.get_session(chat_request.session_id)
        if session is None:
            logger.warning(f"Session {chat_request.session_id} not found, creating new")
            return session_manager.create_session(chat_request.session_id)
        return chat_request.session_id
    return session_manager.create_session()


@app.post("/api/chat/stream")
@limiter.limit(f"{config.RATE_LIMIT}/minute")
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    api_key: str = Depends(verify_api_key),
):
    """Streaming chat endpoint with conversation history and optional RAG.

    Returns Server-Sent Events (SSE) with tokens as they are generated.
    Requires X-API-Key header for authentication.
    """
    try:
        session_id = _resolve_session(chat_request)
        session_manager.add_message(session_id, "user", chat_request.message)

        context_messages = session_manager.get_context_window(
            session_id, last_n=chat_request.context_window,
        )

        rag_limit = chat_request.rag_limit or config.SEARCH_TOP_K

        def generate_stream():
            """Sync generator yielding SSE events.

            Starlette runs this via ``iterate_in_threadpool``, keeping
            the event loop free to flush each chunk immediately.
            """
            start_time = time.time()
            full_response: list[str] = []

            try:
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

                if chat_request.use_rag:
                    result = rag.query(
                        chat_request.message,
                        history=context_messages[:-1],
                        limit=rag_limit,
                        date=chat_request.rag_date,
                        stream=True,
                        temperature=chat_request.temperature,
                        max_tokens=chat_request.max_tokens,
                    )

                    sources_data = [
                        {
                            "regulation_id": s.get("regulation_id"),
                            "score": s.get("score"),
                            "text": s.get("text", "")[:2000],
                        }
                        for s in result["sources"]
                    ] if result["sources"] else []
                    yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data})}\n\n"

                    answer_stream = result.get("answer_stream")
                    if answer_stream:
                        for chunk in answer_stream:
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

                complete_response = "".join(full_response)
                session_manager.add_message(session_id, "assistant", complete_response)

                session = session_manager.get_session(session_id)
                message_count = session.get_message_count() if session else 0
                processing_time_ms = int((time.time() - start_time) * 1000)

                done_payload = {
                    "type": "done",
                    "session_id": session_id,
                    "message_count": message_count,
                    "processing_time_ms": processing_time_ms,
                    "model_used": llm.model_name,
                }
                yield f"data: {json.dumps(done_payload)}\n\n"

            except Exception as e:
                logger.error(f"Error in stream generation: {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except ValueError as e:
        logger.error(f"Validation error in chat stream: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing chat stream request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def _stream_without_rag(request: ChatRequest, context_messages: list):
    """Generate streaming response without RAG context."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(context_messages)

    for chunk in llm.chat(
        messages=messages,
        temperature=request.temperature or 0.7,
        max_tokens=request.max_tokens,
        stream=True,
    ):
        yield chunk


@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit(f"{config.RATE_LIMIT}/minute")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    api_key: str = Depends(verify_api_key),
):
    """Chat endpoint with conversation history and optional RAG.

    Requires X-API-Key header for authentication.
    """
    start_time = time.time()

    try:
        session_id = _resolve_session(chat_request)
        session_manager.add_message(session_id, "user", chat_request.message)

        context_messages = session_manager.get_context_window(
            session_id, last_n=chat_request.context_window,
        )

        if chat_request.use_rag:
            rag_limit = chat_request.rag_limit or config.SEARCH_TOP_K
            result = await asyncio.to_thread(
                rag.query,
                chat_request.message,
                history=context_messages[:-1],
                limit=rag_limit,
                date=chat_request.rag_date,
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens,
            )
            assistant_message = {
                "content": result["answer"],
                "sources": result.get("sources"),
            }
        else:
            assistant_message = await _chat_without_rag(
                request=chat_request,
                context_messages=context_messages,
            )

        session_manager.add_message(session_id, "assistant", assistant_message["content"])

        session = session_manager.get_session(session_id)
        message_count = session.get_message_count() if session else 0
        processing_time_ms = int((time.time() - start_time) * 1000)

        return ChatResponse(
            message=assistant_message["content"],
            session_id=session_id,
            message_count=message_count,
            sources=assistant_message.get("sources"),
            processing_time_ms=processing_time_ms,
            model_used=llm.model_name,
        )

    except ValueError as e:
        logger.error(f"Validation error in chat: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except SearchBackendError:
        raise HTTPException(status_code=503, detail="Search service unavailable")
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def _chat_without_rag(
    request: ChatRequest,
    context_messages: list,
) -> dict:
    """Handle chat without RAG (general conversation)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(context_messages)

    response = await asyncio.to_thread(
        llm.chat,
        messages=messages,
        temperature=request.temperature or 0.7,
        max_tokens=request.max_tokens,
    )

    return {"content": response, "sources": None}


# ========================================
# Session Endpoints
# ========================================

@app.get("/api/chat/session/{session_id}", response_model=SessionInfo)
async def get_session_info(
    session_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Get information about a chat session."""
    session = session_manager.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    return SessionInfo(**session.to_dict())


@app.delete("/api/chat/session/{session_id}")
async def delete_session(
    session_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Delete a chat session."""
    success = session_manager.delete_session(session_id)

    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": "Session deleted successfully", "session_id": session_id}


@app.post("/api/chat/session/{session_id}/clear")
async def clear_session(
    session_id: str,
    api_key: str = Depends(verify_api_key),
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
    """List all available Ollama models."""
    try:
        models_response = llm.client.list()

        if isinstance(models_response, dict) and "models" in models_response:
            models_list = models_response["models"]
        elif isinstance(models_response, (list, tuple)):
            models_list = models_response
        else:
            models_list = getattr(models_response, "models", [])

        models = []
        for m in models_list:
            if isinstance(m, dict):
                models.append(OllamaModel(
                    name=m.get("name", m.get("model", "unknown")),
                    size=_format_size(m.get("size", 0)),
                    modified_at=m.get("modified_at", ""),
                    digest=m.get("digest", "")[:12] if m.get("digest") else None,
                ))
            else:
                models.append(OllamaModel(
                    name=getattr(m, "name", getattr(m, "model", "unknown")),
                    size=_format_size(getattr(m, "size", 0)),
                    modified_at=str(getattr(m, "modified_at", "")),
                    digest=str(getattr(m, "digest", ""))[:12] if getattr(m, "digest", None) else None,
                ))

        logger.info(f"Available models: {[model.name for model in models]}")
        logger.info(f"Current model: {llm.model_name}")
        sorted_models = sorted(models, key=lambda x: x.name.lower())
        return ModelsListResponse(models=sorted_models, current_model=llm.model_name)

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/change", response_model=ChangeModelResponse)
async def change_model(
    request: Request,
    change_request: ChangeModelRequest,
    api_key: str = Depends(verify_api_key),
):
    """Change the current LLM model.

    Because all code paths share the same ``llm`` instance (via DI),
    mutating ``llm.model_name`` takes effect everywhere immediately.
    """
    try:
        previous_model = llm.model_name
        new_model = change_request.model_name

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
                detail=f"Model '{new_model}' not found. Available models: {available_models}",
            )

        llm.model_name = new_model
        logger.info(f"Model changed from '{previous_model}' to '{new_model}'")

        return ChangeModelResponse(
            success=True,
            previous_model=previous_model,
            current_model=new_model,
            message=f"Successfully changed model from '{previous_model}' to '{new_model}'",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# Utilities
# ========================================

def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    if not size_bytes:
        return "N/A"

    for unit in ["B", "KB", "MB", "GB", "TB"]:
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
        reload=config.RELOAD,
    )
