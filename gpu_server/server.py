"""GPU Inference Server — exposes embedding, reranking, and LLM generation via FastAPI.

Deploy this on a machine with GPU(s). Clients call it over HTTP instead of
loading heavy models locally.

Endpoints
---------
POST /v1/embeddings       — dense embeddings (SentenceTransformer)
POST /v1/rerank           — cross-encoder relevance scoring
POST /v1/generate         — LLM chat completion (Ollama proxy)
POST /v1/generate/stream  — LLM streaming chat completion (SSE)
GET  /health              — readiness probe
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration via environment variables
# ---------------------------------------------------------------------------
GPU_SERVER_API_KEY = os.getenv("GPU_SERVER_API_KEY", "")
GPU_SERVER_HOST = os.getenv("GPU_SERVER_HOST", "0.0.0.0")
GPU_SERVER_PORT = int(os.getenv("GPU_SERVER_PORT", "8090"))

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "BAAI/bge-m3",
)
CROSS_ENCODER_MODEL_NAME = os.getenv(
    "CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base",
)
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/dados/airdata/models_cache")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
    texts: List[str]
    normalize: bool = True
    batch_size: int = 32

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    model: str
    elapsed_ms: int

class RerankRequest(BaseModel):
    query: str
    documents: List[Dict[str, Any]]
    max_tokens: int = 480
    batch_size: int = 32

class RerankScore(BaseModel):
    index: int
    score: float

class RerankResponse(BaseModel):
    scores: List[RerankScore]
    model: str
    elapsed_ms: int

class GenerateRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    options: Optional[Dict[str, Any]] = None
    format: Optional[Dict[str, Any]] = None
    stream: bool = False
    # Reasoning models (e.g. gemma4:*) emit a separate ``thinking`` channel
    # that consumes the ``num_predict`` budget. Set ``think=False`` to disable
    # it for short/structured outputs (NL-to-SQL, JSON), or
    # ``"low"|"medium"|"high"`` to bound the reasoning effort.
    think: Optional[Union[bool, str]] = None
    # Forwarded as-is to Ollama; lets callers pin a model in VRAM longer.
    keep_alive: Optional[Union[float, str]] = None

class GenerateResponse(BaseModel):
    content: str
    model: str
    elapsed_ms: int
    # Optional diagnostics — populated when the upstream Ollama response
    # includes them. Adding fields with defaults preserves wire-compat with
    # any pre-existing clients that only deserialize ``content``/``elapsed_ms``.
    thinking: Optional[str] = None
    done_reason: Optional[str] = None
    eval_count: Optional[int] = None
    prompt_eval_count: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    device: str
    gpu_name: Optional[str] = None
    models_loaded: List[str]

# ---------------------------------------------------------------------------
# Model holders (lazy-loaded at startup)
# ---------------------------------------------------------------------------

_embedding_model = None
_cross_encoder = None

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _load_embedding_model():
    global _embedding_model
    from sentence_transformers import SentenceTransformer

    device = _get_device()
    logger.info(f"Loading embedding model {EMBEDDING_MODEL_NAME} on {device}…")
    t0 = time.time()
    _embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME, device=device, cache_folder=MODEL_CACHE_DIR,
    )
    logger.success(
        f"Embedding model loaded in {time.time()-t0:.1f}s "
        f"(dim={_embedding_model.get_sentence_embedding_dimension()}, device={device})"
    )

def _load_cross_encoder():
    global _cross_encoder
    from sentence_transformers import CrossEncoder

    device = _get_device()
    logger.info(f"Loading cross-encoder {CROSS_ENCODER_MODEL_NAME} on {device}…")
    t0 = time.time()
    _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=device)
    logger.success(f"Cross-encoder loaded in {time.time()-t0:.1f}s (device={device})")

def _get_ollama_client():
    from ollama import Client
    return Client(host=OLLAMA_HOST)

# ---------------------------------------------------------------------------
# App lifespan — preload models
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_embedding_model()
    _load_cross_encoder()
    logger.info(f"GPU server ready — listening on {GPU_SERVER_HOST}:{GPU_SERVER_PORT}")
    yield
    logger.info("GPU server shutting down")

app = FastAPI(
    title="GPU Inference Server",
    version="1.0.0",
    description="Serves embeddings, reranking, and LLM inference on GPU",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def _verify_key(x_api_key: Optional[str] = Header(None)):
    if GPU_SERVER_API_KEY and x_api_key != GPU_SERVER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    device = _get_device()
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    loaded = []
    if _embedding_model is not None:
        loaded.append(EMBEDDING_MODEL_NAME)
    if _cross_encoder is not None:
        loaded.append(CROSS_ENCODER_MODEL_NAME)
    return HealthResponse(
        status="healthy",
        device=device,
        gpu_name=gpu_name,
        models_loaded=loaded,
    )


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embed(req: EmbeddingRequest, x_api_key: Optional[str] = Header(None)):
    _verify_key(x_api_key)
    if _embedding_model is None:
        raise HTTPException(503, "Embedding model not loaded")
    if not req.texts:
        raise HTTPException(400, "texts list is empty")

    t0 = time.time()
    embeddings: np.ndarray = _embedding_model.encode(
        req.texts,
        batch_size=req.batch_size,
        normalize_embeddings=req.normalize,
        convert_to_numpy=True,
        device=_get_device(),
    )
    elapsed = int((time.time() - t0) * 1000)

    return EmbeddingResponse(
        embeddings=embeddings.tolist(),
        dimension=embeddings.shape[1],
        model=EMBEDDING_MODEL_NAME,
        elapsed_ms=elapsed,
    )


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest, x_api_key: Optional[str] = Header(None)):
    _verify_key(x_api_key)
    if _cross_encoder is None:
        raise HTTPException(503, "Cross-encoder model not loaded")
    if not req.documents:
        raise HTTPException(400, "documents list is empty")

    t0 = time.time()

    pairs = []
    for doc in req.documents:
        text = _build_eval_text(doc, req.max_tokens)
        pairs.append([req.query, text])

    raw_scores = _cross_encoder.predict(
        pairs, batch_size=req.batch_size, show_progress_bar=False,
    )

    scores = [
        RerankScore(index=i, score=float(s))
        for i, s in enumerate(raw_scores)
    ]

    elapsed = int((time.time() - t0) * 1000)
    return RerankResponse(
        scores=scores, model=CROSS_ENCODER_MODEL_NAME, elapsed_ms=elapsed,
    )


def _resp_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a pydantic model or a plain dict.

    The Ollama Python client returns ``ChatResponse`` (a pydantic model that
    also supports subscript access). Mocked responses in tests are plain
    dicts. This helper keeps the endpoint code agnostic.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _build_chat_kwargs(req: GenerateRequest, *, stream: bool) -> Dict[str, Any]:
    """Translate a :class:`GenerateRequest` into ``ollama.Client.chat`` kwargs.

    Only forwards optional fields when the caller actually set them, so the
    Ollama client falls back to its own defaults instead of receiving
    ``None``s that some upstream versions reject.
    """
    kwargs: Dict[str, Any] = {
        "model": req.model,
        "messages": req.messages,
        "stream": stream,
    }
    if req.options:
        kwargs["options"] = req.options
    if req.format:
        kwargs["format"] = req.format
    if req.think is not None:
        kwargs["think"] = req.think
    if req.keep_alive is not None:
        kwargs["keep_alive"] = req.keep_alive
    return kwargs


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, x_api_key: Optional[str] = Header(None)):
    _verify_key(x_api_key)

    client = _get_ollama_client()
    t0 = time.time()

    kwargs = _build_chat_kwargs(req, stream=False)

    try:
        response = client.chat(**kwargs)
    except Exception as e:
        raise HTTPException(502, f"Ollama error: {e}")

    message = _resp_get(response, "message", {}) or {}
    content = _resp_get(message, "content", "") or ""
    thinking = _resp_get(message, "thinking", "") or ""
    done_reason = _resp_get(response, "done_reason")
    eval_count = _resp_get(response, "eval_count")
    prompt_eval_count = _resp_get(response, "prompt_eval_count")
    elapsed = int((time.time() - t0) * 1000)

    # Reasoning models can spend the entire ``num_predict`` budget inside
    # ``thinking`` and return an empty ``content`` — the caller would
    # otherwise see "10s, no output" with no clue why. Surface it.
    if done_reason == "length" and not content:
        logger.warning(
            f"/v1/generate: empty content with done_reason=length "
            f"(model={req.model}, eval_count={eval_count}, "
            f"thinking_chars={len(thinking)}). "
            f"Increase num_predict or pass think=false."
        )

    return GenerateResponse(
        content=content,
        model=req.model,
        elapsed_ms=elapsed,
        thinking=thinking or None,
        done_reason=done_reason,
        eval_count=eval_count,
        prompt_eval_count=prompt_eval_count,
    )


@app.post("/v1/generate/stream")
async def generate_stream(
    req: GenerateRequest, x_api_key: Optional[str] = Header(None),
):
    _verify_key(x_api_key)
    client = _get_ollama_client()

    kwargs = _build_chat_kwargs(req, stream=True)

    def event_stream():
        try:
            for chunk in client.chat(**kwargs):
                message = _resp_get(chunk, "message", {}) or {}
                content_piece = _resp_get(message, "content", "") or ""
                if content_piece:
                    data = json.dumps({"content": content_piece})
                    yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/v1/models")
async def list_models(x_api_key: Optional[str] = Header(None)):
    """List available Ollama models (proxied from Ollama API)."""
    _verify_key(x_api_key)
    try:
        client = _get_ollama_client()
        response = client.list()

        if isinstance(response, dict) and "models" in response:
            models_list = response["models"]
        elif isinstance(response, (list, tuple)):
            models_list = response
        else:
            models_list = getattr(response, "models", [])

        models = []
        for m in models_list:
            if isinstance(m, dict):
                models.append({
                    "name": m.get("name", m.get("model", "unknown")),
                    "size": m.get("size", 0),
                    "modified_at": str(m.get("modified_at", "")),
                })
            else:
                models.append({
                    "name": getattr(m, "name", getattr(m, "model", "unknown")),
                    "size": getattr(m, "size", 0),
                    "modified_at": str(getattr(m, "modified_at", "")),
                })

        return {"models": models}
    except Exception as e:
        raise HTTPException(502, f"Ollama list error: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_eval_text(doc: Dict[str, Any], max_tokens: int = 480) -> str:
    """Build enriched text for cross-encoder (mirrors DocumentEvaluator logic)."""
    meta = doc.get("metadata") or {}
    prefix_parts: List[str] = []

    title = meta.get("title") or doc.get("title") or ""
    if title:
        prefix_parts.append(title)

    identifiers: List[str] = []
    for key in ("type", "category"):
        val = meta.get(key, "")
        if val:
            identifiers.append(val)
            break
    number = meta.get("number", "")
    authority = meta.get("authority", "")
    if number:
        identifiers.append(f"nº {number}")
    if authority:
        identifiers.append(authority)
    if identifiers:
        prefix_parts.append(" — ".join(identifiers))

    prefix = " | ".join(prefix_parts)
    body = doc.get("text") or ""
    combined = f"{prefix}\n{body}" if prefix else body

    words = combined.split()
    if len(words) > max_tokens:
        return " ".join(words[:max_tokens])
    return combined


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=GPU_SERVER_HOST,
        port=GPU_SERVER_PORT,
        workers=1,
        log_level="info",
    )
