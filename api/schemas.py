"""Pydantic schemas for API."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search request schema."""
    query: str = Field(..., max_length=1000, description="Search query")
    date: Optional[str] = Field(None, description="Target date (ISO format: YYYY-MM-DD)")
    limit: int = Field(5, ge=1, le=50, description="Number of results")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    filters: Optional[Dict] = Field(None, description="Additional filters")


class SourceDocument(BaseModel):
    """Source document schema."""
    regulation_id: str
    text: str
    score: float
    version: Optional[str] = None
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    metadata: Dict = {}


class SearchResponse(BaseModel):
    """Search response schema."""
    answer: str
    sources: List[SourceDocument]
    search_time_ms: int
    llm_time_ms: int
    total_time_ms: int


class VectorSearchRequest(BaseModel):
    """Vector-only search request schema (no LLM)."""
    query: str = Field(..., max_length=1000, description="Search query")
    date: Optional[str] = Field(None, description="Target date (ISO format: YYYY-MM-DD)")
    limit: int = Field(5, ge=1, le=50, description="Number of results")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class VectorSearchResponse(BaseModel):
    """Vector-only search response schema (no LLM)."""
    query: str
    sources: List[SourceDocument]
    total_results: int
    search_time_ms: int


class DocumentTypeCount(BaseModel):
    """Count for a document type."""
    count: int
    label: str


class RelationStats(BaseModel):
    """Document relation statistics."""
    total: int = 0
    by_type: Dict[str, int] = {}
    resolved: int = 0
    unresolved: int = 0


class DocumentStats(BaseModel):
    """Document statistics."""
    by_type: Dict[str, DocumentTypeCount] = {}
    by_status: Dict[str, int] = {}
    sources: Dict[str, int] = {}
    total_originals: int = 0
    total_processed: int = 0
    embedded_documents: int = 0
    total_chunks: int = 0
    embedding_models: List[str] = []
    relations: Optional[RelationStats] = None
    last_updated: Optional[str] = None
    last_embedded: Optional[str] = None


class ModelInfo(BaseModel):
    """Information about models in use."""
    llm_model: str = ""
    embedding_model: str = ""
    embedding_dimension: int = 0
    sparse_model: str = ""


class StatsResponse(BaseModel):
    """System statistics response."""
    vectors_count: int
    points_count: int
    status: str
    documents: Optional[DocumentStats] = None
    model_info: Optional[ModelInfo] = None


# ========================================
# Chatbot Schemas
# ========================================

class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., max_length=5000, description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp (ISO format)")


class ChatRequest(BaseModel):
    """Chat request schema."""
    message: str = Field(..., max_length=5000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    use_rag: bool = Field(False, description="Whether to use RAG for context")
    rag_date: Optional[str] = Field(None, description="Date for RAG temporal search (YYYY-MM-DD)")
    context_window: int = Field(10, ge=1, le=50, description="Number of previous messages to include")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="LLM temperature override")
    max_tokens: Optional[int] = Field(None, ge=50, le=2000, description="Max tokens override")


class ChatResponse(BaseModel):
    """Chat response schema."""
    message: str = Field(..., description="Assistant's response")
    session_id: str = Field(..., description="Session ID")
    message_count: int = Field(..., description="Total messages in session")
    sources: Optional[List[SourceDocument]] = Field(None, description="RAG sources if used")
    processing_time_ms: int = Field(..., description="Total processing time")
    model_used: str = Field(..., description="LLM model used")


class SessionInfo(BaseModel):
    """Session information schema."""
    session_id: str
    message_count: int
    created_at: str
    last_accessed: str
    age_minutes: float
    metadata: Dict = {}


class SessionStatsResponse(BaseModel):
    """Session statistics response."""
    active_sessions: int
    total_messages: int
    sessions_created: int
    sessions_expired: int
    ttl_minutes: int
    max_sessions: int


# ========================================
# Model Management Schemas
# ========================================

class OllamaModel(BaseModel):
    """Ollama model information."""
    name: str = Field(..., description="Model name")
    size: Optional[str] = Field(None, description="Model size")
    modified_at: Optional[str] = Field(None, description="Last modified date")
    digest: Optional[str] = Field(None, description="Model digest")


class ModelsListResponse(BaseModel):
    """Response for listing available models."""
    models: List[OllamaModel]
    current_model: str


class ChangeModelRequest(BaseModel):
    """Request to change the current model."""
    model_name: str = Field(..., description="Name of the model to use")


class ChangeModelResponse(BaseModel):
    """Response after changing model."""
    success: bool
    previous_model: str
    current_model: str
    message: str
