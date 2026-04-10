"""
Models module for Aviation RAG System.

This module provides interfaces to ML models:
- Embedding models (Legal-BERTimbau)
- LLM models (Llama via Ollama)
- GPU client (remote inference via HTTP)

Heavy imports (torch, sentence-transformers, ollama) are deferred so that
``INFERENCE_MODE=remote`` starts without loading any ML libraries.
"""

from models.gpu_client import (
    create_embedding_model,
    create_evaluator,
    create_llm,
)


def __getattr__(name: str):
    """Lazy-load heavy model classes only when accessed by name."""
    if name == "EmbeddingModel":
        from models.embeddings import EmbeddingModel
        return EmbeddingModel
    if name == "LlamaModel":
        from models.llm import LlamaModel
        return LlamaModel
    raise AttributeError(f"module 'models' has no attribute {name!r}")


__all__ = [
    "EmbeddingModel",
    "LlamaModel",
    "create_embedding_model",
    "create_evaluator",
    "create_llm",
]
