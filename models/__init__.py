"""
Models module for Aviation RAG System.

This module provides interfaces to ML models:
- Embedding models (Legal-BERTimbau)
- LLM models (Llama via Ollama)
"""

from models.embeddings import EmbeddingModel
from models.llm import LlamaModel

__all__ = ["EmbeddingModel", "LlamaModel"]
