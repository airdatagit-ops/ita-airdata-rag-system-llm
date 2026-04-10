"""RAG pipeline — backward-compatible re-export.

The canonical implementation now lives in ``search.orchestrator.pipeline``.
This module re-exports ``RAGPipeline`` so that existing imports like
``from search.rag import RAGPipeline`` continue to work without changes.
"""

from search.orchestrator.pipeline import RAGPipeline  # noqa: F401

__all__ = ["RAGPipeline"]
