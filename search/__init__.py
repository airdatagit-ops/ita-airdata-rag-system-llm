"""Search module for Aviation RAG System.

Public API re-exports for the modular RAG pipeline.
"""

from search.shared.exceptions import SearchError, SearchBackendError
from search.orchestrator import RAGPipeline
from search.rewriter import QueryRewriter
from search.searcher import DocumentSearcher
from search.evaluator import DocumentEvaluator
from search.generator import ResponseGenerator

__all__ = [
    "SearchError",
    "SearchBackendError",
    "RAGPipeline",
    "QueryRewriter",
    "DocumentSearcher",
    "DocumentEvaluator",
    "ResponseGenerator",
]
