"""Exceptions for the modular RAG pipeline.

Each module has its own exception class so callers (and the orchestrator)
can handle failures per-stage with appropriate fallbacks.
"""


class SearchError(Exception):
    """Base exception for all search/pipeline operations."""


class SearchBackendError(SearchError):
    """Raised when the vector database or other search backend fails."""


class PipelineTimeoutError(SearchError):
    """Raised when a pipeline stage exceeds its configured timeout."""

    def __init__(self, stage: str, timeout_seconds: int, message: str = ""):
        self.stage = stage
        self.timeout_seconds = timeout_seconds
        super().__init__(
            message
            or f"{stage} exceeded timeout of {timeout_seconds}s"
        )


class RewriterError(SearchError):
    """Raised when query rewriting fails."""


class SearcherError(SearchError):
    """Raised when the search stage fails."""


class EvaluatorError(SearchError):
    """Raised when document evaluation/reranking fails."""


class GeneratorError(SearchError):
    """Raised when response generation fails."""
