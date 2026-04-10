"""Backward-compatible re-export of search exceptions.

The canonical definitions now live in ``search.shared.exceptions``.
"""

from search.shared.exceptions import (  # noqa: F401
    SearchError,
    SearchBackendError,
    RewriterError,
    SearcherError,
    EvaluatorError,
    GeneratorError,
    PipelineTimeoutError,
)

__all__ = [
    "SearchError",
    "SearchBackendError",
    "RewriterError",
    "SearcherError",
    "EvaluatorError",
    "GeneratorError",
    "PipelineTimeoutError",
]
