"""Custom exceptions for the search module."""


class SearchError(Exception):
    """Base exception for search operations."""


class SearchBackendError(SearchError):
    """Raised when the vector database or other search backend fails."""
