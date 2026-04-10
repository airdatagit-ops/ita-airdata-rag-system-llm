"""Search module for Aviation RAG System.

Public API re-exports for the modular RAG pipeline.
Imports are lazy to avoid circular dependencies with the database layer.
"""


def __getattr__(name: str):
    """Lazy import to break circular dependency chains."""
    _exports = {
        "SearchError": ("search.shared.exceptions", "SearchError"),
        "SearchBackendError": ("search.shared.exceptions", "SearchBackendError"),
        "RAGPipeline": ("search.orchestrator.pipeline", "RAGPipeline"),
        "QueryRewriter": ("search.rewriter.rewriter", "QueryRewriter"),
        "DocumentSearcher": ("search.searcher.searcher", "DocumentSearcher"),
        "DocumentEvaluator": ("search.evaluator.evaluator", "DocumentEvaluator"),
        "ResponseGenerator": ("search.generator.generator", "ResponseGenerator"),
    }

    if name in _exports:
        module_path, attr = _exports[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)

    raise AttributeError(f"module 'search' has no attribute {name!r}")


__all__ = [
    "SearchError",
    "SearchBackendError",
    "RAGPipeline",
    "QueryRewriter",
    "DocumentSearcher",
    "DocumentEvaluator",
    "ResponseGenerator",
]
