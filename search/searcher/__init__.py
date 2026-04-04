"""Document searcher module for the RAG pipeline."""

from search.searcher.searcher import DocumentSearcher
from search.searcher.enrichment import enrich_documents

__all__ = ["DocumentSearcher", "enrich_documents"]
