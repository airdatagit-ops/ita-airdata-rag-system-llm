"""Vector search with support for dense, sparse, or hybrid (RRF) modes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Dict, List, Optional

from loguru import logger

from models.gpu_client import create_embedding_model
from database.qdrant_manager import QdrantManager
from search.cache import CacheBackend, make_cache_key
from config import config

if TYPE_CHECKING:
    from models.embeddings import EmbeddingModel, SparseEncoder

_SENTINEL = object()


class VectorSearch:
    """Vector search with temporal and semantic filtering.

    Accepts pre-built instances via constructor (dependency injection).
    When a dependency is not provided, a new instance is created based
    on the current config flags.  Pass ``None`` explicitly to disable
    a component (e.g. ``dense_model=None`` disables dense search).
    """

    def __init__(
        self,
        dense_model: Optional[EmbeddingModel] = _SENTINEL,
        sparse_model: Optional[SparseEncoder] = _SENTINEL,
        db: Optional[QdrantManager] = None,
        embedding_cache: Optional[CacheBackend] = None,
    ):
        self.db = db or QdrantManager()

        if dense_model is _SENTINEL:
            want_dense = config.SEARCH_DENSE_ENABLED
            if want_dense and not self.db.has_dense:
                logger.warning(
                    "SEARCH_DENSE_ENABLED=true but collection has no dense vectors "
                    "— skipping dense model load"
                )
                want_dense = False
            self.dense_model = create_embedding_model() if want_dense else None
        else:
            self.dense_model = dense_model

        if sparse_model is _SENTINEL:
            want_sparse = config.SEARCH_SPARSE_ENABLED
            if want_sparse and not self.db.has_sparse:
                logger.warning(
                    "SEARCH_SPARSE_ENABLED=true but collection has no sparse vectors "
                    "— skipping sparse model load"
                )
                want_sparse = False
            if want_sparse:
                from models.embeddings import SparseEncoder as _SparseEncoder
                self.sparse_model = _SparseEncoder()
            else:
                self.sparse_model = None
        else:
            self.sparse_model = sparse_model

        self._embedding_cache = embedding_cache

        modes = []
        if self.dense_model:
            modes.append("dense")
        if self.sparse_model:
            modes.append("sparse")
        logger.info(f"VectorSearch initialized (modes: {'+'.join(modes) or 'none'})")

    def _encode_query(self, query: str):
        """Encode query into dense and/or sparse vectors.

        Results are cached when an ``embedding_cache`` is provided.
        When both models are available the encodings run in parallel.
        """
        if self._embedding_cache is not None:
            key = make_cache_key("emb", query)
            cached = self._embedding_cache.get(key)
            if cached is not None:
                logger.debug(f"Embedding cache hit for: {query[:50]}...")
                return cached

        if self.dense_model and self.sparse_model:
            with ThreadPoolExecutor(max_workers=2) as pool:
                dense_future = pool.submit(lambda: self.dense_model.encode(query).tolist())
                sparse_future = pool.submit(self.sparse_model.encode_single, query)
                result = dense_future.result(), sparse_future.result()
        else:
            dense_vector = self.dense_model.encode(query).tolist() if self.dense_model else None
            sparse_vector = self.sparse_model.encode_single(query) if self.sparse_model else None
            result = dense_vector, sparse_vector

        if self._embedding_cache is not None:
            self._embedding_cache.set(key, result)

        return result

    @staticmethod
    def _format_results(results) -> List[Dict]:
        """Format Qdrant results into a uniform dict structure."""
        formatted = []
        for result in results:
            entry = {
                "regulation_id": result.payload.get("regulation_id"),
                "text": result.payload.get("text"),
                "score": result.score,
                "metadata": result.payload.get("metadata", {}),
            }
            entry.update(
                {k: v for k, v in result.payload.items()
                 if k not in ("text", "regulation_id", "metadata")}
            )
            formatted.append(entry)
        return formatted

    def search(
        self,
        query: str,
        limit: int = None,
        score_threshold: float = None,
        filters: Dict = None
    ) -> List[Dict]:
        """Search for similar regulations using configured search modes.

        When both dense and sparse are enabled, uses Reciprocal Rank Fusion.
        Raises ``SearchBackendError`` on infrastructure failures.
        """
        dense_vector, sparse_vector = self._encode_query(query)

        results = self.db.search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=limit or config.SEARCH_TOP_K,
            score_threshold=score_threshold,
            filters=filters,
        )

        formatted = self._format_results(results)
        logger.info(f"Found {len(formatted)} results for query: {query[:50]}...")
        return formatted

    def search_temporal(
        self,
        query: str,
        date: str,
        limit: int = None,
        **kwargs
    ) -> List[Dict]:
        """Search for regulations valid on a specific date."""
        dense_vector, sparse_vector = self._encode_query(query)

        results = self.db.search_temporal(
            target_date=date,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=limit or config.SEARCH_TOP_K,
        )

        formatted = []
        for result in results:
            formatted.append({
                "regulation_id": result.payload.get("regulation_id"),
                "version": result.payload.get("version"),
                "text": result.payload.get("text"),
                "score": result.score,
                "effective_date": result.payload.get("effective_date"),
                "expiry_date": result.payload.get("expiry_date"),
                "metadata": result.payload.get("metadata", {}),
            })

        logger.info(f"Found {len(formatted)} results valid on {date}")
        return formatted


if __name__ == "__main__":
    search = VectorSearch()
    print("VectorSearch ready")
