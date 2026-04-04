"""Vector search with support for dense, sparse, or hybrid (RRF) modes."""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from loguru import logger

from models.embeddings import EmbeddingModel, SparseEncoder
from database.qdrant_manager import QdrantManager
from config import config

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
    ):
        if dense_model is _SENTINEL:
            self.dense_model = EmbeddingModel() if config.SEARCH_DENSE_ENABLED else None
        else:
            self.dense_model = dense_model

        if sparse_model is _SENTINEL:
            self.sparse_model = SparseEncoder() if config.SEARCH_SPARSE_ENABLED else None
        else:
            self.sparse_model = sparse_model

        self.db = db or QdrantManager()

        modes = []
        if self.dense_model:
            modes.append("dense")
        if self.sparse_model:
            modes.append("sparse")
        logger.info(f"VectorSearch initialized (modes: {'+'.join(modes)})")

    def _encode_query(self, query: str):
        """Encode query into dense and/or sparse vectors.

        When both models are available the encodings run in parallel.
        """
        if self.dense_model and self.sparse_model:
            with ThreadPoolExecutor(max_workers=2) as pool:
                dense_future = pool.submit(lambda: self.dense_model.encode(query).tolist())
                sparse_future = pool.submit(self.sparse_model.encode_single, query)
                return dense_future.result(), sparse_future.result()

        dense_vector = self.dense_model.encode(query).tolist() if self.dense_model else None
        sparse_vector = self.sparse_model.encode_single(query) if self.sparse_model else None
        return dense_vector, sparse_vector

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
