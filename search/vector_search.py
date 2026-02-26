"""Vector search with temporal filtering."""

from typing import Dict, List, Optional
from loguru import logger

from models.embeddings import EmbeddingModel
from database.qdrant_manager import QdrantManager
from config import config


class VectorSearch:
    """Vector search with temporal and semantic filtering."""

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.db = QdrantManager()
        logger.info("VectorSearch initialized")

    def search(
        self,
        query: str,
        limit: int = None,
        score_threshold: float = None,
        filters: Dict = None
    ) -> List[Dict]:
        """
        Search for similar regulations.

        Args:
            query: Search query
            limit: Number of results
            score_threshold: Minimum similarity
            filters: Additional filters

        Returns:
            List of results with metadata
        """
        # Embed query
        query_vector = self.embedding_model.encode(query)

        # Search
        results = self.db.search(
            query_vector=query_vector.tolist(),
            limit=limit or config.SEARCH_TOP_K,
            score_threshold=score_threshold or config.SEARCH_SCORE_THRESHOLD,
            filters=filters
        )

        # Format results
        formatted = []
        for result in results:
            formatted.append({
                "regulation_id": result.payload.get("regulation_id"),
                "text": result.payload.get("text"),
                "score": result.score,
                "metadata": result.payload.get("metadata", {}),
                **{k: v for k, v in result.payload.items()
                   if k not in ["text", "regulation_id", "metadata"]}
            })

        logger.info(f"Found {len(formatted)} results for query: {query[:50]}...")
        return formatted

    def search_temporal(
        self,
        query: str,
        date: str,
        limit: int = None,
        **kwargs
    ) -> List[Dict]:
        """
        Search for regulations valid on a specific date.

        Args:
            query: Search query
            date: Target date (ISO format)
            limit: Number of results

        Returns:
            List of results valid on date
        """
        query_vector = self.embedding_model.encode(query)

        results = self.db.search_temporal(
            query_vector=query_vector.tolist(),
            target_date=date,
            limit=limit or config.SEARCH_TOP_K
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
                "metadata": result.payload.get("metadata", {})
            })

        logger.info(f"Found {len(formatted)} results valid on {date}")
        return formatted


if __name__ == "__main__":
    search = VectorSearch()
    print("VectorSearch ready")
