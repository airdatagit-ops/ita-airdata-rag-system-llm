"""
Qdrant Manager for Aviation RAG System.

Handles all interactions with Qdrant vector database.
Supports dense-only, sparse-only, or hybrid (RRF) search via config flags.
"""

import uuid
from typing import Dict, List, Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    DatetimeRange, MatchValue, PayloadSchemaType, HnswConfigDiff,
    IsNullCondition, PayloadField, SparseVectorParams, SparseVector,
    SearchParams, Prefetch, FusionQuery, Fusion,
    OptimizersConfigDiff,
)

from config import config
from search.exceptions import SearchBackendError

PREFETCH_MULTIPLIER = 3


class QdrantManager:
    """Manager for Qdrant vector database operations."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = None,
        api_key: str = None
    ):
        self.host = host or config.QDRANT_HOST
        self.port = port or config.QDRANT_PORT
        self.collection_name = collection_name or config.QDRANT_COLLECTION_NAME
        self.api_key = api_key or config.QDRANT_API_KEY

        if self.api_key:
            self.client = QdrantClient(
                url=self.host, api_key=self.api_key,
                prefer_grpc=True, timeout=120,
            )
        else:
            self.client = QdrantClient(
                host=self.host, port=self.port,
                prefer_grpc=True, timeout=120,
            )

        logger.info(f"QdrantManager initialized ({self.host}:{self.port})")

    def create_collection(
        self,
        vector_size: int = None,
        distance: Distance = Distance.COSINE,
        recreate: bool = False,
        sparse_only: bool = False,
    ) -> bool:
        """
        Create collection.

        When sparse_only=True, creates a collection with only sparse vectors
        (no dense vector config needed). Otherwise creates named vectors
        ("dense" + "sparse") for hybrid search.
        """
        import time

        vector_size = vector_size or config.EMBEDDING_DIMENSION

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and not recreate:
            logger.info(f"Collection '{self.collection_name}' already exists")
            return True

        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            logger.warning(f"Deleted existing collection '{self.collection_name}'")
            time.sleep(2)

        vectors_config = {}
        if not sparse_only:
            vectors_config["dense"] = VectorParams(
                size=vector_size,
                distance=distance,
                hnsw_config=HnswConfigDiff(
                    m=config.HNSW_M,
                    ef_construct=config.HNSW_EF_CONSTRUCT,
                ),
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config or None,
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        logger.success(
            f"Created collection '{self.collection_name}' "
            f"({'sparse-only' if sparse_only else 'dense+sparse'})"
        )

        self._create_payload_indexes()
        return True

    def _create_payload_indexes(self):
        """Create indexes on payload fields for fast filtering."""
        indexes = [
            ("effective_date", PayloadSchemaType.DATETIME),
            ("expiry_date", PayloadSchemaType.DATETIME),
            ("status", PayloadSchemaType.KEYWORD),
            ("regulation_id", PayloadSchemaType.KEYWORD),
            ("version_year", PayloadSchemaType.KEYWORD),
            ("is_latest", PayloadSchemaType.BOOL),
            ("canonical_id", PayloadSchemaType.KEYWORD),
            ("metadata.category", PayloadSchemaType.KEYWORD),
            ("metadata.source", PayloadSchemaType.KEYWORD),
            ("metadata.authority", PayloadSchemaType.KEYWORD),
            ("metadata.number", PayloadSchemaType.KEYWORD),
        ]

        for field_name, schema_type in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type
                )
                logger.debug(f"Created index on '{field_name}'")
            except Exception as e:
                logger.warning(f"Could not create index on '{field_name}': {e}")

    def disable_indexing(self):
        """Disable HNSW indexing for faster bulk uploads."""
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=OptimizersConfigDiff(indexing_threshold=0),
        )
        logger.info("Indexing disabled (threshold=0) for bulk upload")

    def enable_indexing(self, threshold: int = 20_000):
        """Re-enable HNSW indexing after bulk upload."""
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=OptimizersConfigDiff(indexing_threshold=threshold),
        )
        logger.info(f"Indexing re-enabled (threshold={threshold})")

    def wait_for_indexing(self, timeout_sec: int = 300):
        """Block until the collection finishes indexing (status=green)."""
        import time

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            info = self.client.get_collection(self.collection_name)
            if info.status.name == "GREEN":
                logger.success("Collection indexing complete")
                return
            time.sleep(2)
        logger.warning("Indexing did not finish within timeout")

    def upsert_points(
        self,
        points: List[Dict],
        batch_size: int = None,
        parallel: int = None,
        wait: bool = False,
    ) -> bool:
        """
        Upsert points to the collection.

        The "vector" field can be a plain list (unnamed, when sparse is off)
        or a dict of named vectors (when sparse is on).

        Args:
            batch_size: Points per batch (default: config.INGESTION_BATCH_SIZE).
            parallel: Parallel upload workers (default: config.NUM_WORKERS).
            wait: Block until each batch is indexed. Use False for bulk
                  uploads when indexing is disabled; True for single-point
                  inserts that need immediate consistency.
        """
        batch_size = batch_size or config.INGESTION_BATCH_SIZE
        parallel = parallel or config.NUM_WORKERS

        try:
            qdrant_points = [
                PointStruct(
                    id=p.get("id") or str(uuid.uuid4()),
                    vector=p["vector"],
                    payload=p.get("payload", {}),
                )
                for p in points
            ]

            self.client.upload_points(
                collection_name=self.collection_name,
                points=qdrant_points,
                batch_size=batch_size,
                parallel=parallel,
                max_retries=3,
                wait=wait,
            )

            logger.success(f"Upserted {len(points)} points")
            return True

        except Exception as e:
            logger.error(f"Error upserting points: {e}")
            return False

    def search(
        self,
        dense_vector: Optional[List[float]] = None,
        sparse_vector: Optional[SparseVector] = None,
        limit: int = None,
        score_threshold: float = None,
        filters=None,
        with_payload: bool = True,
        *,
        query_vector: Optional[List[float]] = None,
    ) -> List:
        """
        Search using dense, sparse, or hybrid (RRF) depending on provided vectors.

        Args:
            dense_vector: Dense embedding for semantic search
            sparse_vector: Sparse vector for keyword/BM25 search
            limit: Number of results
            score_threshold: Min score (only for dense-only search; ignored in hybrid/sparse)
            filters: Qdrant filter
            with_payload: Return payload with results
            query_vector: Deprecated alias for dense_vector (backward compat)
        """
        if query_vector is not None and dense_vector is None:
            dense_vector = query_vector

        if dense_vector is None and sparse_vector is None:
            raise ValueError("At least one of dense_vector or sparse_vector is required")

        if filters is None:
            filters = Filter(must=[
                FieldCondition(key="status", match=MatchValue(value="active")),
                FieldCondition(key="is_latest", match=MatchValue(value=True)),
            ])

        limit = limit or config.SEARCH_TOP_K
        has_dense = dense_vector is not None
        has_sparse = sparse_vector is not None

        try:
            if has_dense and has_sparse:
                results = self._search_hybrid(dense_vector, sparse_vector, limit, filters, with_payload)
            elif has_dense:
                results = self._search_dense(dense_vector, limit, score_threshold, filters, with_payload)
            else:
                results = self._search_sparse(sparse_vector, limit, filters, with_payload)

            points = results.points if hasattr(results, 'points') else results
            logger.debug(f"Search returned {len(points)} results")
            return points

        except Exception as e:
            raise SearchBackendError(f"Qdrant search failed: {e}") from e

    def _search_hybrid(self, dense_vector, sparse_vector, limit, filters, with_payload):
        """Hybrid search: prefetch from both branches, fuse with RRF."""
        prefetch_limit = limit * PREFETCH_MULTIPLIER
        return self.client.query_points(
            collection_name=self.collection_name,
            query=FusionQuery(fusion=Fusion.RRF),
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=prefetch_limit,
                    params=SearchParams(hnsw_ef=config.HNSW_EF_SEARCH),
                ),
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=prefetch_limit,
                ),
            ],
            limit=limit,
            query_filter=filters,
            with_payload=with_payload,
        )

    def _search_dense(self, dense_vector, limit, score_threshold, filters, with_payload):
        """Dense-only semantic search."""
        if score_threshold is None:
            score_threshold = config.SEARCH_SCORE_THRESHOLD

        params = {
            "collection_name": self.collection_name,
            "query": dense_vector,
            "using": "dense",
            "limit": limit,
            "query_filter": filters,
            "with_payload": with_payload,
            "search_params": SearchParams(hnsw_ef=config.HNSW_EF_SEARCH),
        }
        if score_threshold and score_threshold > 0:
            params["score_threshold"] = score_threshold

        return self.client.query_points(**params)

    def _search_sparse(self, sparse_vector, limit, filters, with_payload):
        """Sparse-only keyword/BM25 search."""
        return self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_vector,
            using="sparse",
            limit=limit,
            query_filter=filters,
            with_payload=with_payload,
        )

    def search_temporal(
        self,
        target_date: str = None,
        limit: int = None,
        dense_vector: Optional[List[float]] = None,
        sparse_vector: Optional[SparseVector] = None,
        additional_filters=None,
        *,
        query_vector: Optional[List[float]] = None,
    ) -> List:
        """
        Search with temporal filtering (regulations valid on target_date).
        """
        if query_vector is not None and dense_vector is None:
            dense_vector = query_vector
        temporal_filter = Filter(
            must=[
                FieldCondition(key="status", match=MatchValue(value="active")),
                FieldCondition(key="effective_date", range=DatetimeRange(lte=target_date)),
            ],
            should=[
                FieldCondition(key="expiry_date", range=DatetimeRange(gte=target_date)),
                IsNullCondition(is_null=PayloadField(key="expiry_date")),
            ]
        )

        if additional_filters:
            if isinstance(additional_filters, Filter):
                temporal_filter.must.extend(additional_filters.must or [])
            elif isinstance(additional_filters, dict):
                temporal_filter.must.append(additional_filters)

        return self.search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=limit,
            filters=temporal_filter,
        )

    def get_collection_info(self) -> Dict:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            points_count = getattr(info, 'points_count', 0)

            return {
                "vectors_count": points_count,
                "points_count": points_count,
                "status": info.status.name.lower(),
                "indexed_vectors_count": points_count,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"vectors_count": 0, "points_count": 0, "status": "unknown"}


if __name__ == "__main__":
    manager = QdrantManager()
    info = manager.get_collection_info()
    print(f"Collection info: {info}")
