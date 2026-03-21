"""
Phase 3 – Index pre-computed embeddings into Qdrant.

Reads dense and/or sparse embeddings from Parquet and bulk-upserts them
into the Qdrant collection.  No embedding model is loaded, so this phase
is purely I/O-bound and runs in seconds.

Usage:
    python -m scripts.index
    python -m scripts.index --recreate --batch-size 500
    make index
    make index RECREATE=1
"""

import argparse
import json
from typing import Dict, List

import pyarrow as pa
from loguru import logger

from config import config
from database.qdrant_manager import QdrantManager
from pipeline.embedding_store import EmbeddingStore


def _detect_mode(emb_store: EmbeddingStore) -> str:
    """Detect which embedding types are available on disk."""
    has_dense = bool(list(emb_store.dense_dir.glob("*.parquet")))
    has_sparse = bool(list(emb_store.sparse_dir.glob("*.parquet")))
    if has_dense and has_sparse:
        return "hybrid"
    if has_sparse:
        return "sparse"
    return "dense"


def _build_points_dense(table: pa.Table) -> List[Dict]:
    """Convert a dense Parquet table into Qdrant point dicts."""
    points = []
    chunk_ids = table.column("chunk_id").to_pylist()
    vectors = table.column("dense_vector").to_pylist()
    texts = table.column("text").to_pylist()
    metadata_col = table.column("metadata").to_pylist()

    for i in range(table.num_rows):
        meta = {}
        if metadata_col[i]:
            try:
                meta = json.loads(metadata_col[i])
            except (json.JSONDecodeError, TypeError):
                pass

        payload = {**meta, "text": texts[i]}
        points.append({
            "id": chunk_ids[i],
            "vector": vectors[i],
            "payload": payload,
        })
    return points


def _build_points_hybrid(
    dense_table: pa.Table,
    sparse_table: pa.Table,
) -> List[Dict]:
    """Merge dense + sparse tables into hybrid point dicts."""
    sparse_lookup: Dict[str, Dict] = {}
    s_ids = sparse_table.column("chunk_id").to_pylist()
    s_indices = sparse_table.column("sparse_indices").to_pylist()
    s_values = sparse_table.column("sparse_values").to_pylist()
    for i in range(sparse_table.num_rows):
        sparse_lookup[s_ids[i]] = {
            "indices": s_indices[i],
            "values": s_values[i],
        }

    points = []
    d_ids = dense_table.column("chunk_id").to_pylist()
    d_vectors = dense_table.column("dense_vector").to_pylist()
    d_texts = dense_table.column("text").to_pylist()
    d_meta = dense_table.column("metadata").to_pylist()

    from qdrant_client.models import SparseVector

    for i in range(dense_table.num_rows):
        chunk_id = d_ids[i]
        meta = {}
        if d_meta[i]:
            try:
                meta = json.loads(d_meta[i])
            except (json.JSONDecodeError, TypeError):
                pass

        vector: Dict = {"dense": d_vectors[i]}
        sp = sparse_lookup.get(chunk_id)
        if sp:
            vector["sparse"] = SparseVector(
                indices=sp["indices"], values=sp["values"]
            )

        payload = {**meta, "text": d_texts[i]}
        points.append({
            "id": chunk_id,
            "vector": vector,
            "payload": payload,
        })
    return points


def _build_points_sparse(table: pa.Table) -> List[Dict]:
    """Convert a sparse-only Parquet table into Qdrant point dicts."""
    from qdrant_client.models import SparseVector

    points = []
    chunk_ids = table.column("chunk_id").to_pylist()
    indices_col = table.column("sparse_indices").to_pylist()
    values_col = table.column("sparse_values").to_pylist()
    texts = table.column("text").to_pylist()
    metadata_col = table.column("metadata").to_pylist()

    for i in range(table.num_rows):
        meta = {}
        if metadata_col[i]:
            try:
                meta = json.loads(metadata_col[i])
            except (json.JSONDecodeError, TypeError):
                pass

        payload = {**meta, "text": texts[i]}
        points.append({
            "id": chunk_ids[i],
            "vector": {"sparse": SparseVector(
                indices=indices_col[i], values=values_col[i],
            )},
            "payload": payload,
        })
    return points


# ── main logic ───────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    emb_store = EmbeddingStore()
    mode = _detect_mode(emb_store)
    use_named = mode in ("hybrid", "sparse")

    logger.info(f"Detected embedding mode: {mode}")

    db = QdrantManager()

    if args.recreate:
        logger.info("Recreating Qdrant collection …")
        if use_named:
            config.SEARCH_SPARSE_ENABLED = True
        db.create_collection(recreate=True)

    dense_table = emb_store.load_dense() if mode in ("dense", "hybrid") else None
    sparse_table = emb_store.load_sparse() if mode in ("sparse", "hybrid") else None

    if dense_table is None and sparse_table is None:
        logger.warning("No embeddings found in store. Run `make embed` first.")
        return 1

    if mode == "hybrid" and dense_table is not None and sparse_table is not None:
        logger.info(
            f"Building hybrid points: {dense_table.num_rows} dense, "
            f"{sparse_table.num_rows} sparse"
        )
        points = _build_points_hybrid(dense_table, sparse_table)
    elif mode == "sparse" and sparse_table is not None:
        logger.info(f"Building sparse-only points: {sparse_table.num_rows}")
        points = _build_points_sparse(sparse_table)
    elif dense_table is not None:
        logger.info(f"Building dense-only points: {dense_table.num_rows}")
        points = _build_points_dense(dense_table)
    else:
        logger.error("Inconsistent embedding state")
        return 1

    batch_size = args.batch_size or 500
    logger.info(
        f"Upserting {len(points)} points to Qdrant "
        f"(batch_size={batch_size}, workers={args.workers}) …"
    )

    db.disable_indexing()
    try:
        db.upsert_points(
            points, batch_size=batch_size, parallel=args.workers,
        )
    finally:
        db.enable_indexing()

    if args.wait:
        logger.info("Waiting for Qdrant indexing to complete …")
        db.wait_for_indexing(timeout_sec=args.timeout)

    info = db.get_collection_info()
    logger.success(
        f"Indexing complete: {info.get('points_count', '?')} points in Qdrant "
        f"(status={info.get('status', '?')})"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 – Index embeddings into Qdrant")
    parser.add_argument(
        "--recreate", action="store_true",
        help="Drop and recreate the Qdrant collection",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Points per upsert batch (default: from config)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel upload workers",
    )
    parser.add_argument(
        "--wait", action="store_true", default=True,
        help="Wait for Qdrant indexing to finish",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Max seconds to wait for indexing",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
