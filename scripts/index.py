"""
Phase 3 – Index pre-computed embeddings into Qdrant.

Reads dense and/or sparse embeddings from Parquet and bulk-upserts them
into the Qdrant collection.

Parquet files are processed **source by source** and streamed in row-group
batches so that memory usage stays bounded regardless of corpus size.

Usage:
    python -m scripts.index
    python -m scripts.index --recreate --batch-size 500
    make index
    make index RECREATE=1
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from config import config
from database.qdrant_manager import QdrantManager
from pipeline.embedding_store import EmbeddingStore


def _detect_mode(emb_store: EmbeddingStore) -> str:
    has_dense = bool(list(emb_store.dense_dir.glob("*.parquet")))
    has_sparse = bool(list(emb_store.sparse_dir.glob("*.parquet")))
    if has_dense and has_sparse:
        return "hybrid"
    if has_sparse:
        return "sparse"
    return "dense"


def _parse_meta(raw: str) -> Dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# Streaming builders — convert one row-group batch into Qdrant points
# ---------------------------------------------------------------------------

def _batch_to_dense_points(batch) -> List[Dict]:
    ids = batch.column("chunk_id").to_pylist()
    vecs = batch.column("dense_vector").to_pylist()
    texts = batch.column("text").to_pylist()
    metas = batch.column("metadata").to_pylist()

    points = []
    for i in range(batch.num_rows):
        payload = {**_parse_meta(metas[i]), "text": texts[i]}
        points.append({"id": ids[i], "vector": vecs[i], "payload": payload})
    return points


def _batch_to_sparse_points(batch) -> List[Dict]:
    from qdrant_client.models import SparseVector

    ids = batch.column("chunk_id").to_pylist()
    idx_col = batch.column("sparse_indices").to_pylist()
    val_col = batch.column("sparse_values").to_pylist()
    texts = batch.column("text").to_pylist()
    metas = batch.column("metadata").to_pylist()

    points = []
    for i in range(batch.num_rows):
        payload = {**_parse_meta(metas[i]), "text": texts[i]}
        points.append({
            "id": ids[i],
            "vector": {"sparse": SparseVector(indices=idx_col[i], values=val_col[i])},
            "payload": payload,
        })
    return points


def _batch_to_hybrid_points(batch, sparse_lookup: Dict) -> List[Dict]:
    from qdrant_client.models import SparseVector

    ids = batch.column("chunk_id").to_pylist()
    vecs = batch.column("dense_vector").to_pylist()
    texts = batch.column("text").to_pylist()
    metas = batch.column("metadata").to_pylist()

    points = []
    for i in range(batch.num_rows):
        cid = ids[i]
        vector: Dict = {"dense": vecs[i]}
        sp = sparse_lookup.get(cid)
        if sp:
            vector["sparse"] = SparseVector(indices=sp[0], values=sp[1])

        payload = {**_parse_meta(metas[i]), "text": texts[i]}
        points.append({"id": cid, "vector": vector, "payload": payload})
    return points


def _build_sparse_lookup(sparse_path: Path) -> Dict:
    """Load sparse vectors into a compact lookup: {chunk_id: (indices, values)}."""
    lookup: Dict = {}
    pf = pq.ParquetFile(sparse_path)
    for batch in pf.iter_batches(batch_size=50_000, columns=["chunk_id", "sparse_indices", "sparse_values"]):
        ids = batch.column("chunk_id").to_pylist()
        idx = batch.column("sparse_indices").to_pylist()
        vals = batch.column("sparse_values").to_pylist()
        for i in range(batch.num_rows):
            lookup[ids[i]] = (idx[i], vals[i])
    return lookup


# ---------------------------------------------------------------------------
# Streaming indexers — process one source file at a time
# ---------------------------------------------------------------------------

_ROW_GROUP_BATCH = 20_000


def _count_rows(directory: Path) -> int:
    return sum(pq.read_metadata(p).num_rows for p in directory.glob("*.parquet"))


def _index_dense_streaming(
    emb_store: EmbeddingStore, db: QdrantManager,
    batch_size: int, workers: int,
) -> int:
    total_rows = _count_rows(emb_store.dense_dir)
    progress = tqdm(total=total_rows, desc="Index [dense]", unit="pts")
    total = 0
    for path in sorted(emb_store.dense_dir.glob("*.parquet")):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=_ROW_GROUP_BATCH):
            points = _batch_to_dense_points(batch)
            db.upsert_points(points, batch_size=batch_size, parallel=workers)
            total += len(points)
            progress.update(len(points))
            del points
    progress.close()
    return total


def _index_sparse_streaming(
    emb_store: EmbeddingStore, db: QdrantManager,
    batch_size: int, workers: int,
) -> int:
    total_rows = _count_rows(emb_store.sparse_dir)
    progress = tqdm(total=total_rows, desc="Index [sparse]", unit="pts")
    total = 0
    for path in sorted(emb_store.sparse_dir.glob("*.parquet")):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=_ROW_GROUP_BATCH):
            points = _batch_to_sparse_points(batch)
            db.upsert_points(points, batch_size=batch_size, parallel=workers)
            total += len(points)
            progress.update(len(points))
            del points
    progress.close()
    return total


def _index_hybrid_streaming(
    emb_store: EmbeddingStore, db: QdrantManager,
    batch_size: int, workers: int,
) -> int:
    total_rows = _count_rows(emb_store.dense_dir)
    progress = tqdm(total=total_rows, desc="Index [hybrid]", unit="pts")
    total = 0
    for dense_path in sorted(emb_store.dense_dir.glob("*.parquet")):
        source = dense_path.stem
        sparse_path = emb_store.sparse_dir / f"{source}.parquet"

        dense_pf = pq.ParquetFile(dense_path)

        sparse_lookup: Optional[Dict] = None
        if sparse_path.exists():
            logger.info(f"[hybrid] {source}: building sparse lookup …")
            sparse_lookup = _build_sparse_lookup(sparse_path)
            logger.info(f"[hybrid] {source}: sparse lookup ready ({len(sparse_lookup)} entries)")
        else:
            logger.warning(f"[hybrid] {source}: no sparse file, dense-only fallback")

        for batch in dense_pf.iter_batches(batch_size=_ROW_GROUP_BATCH):
            if sparse_lookup:
                points = _batch_to_hybrid_points(batch, sparse_lookup)
            else:
                points = _batch_to_dense_points(batch)
            db.upsert_points(points, batch_size=batch_size, parallel=workers)
            total += len(points)
            progress.update(len(points))
            del points

        del sparse_lookup

    progress.close()
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        sparse_only = mode == "sparse"
        if not db.create_collection(recreate=True, sparse_only=sparse_only):
            logger.error("Failed to create Qdrant collection — aborting.")
            return 1

    batch_size = args.batch_size or 500
    logger.info(f"Indexing with batch_size={batch_size}, workers={args.workers}, row_group_batch={_ROW_GROUP_BATCH}")

    db.disable_indexing()
    try:
        if mode == "hybrid":
            total = _index_hybrid_streaming(emb_store, db, batch_size, args.workers)
        elif mode == "sparse":
            total = _index_sparse_streaming(emb_store, db, batch_size, args.workers)
        else:
            total = _index_dense_streaming(emb_store, db, batch_size, args.workers)
    finally:
        db.enable_indexing()

    if args.wait:
        logger.info("Waiting for Qdrant indexing to complete …")
        db.wait_for_indexing(timeout_sec=args.timeout)

    info = db.get_collection_info()
    logger.success(
        f"Indexing complete: {info.get('points_count', '?')} points in Qdrant "
        f"(status={info.get('status', '?')}), total upserted: {total}"
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
        help="Points per upsert batch (default: 500)",
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
