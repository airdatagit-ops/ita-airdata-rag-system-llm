"""
Parquet-backed embedding store for the 3-phase ingestion pipeline.

Stores dense and sparse embeddings in partitioned Parquet files for
fast batch loading during the indexing phase.

Usage:
    from pipeline.embedding_store import EmbeddingStore

    store = EmbeddingStore()
    store.save_dense(df, source="lexml")
    dense_df = store.load_dense()
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, Optional, Set

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from config import config

_DENSE_SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("doc_id", pa.string()),
    ("chunk_index", pa.int32()),
    ("text", pa.string()),
    ("dense_vector", pa.list_(pa.float32())),
    ("content_hash", pa.string()),
    ("metadata", pa.string()),
])

_SPARSE_SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("doc_id", pa.string()),
    ("chunk_index", pa.int32()),
    ("text", pa.string()),
    ("sparse_indices", pa.list_(pa.int64())),
    ("sparse_values", pa.list_(pa.float32())),
    ("content_hash", pa.string()),
    ("metadata", pa.string()),
])


class EmbeddingStore:
    """Read/write embeddings as partitioned Parquet files."""

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or config.EMBEDDINGS_DIR)
        self.dense_dir = self.base_dir / "dense"
        self.sparse_dir = self.base_dir / "sparse"
        self.dense_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"EmbeddingStore ready ({self.base_dir})")

    # ── file paths ──────────────────────────────────────────────

    def _dense_path(self, source: str) -> Path:
        return self.dense_dir / f"{source}.parquet"

    def _sparse_path(self, source: str) -> Path:
        return self.sparse_dir / f"{source}.parquet"

    # ── write ───────────────────────────────────────────────────

    def save_dense(self, table: pa.Table, source: str) -> None:
        path = self._dense_path(source)
        pq.write_table(table, path, compression="snappy")
        logger.info(f"Saved {table.num_rows} dense embeddings → {path}")

    def save_sparse(self, table: pa.Table, source: str) -> None:
        path = self._sparse_path(source)
        pq.write_table(table, path, compression="snappy")
        logger.info(f"Saved {table.num_rows} sparse embeddings → {path}")

    @contextmanager
    def streaming_writer(
        self,
        source: str,
        kind: str,
        exclude_doc_ids: Set[str] = None,
    ) -> Generator[pq.ParquetWriter, None, None]:
        """Context manager for streaming row-group writes to a source Parquet.

        Writes go to a temp file; on successful close the temp atomically
        replaces the original.  If *exclude_doc_ids* is given, existing rows
        whose doc_id is NOT in the set are preserved (copied as the first
        row group).
        """
        schema = _DENSE_SCHEMA if kind == "dense" else _SPARSE_SCHEMA
        directory = self.dense_dir if kind == "dense" else self.sparse_dir
        path = directory / f"{source}.parquet"
        tmp_path = path.with_suffix(".parquet.tmp")

        writer = pq.ParquetWriter(str(tmp_path), schema, compression="snappy")

        if exclude_doc_ids and path.exists():
            existing = pq.read_table(path)
            mask = pa.compute.invert(
                pa.compute.is_in(
                    existing.column("doc_id"),
                    value_set=pa.array(list(exclude_doc_ids)),
                )
            )
            kept = existing.filter(mask)
            if kept.num_rows > 0:
                writer.write_table(kept)
                logger.debug(f"Preserved {kept.num_rows} existing rows in {path.name}")
            del existing, mask, kept

        try:
            yield writer
        except Exception:
            writer.close()
            tmp_path.unlink(missing_ok=True)
            raise
        else:
            writer.close()
            tmp_path.rename(path)

    # ── read ────────────────────────────────────────────────────

    def load_dense(self, source: str = None) -> Optional[pa.Table]:
        return self._load(self.dense_dir, source)

    def load_sparse(self, source: str = None) -> Optional[pa.Table]:
        return self._load(self.sparse_dir, source)

    def _load(self, directory: Path, source: str = None) -> Optional[pa.Table]:
        if source:
            path = directory / f"{source}.parquet"
            if not path.exists():
                return None
            return pq.read_table(path)

        parts = sorted(directory.glob("*.parquet"))
        if not parts:
            return None
        tables = [pq.read_table(p) for p in parts]
        return pa.concat_tables(tables, promote_options="default")

    # ── incremental helpers ─────────────────────────────────────

    def get_embedded_doc_ids(self, kind: str = "dense") -> Set[str]:
        """Return all doc_ids present in stored embeddings."""
        directory = self.dense_dir if kind == "dense" else self.sparse_dir
        table = self._load(directory)
        if table is None:
            return set()
        return set(table.column("doc_id").to_pylist())

    def remove_by_doc_ids(
        self, doc_ids: Set[str], source: str, kind: str = "dense",
    ) -> None:
        """Remove rows for given doc_ids from a source partition."""
        directory = self.dense_dir if kind == "dense" else self.sparse_dir
        path = directory / f"{source}.parquet"
        if not path.exists():
            return

        table = pq.read_table(path)
        mask = pa.compute.invert(
            pa.compute.is_in(table.column("doc_id"), value_set=pa.array(list(doc_ids)))
        )
        filtered = table.filter(mask)
        pq.write_table(filtered, path, compression="snappy")
        removed = table.num_rows - filtered.num_rows
        logger.info(f"Removed {removed} chunks for {len(doc_ids)} docs from {path}")

    def append_to_source(
        self, new_table: pa.Table, source: str, kind: str = "dense",
    ) -> None:
        """Append rows to an existing source partition (or create it)."""
        directory = self.dense_dir if kind == "dense" else self.sparse_dir
        path = directory / f"{source}.parquet"

        if path.exists():
            existing = pq.read_table(path)
            merged = pa.concat_tables([existing, new_table], promote_options="default")
        else:
            merged = new_table

        pq.write_table(merged, path, compression="snappy")
        logger.info(
            f"Appended {new_table.num_rows} chunks → {path} "
            f"(total: {merged.num_rows})"
        )

    # ── stats ───────────────────────────────────────────────────

    def stats(self) -> Dict:
        result: Dict = {"dense": {}, "sparse": {}}
        for kind, directory in [("dense", self.dense_dir), ("sparse", self.sparse_dir)]:
            for path in sorted(directory.glob("*.parquet")):
                meta = pq.read_metadata(path)
                result[kind][path.stem] = {
                    "rows": meta.num_rows,
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                }
        return result
