"""
Phase 2 – Embed documents stored in the DocumentStore.

Reads documents from SQLite, chunks them, generates dense and/or sparse
embeddings, and persists the results as Parquet files.  Only documents
whose content has changed since the last embedding are re-processed.

Documents are processed in streaming batches (--embed-batch, default 500)
so that memory usage stays bounded regardless of corpus size and completed
batches are logged immediately for crash resilience.

Usage:
    python -m scripts.embed --mode dense
    python -m scripts.embed --mode hybrid --force
    make embed
    make embed MODE=hybrid FORCE=1
"""

import argparse
import hashlib
import json
import uuid
from contextlib import ExitStack
from typing import Dict, List

import numpy as np
import pyarrow as pa
from loguru import logger
from tqdm import tqdm

from config import config
from pipeline.chunking import get_chunker
from pipeline.document_store import DocumentStore
from pipeline.embedding_store import EmbeddingStore
from pipeline.text_cleaner import TextCleaner
from pipeline.quality import QualityValidator

_DEFAULT_EMBED_BATCH = 500


def _generate_chunk_id(regulation_id: str, text: str) -> str:
    """Deterministic UUID from regulation_id + text prefix."""
    raw = f"{regulation_id}|{text[:100]}"
    return str(uuid.UUID(bytes=hashlib.md5(raw.encode()).digest()))


def _parse_metadata(doc: Dict) -> Dict:
    """Parse metadata from a DocumentStore row, handling JSON strings."""
    meta = doc.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            meta = {}
    return meta


def _build_article(doc: Dict, meta: Dict, cleaned_text: str) -> Dict:
    """Convert a DocumentStore row into the article dict expected by chunkers."""
    doc_type = doc.get("doc_type") or meta.get("type") or meta.get("doc_type") or ""

    return {
        "regulation_id": doc["doc_id"],
        "title": doc.get("title") or meta.get("title", ""),
        "text": cleaned_text,
        "effective_date": doc.get("effective_date") or meta.get("date") or meta.get("date_published"),
        "expiry_date": doc.get("expiry_date"),
        "status": doc.get("status") or meta.get("status", "active"),
        "metadata": {
            "title": doc.get("title") or meta.get("title"),
            "url": doc.get("url") or meta.get("source_url"),
            "urn": doc.get("urn") or meta.get("urn"),
            "type": doc_type,
            "number": meta.get("number"),
            "description": meta.get("description"),
            "source": doc.get("source"),
            "chunk_type": "document",
        },
    }


def _chunk_document(
    doc: Dict,
    text_cleaner: TextCleaner,
    quality_validator: QualityValidator,
) -> List[Dict]:
    """Validate, clean, chunk a single document. Returns list of chunk dicts."""
    content = doc["content"]
    doc_id = doc["doc_id"]

    quality = quality_validator.validate(content, doc_id=doc_id)
    if not quality.is_acceptable:
        logger.warning(f"Rejected {doc_id} [{quality.level.value}]: {quality.reject_reason}")
        return []

    content, _ = text_cleaner.clean(content, doc_id=doc_id)
    if len(content) < 50:
        logger.warning(f"Skipping {doc_id}: content too short after cleaning")
        return []

    meta = _parse_metadata(doc)
    article = _build_article(doc, meta, content)
    doc_type = doc.get("doc_type") or meta.get("type") or ""
    chunker = get_chunker(doc_type)
    return chunker.chunk(article)


def _chunks_to_dense_table(
    chunks: List[Dict],
    dense_embeddings: np.ndarray,
    content_hashes: List[str],
) -> pa.Table:
    """Build a PyArrow table for dense embeddings."""
    rows = {
        "chunk_id": [],
        "doc_id": [],
        "chunk_index": [],
        "text": [],
        "dense_vector": [],
        "content_hash": [],
        "metadata": [],
    }
    for i, chunk in enumerate(chunks):
        chunk_id = _generate_chunk_id(
            chunk.get("regulation_id", ""), chunk["text"]
        )
        rows["chunk_id"].append(chunk_id)
        rows["doc_id"].append(
            chunk.get("regulation_id", "").split("-chunk-")[0].split("-art")[0]
        )
        rows["chunk_index"].append(chunk.get("chunk_index", i))
        rows["text"].append(chunk["text"])
        rows["dense_vector"].append(dense_embeddings[i].tolist())
        rows["content_hash"].append(content_hashes[i])

        meta = {k: v for k, v in chunk.items() if k not in ("text",)}
        rows["metadata"].append(json.dumps(meta, ensure_ascii=False, default=str))

    return pa.table(rows)


def _chunks_to_sparse_table(
    chunks: List[Dict],
    sparse_vectors: list,
    content_hashes: List[str],
) -> pa.Table:
    rows = {
        "chunk_id": [],
        "doc_id": [],
        "chunk_index": [],
        "text": [],
        "sparse_indices": [],
        "sparse_values": [],
        "content_hash": [],
        "metadata": [],
    }
    for i, chunk in enumerate(chunks):
        chunk_id = _generate_chunk_id(
            chunk.get("regulation_id", ""), chunk["text"]
        )
        sv = sparse_vectors[i]
        rows["chunk_id"].append(chunk_id)
        rows["doc_id"].append(
            chunk.get("regulation_id", "").split("-chunk-")[0].split("-art")[0]
        )
        rows["chunk_index"].append(chunk.get("chunk_index", i))
        rows["text"].append(chunk["text"])
        rows["sparse_indices"].append(
            sv.indices if isinstance(sv.indices, list) else sv.indices.tolist()
        )
        rows["sparse_values"].append(
            sv.values if isinstance(sv.values, list) else sv.values.tolist()
        )
        rows["content_hash"].append(content_hashes[i])

        meta = {k: v for k, v in chunk.items() if k not in ("text",)}
        rows["metadata"].append(json.dumps(meta, ensure_ascii=False, default=str))

    return pa.table(rows)


# ── main logic ───────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    mode = args.mode or config.DEFAULT_EMBEDDING_MODE
    do_dense = mode in ("dense", "hybrid")
    do_sparse = mode in ("sparse", "hybrid")
    embed_batch = args.embed_batch

    doc_store = DocumentStore()
    emb_store = EmbeddingStore()
    text_cleaner = TextCleaner()
    quality_validator = QualityValidator()

    # Load models once (expensive; reused across all sources and batches)
    dense_model = None
    sparse_model = None
    if do_dense:
        from models.embeddings import EmbeddingModel
        dense_model = EmbeddingModel()
    if do_sparse:
        from models.embeddings import SparseEncoder
        sparse_model = SparseEncoder()

    sources = [s.strip() for s in args.source.split(",")] if args.source != "all" else None

    if args.force:
        docs = doc_store.get_all_documents()
        if sources:
            docs = [d for d in docs if d["source"] in sources]
        logger.info(f"Force mode: processing all {len(docs)} documents")
    else:
        docs = doc_store.get_docs_needing_embedding(mode=mode)
        if sources:
            docs = [d for d in docs if d["source"] in sources]
        logger.info(f"Incremental mode: {len(docs)} documents need embedding")

    if not docs:
        logger.success("All documents are up-to-date. Nothing to embed.")
        return 0

    by_source: Dict[str, List[Dict]] = {}
    for doc in docs:
        by_source.setdefault(doc["source"], []).append(doc)

    model_name = config.EMBEDDING_MODEL

    for source, source_docs in by_source.items():
        n_batches = (len(source_docs) + embed_batch - 1) // embed_batch
        logger.info(
            f"Processing {len(source_docs)} documents from source={source} "
            f"({n_batches} batch{'es' if n_batches != 1 else ''} of {embed_batch})"
        )

        all_doc_ids = {d["doc_id"] for d in source_docs}
        exclude_ids = None if args.force else all_doc_ids

        with ExitStack() as stack:
            dense_writer = None
            sparse_writer = None

            if do_dense:
                dense_writer = stack.enter_context(
                    emb_store.streaming_writer(source, "dense", exclude_doc_ids=exclude_ids)
                )
            if do_sparse:
                sparse_writer = stack.enter_context(
                    emb_store.streaming_writer(source, "sparse", exclude_doc_ids=exclude_ids)
                )

            total_chunks = 0

            for batch_idx in range(n_batches):
                batch_start = batch_idx * embed_batch
                batch_docs = source_docs[batch_start:batch_start + embed_batch]

                # ── chunk ────────────────────────────────────────────
                batch_chunks: List[Dict] = []
                batch_hashes: List[str] = []
                batch_doc_info: Dict[str, Dict] = {}

                for doc in tqdm(
                    batch_docs,
                    desc=f"Chunking [{source}] {batch_idx + 1}/{n_batches}",
                    leave=False,
                ):
                    chunks = _chunk_document(doc, text_cleaner, quality_validator)
                    if not chunks:
                        continue
                    batch_chunks.extend(chunks)
                    batch_hashes.extend([doc["content_hash"]] * len(chunks))
                    batch_doc_info[doc["doc_id"]] = {
                        "content_hash": doc["content_hash"],
                        "num_chunks": len(chunks),
                    }

                if not batch_chunks:
                    continue

                total_chunks += len(batch_chunks)
                logger.info(
                    f"[{source}] batch {batch_idx + 1}/{n_batches}: "
                    f"{len(batch_chunks)} chunks from {len(batch_doc_info)} docs"
                )

                # ── encode & write ────────────────────────────────────
                texts = [c["text"] for c in batch_chunks]

                if dense_writer is not None:
                    dense_embs = dense_model.encode(
                        texts, batch_size=args.batch_size, show_progress=True,
                    )
                    table = _chunks_to_dense_table(batch_chunks, dense_embs, batch_hashes)
                    dense_writer.write_table(table)
                    del dense_embs, table

                if sparse_writer is not None:
                    sparse_vecs = sparse_model.encode(texts)
                    table = _chunks_to_sparse_table(batch_chunks, sparse_vecs, batch_hashes)
                    sparse_writer.write_table(table)
                    del sparse_vecs, table

                # ── log per-batch for crash resilience ────────────────
                for doc_id, info in batch_doc_info.items():
                    doc_store.log_embedding(
                        doc_id=doc_id,
                        content_hash=info["content_hash"],
                        embedding_mode=mode,
                        model_name=model_name,
                        num_chunks=info["num_chunks"],
                    )

                del batch_chunks, batch_hashes, batch_doc_info

            logger.info(f"Total chunks for {source}: {total_chunks}")

    logger.success(
        f"Embedding complete: {len(docs)} documents, "
        f"mode={mode}, store stats={emb_store.stats()}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 – Generate embeddings")
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["dense", "sparse", "hybrid"],
        help="Embedding mode (default: from config DEFAULT_EMBEDDING_MODE)",
    )
    parser.add_argument(
        "--source", type=str, default="all",
        help="Source to embed (lexml, decea, or 'all')",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-embed all documents (ignore cached embeddings)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for dense model encoding (default: from config)",
    )
    parser.add_argument(
        "--embed-batch", type=int, default=_DEFAULT_EMBED_BATCH,
        help=f"Documents per streaming batch (default: {_DEFAULT_EMBED_BATCH})",
    )
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = config.EMBEDDING_BATCH_SIZE
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
