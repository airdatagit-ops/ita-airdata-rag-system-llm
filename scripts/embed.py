"""
Phase 2 – Embed documents stored in the DocumentStore.

Reads documents from SQLite, chunks them, generates dense and/or sparse
embeddings, and persists the results as Parquet files.  Only documents
whose content has changed since the last embedding are re-processed.

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
from typing import Dict, List, Set

import numpy as np
import pyarrow as pa
from loguru import logger
from tqdm import tqdm

from config import config
from pipeline.chunking import get_chunker
from pipeline.document_store import DocumentStore
from pipeline.embedding_store import EmbeddingStore
from pipeline.text_cleaner import TextCleaner
from pipeline.quality import QualityValidator, QualityLevel


def _generate_chunk_id(regulation_id: str, text: str) -> str:
    """Deterministic UUID from regulation_id + text prefix."""
    raw = f"{regulation_id}|{text[:100]}"
    return str(uuid.UUID(bytes=hashlib.md5(raw.encode()).digest()))


def _build_article(doc: Dict) -> Dict:
    """Convert a DocumentStore row into the article dict expected by chunkers."""
    content = doc["content"]
    doc_id = doc["doc_id"]
    meta = doc.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            meta = {}

    doc_type = doc.get("doc_type") or meta.get("type") or meta.get("doc_type") or ""

    return {
        "regulation_id": doc_id,
        "title": doc.get("title") or meta.get("title", ""),
        "text": content,
        "effective_date": meta.get("date") or meta.get("date_published"),
        "status": meta.get("status", "active"),
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

    article = _build_article(doc)
    article["text"] = content

    meta = doc.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            meta = {}

    doc_type = doc.get("doc_type") or meta.get("type") or ""
    chunker = get_chunker(doc_type)
    return chunker.chunk(article)


def _embed_dense(
    chunks: List[Dict],
    batch_size: int,
) -> np.ndarray:
    from models.embeddings import EmbeddingModel

    model = EmbeddingModel()
    texts = [c["text"] for c in chunks]
    logger.info(f"Generating dense embeddings for {len(texts)} chunks …")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress=True)
    return embeddings


def _embed_sparse(chunks: List[Dict]) -> list:
    from models.embeddings import SparseEncoder

    encoder = SparseEncoder()
    texts = [c["text"] for c in chunks]
    logger.info(f"Generating sparse embeddings for {len(texts)} chunks …")
    return encoder.encode(texts)


def _chunks_to_dense_table(
    chunks: List[Dict],
    dense_embeddings: np.ndarray,
    content_hash: str,
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
        rows["content_hash"].append(content_hash)

        meta = {k: v for k, v in chunk.items() if k not in ("text",)}
        rows["metadata"].append(json.dumps(meta, ensure_ascii=False, default=str))

    return pa.table(rows)


def _chunks_to_sparse_table(
    chunks: List[Dict],
    sparse_vectors: list,
    content_hash: str,
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
        rows["content_hash"].append(content_hash)

        meta = {k: v for k, v in chunk.items() if k not in ("text",)}
        rows["metadata"].append(json.dumps(meta, ensure_ascii=False, default=str))

    return pa.table(rows)


# ── main logic ───────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    mode = args.mode or config.DEFAULT_EMBEDDING_MODE
    do_dense = mode in ("dense", "hybrid")
    do_sparse = mode in ("sparse", "hybrid")

    doc_store = DocumentStore()
    emb_store = EmbeddingStore()
    text_cleaner = TextCleaner()
    quality_validator = QualityValidator()

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
        logger.info(f"Processing {len(source_docs)} documents from source={source}")

        if args.force:
            doc_ids_to_remove = {d["doc_id"] for d in source_docs}
            if do_dense:
                emb_store.remove_by_doc_ids(doc_ids_to_remove, source, kind="dense")
            if do_sparse:
                emb_store.remove_by_doc_ids(doc_ids_to_remove, source, kind="sparse")

        all_chunks: List[Dict] = []
        doc_chunk_map: Dict[str, List[int]] = {}
        doc_hashes: Dict[str, str] = {}

        for doc in tqdm(source_docs, desc=f"Chunking [{source}]"):
            chunks = _chunk_document(doc, text_cleaner, quality_validator)
            if not chunks:
                continue
            start = len(all_chunks)
            all_chunks.extend(chunks)
            doc_chunk_map[doc["doc_id"]] = list(range(start, start + len(chunks)))
            doc_hashes[doc["doc_id"]] = doc["content_hash"]

        if not all_chunks:
            logger.warning(f"No chunks produced for source={source}")
            continue

        logger.info(f"Total chunks for {source}: {len(all_chunks)}")

        if do_dense:
            dense_embs = _embed_dense(all_chunks, batch_size=args.batch_size)

            dense_tables = []
            for doc_id, indices in doc_chunk_map.items():
                doc_chunks = [all_chunks[i] for i in indices]
                doc_dense = dense_embs[indices[0]:indices[-1] + 1]
                t = _chunks_to_dense_table(doc_chunks, doc_dense, doc_hashes[doc_id])
                dense_tables.append(t)

            dense_table = pa.concat_tables(dense_tables, promote_options="default")

            if args.force:
                emb_store.save_dense(dense_table, source)
            else:
                doc_ids_to_remove = set(doc_chunk_map.keys())
                emb_store.remove_by_doc_ids(doc_ids_to_remove, source, kind="dense")
                emb_store.append_to_source(dense_table, source, kind="dense")

        if do_sparse:
            sparse_vecs = _embed_sparse(all_chunks)

            sparse_tables = []
            for doc_id, indices in doc_chunk_map.items():
                doc_chunks = [all_chunks[i] for i in indices]
                doc_sparse = [sparse_vecs[i] for i in indices]
                t = _chunks_to_sparse_table(doc_chunks, doc_sparse, doc_hashes[doc_id])
                sparse_tables.append(t)

            sparse_table = pa.concat_tables(sparse_tables, promote_options="default")

            if args.force:
                emb_store.save_sparse(sparse_table, source)
            else:
                doc_ids_to_remove = set(doc_chunk_map.keys())
                emb_store.remove_by_doc_ids(doc_ids_to_remove, source, kind="sparse")
                emb_store.append_to_source(sparse_table, source, kind="sparse")

        for doc_id, indices in doc_chunk_map.items():
            doc_store.log_embedding(
                doc_id=doc_id,
                content_hash=doc_hashes[doc_id],
                embedding_mode=mode,
                model_name=model_name,
                num_chunks=len(indices),
            )

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
        help="Batch size for dense embedding (default: from config)",
    )
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = config.EMBEDDING_BATCH_SIZE
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
