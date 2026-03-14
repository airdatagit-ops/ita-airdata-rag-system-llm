"""End-to-end ingestion pipeline with support for dense + sparse vectors."""

import json
import hashlib
import uuid
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from models.embeddings import EmbeddingModel, SparseEncoder
from parsers import LexMLParser, PDFParser
from pipeline.chunking import ArticleChunker, ICAChunker, get_chunker
from pipeline.text_cleaner import TextCleaner
from pipeline.quality import QualityValidator, QualityLevel
from database.qdrant_manager import QdrantManager
from config import config


def generate_point_id(text: str) -> str:
    """Generate a deterministic UUID point ID from text."""
    hash_bytes = hashlib.md5(text.encode('utf-8')).digest()
    return str(uuid.UUID(bytes=hash_bytes))


class IngestionPipeline:
    """End-to-end pipeline for ingesting documents."""

    def __init__(self, enable_cleaning: bool = True, enable_embedding: bool = True):
        self.text_cleaner = TextCleaner()
        self.quality_validator = QualityValidator()
        self.enable_cleaning = enable_cleaning
        self.enable_embedding = enable_embedding

        if enable_embedding:
            self.dense_model = EmbeddingModel() if config.SEARCH_DENSE_ENABLED else None
            self.sparse_model = SparseEncoder() if config.SEARCH_SPARSE_ENABLED else None
            self.db = QdrantManager()

        self.lexml_parser = LexMLParser()
        self.pdf_parser = PDFParser()
        self.chunker = ArticleChunker()
        self.ica_chunker = ICAChunker()

        modes = []
        if enable_embedding:
            if config.SEARCH_DENSE_ENABLED:
                modes.append("dense")
            if config.SEARCH_SPARSE_ENABLED:
                modes.append("sparse")
        logger.info(
            f"IngestionPipeline initialized "
            f"(cleaning={enable_cleaning}, embedding={enable_embedding}"
            f"{', vectors=' + '+'.join(modes) if modes else ''})"
        )

    def _encode_chunks(self, texts: List[str]):
        """Encode texts into dense and/or sparse embeddings based on config."""
        dense_embeddings = None
        sparse_embeddings = None

        if self.dense_model:
            logger.info(f"Encoding {len(texts)} chunks (dense)...")
            dense_embeddings = self.dense_model.encode(texts, show_progress=True)

        if self.sparse_model:
            logger.info(f"Encoding {len(texts)} chunks (sparse)...")
            sparse_embeddings = self.sparse_model.encode(texts)

        return dense_embeddings, sparse_embeddings

    def _build_vector(self, idx: int, dense_embeddings, sparse_embeddings):
        """Build vector for a single point.

        Returns a plain list when only dense is used (backward compat with
        unnamed vectors), or a named dict when sparse is also present.
        """
        if sparse_embeddings is not None:
            vector = {}
            if dense_embeddings is not None:
                vector["dense"] = dense_embeddings[idx].tolist()
            vector["sparse"] = sparse_embeddings[idx]
            return vector
        return dense_embeddings[idx].tolist()

    def _encode_and_upsert(self, chunks: List[Dict], id_fn=None) -> int:
        """Encode chunks, build points, and upsert to Qdrant."""
        texts = [c["text"] for c in chunks]
        dense_embeddings, sparse_embeddings = self._encode_chunks(texts)

        points = []
        for i, chunk in enumerate(chunks):
            if id_fn:
                point_id = id_fn(chunk)
            else:
                point_id = generate_point_id(
                    chunk.get("regulation_id", "") + chunk["text"][:100]
                )
            points.append({
                "id": point_id,
                "vector": self._build_vector(i, dense_embeddings, sparse_embeddings),
                "payload": chunk,
            })

        logger.info(f"Upserting {len(points)} points to Qdrant...")
        self.db.upsert_points(points)
        return len(points)

    def ingest_lexml(self, xml_paths: List[str]) -> int:
        """Ingest LexML XML documents."""
        all_chunks = []

        for xml_path in tqdm(xml_paths, desc="Chunking LexML"):
            try:
                articles = self.lexml_parser.parse_xml(xml_path)
                for article in articles:
                    all_chunks.extend(self.chunker.chunk(article))
            except Exception as e:
                logger.error(f"Error chunking {xml_path}: {e}")

        if not all_chunks:
            logger.warning("No chunks produced from any LexML document")
            return 0

        count = self._encode_and_upsert(all_chunks)
        logger.success(f"Ingested {count} chunks from {len(xml_paths)} documents")
        return count

    def _load_and_chunk(self, json_path: str) -> List[Dict]:
        """Load a JSON document, validate, clean, and chunk it."""
        with open(json_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)

        content = doc.get('content', '')
        if not content or len(content) < 50:
            logger.warning(f"Skipping {json_path}: insufficient content")
            return []

        doc_id = doc.get('urn') or doc.get('slug') or doc.get('url') or Path(json_path).stem
        doc_id = doc_id.replace(':', '_').replace('/', '_').replace('?', '_')[:100]

        quality = self.quality_validator.validate(content, doc_id=doc_id)
        if not quality.is_acceptable:
            logger.warning(
                f"Rejected {doc_id} [{quality.level.value}]: {quality.reject_reason}"
            )
            return []

        if self.enable_cleaning:
            content, stats = self.text_cleaner.clean(content, doc_id=doc_id)
            if len(content) < 50:
                logger.warning(f"Skipping {doc_id}: content too short after cleaning")
                return []

        doc_type = doc.get('type', '') or doc.get('doc_type', '')

        article = {
            "regulation_id": doc_id,
            "title": doc.get('title', ''),
            "text": content,
            "effective_date": doc.get('date') or doc.get('date_published'),
            "status": doc.get('status', 'active'),
            "metadata": {
                "title": doc.get('title'),
                "url": doc.get('url') or doc.get('source_url'),
                "urn": doc.get('urn'),
                "type": doc_type,
                "number": doc.get('number'),
                "description": doc.get('description'),
                "origin": doc.get('origin'),
                "source": doc.get('source', 'web'),
                "chunk_type": "document",
                "quality_level": quality.level.value,
            }
        }

        chunker = get_chunker(doc_type)
        return chunker.chunk(article)

    def ingest_json_documents(self, json_paths: List[str]) -> int:
        """
        Ingest JSON documents (from web scraping).

        Processes in phases:
        1. Read all JSONs, validate, clean, and chunk them
        2. Batch-encode all texts (dense + sparse if enabled)
        3. Batch-upsert all points to Qdrant
        """
        all_chunks = []

        for json_path in tqdm(json_paths, desc="Processing Documents"):
            try:
                chunks = self._load_and_chunk(json_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {json_path}: {e}")

        if not all_chunks:
            logger.warning("No chunks produced from any document")
            return 0


        if not self.enable_embedding:
            logger.info("Embedding disabled — skipping encode and upsert")
            return len(all_chunks)

        count = self._encode_and_upsert(all_chunks)
        logger.success(f"Ingested {count} chunks from {len(json_paths)} JSON documents")
        return count

    def ingest_pdfs(self, pdf_paths: List[str]) -> int:
        """Ingest PDF documents."""
        all_sections = []

        for pdf_path in tqdm(pdf_paths, desc="Parsing PDFs"):
            try:
                all_sections.extend(self.pdf_parser.parse_pdf(pdf_path))
            except Exception as e:
                logger.error(f"Error parsing {pdf_path}: {e}")

        if not all_sections:
            logger.warning("No sections parsed from any PDF")
            return 0

        def pdf_id_fn(section):
            return section.get("regulation_id") or generate_point_id(section["text"][:200])

        count = self._encode_and_upsert(all_sections, id_fn=pdf_id_fn)
        logger.success(f"Ingested {count} sections from {len(pdf_paths)} PDFs")
        return count


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    print("IngestionPipeline ready")
