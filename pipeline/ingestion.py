"""End-to-end ingestion pipeline."""

import json
import hashlib
import uuid
from typing import List, Dict
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from models.embeddings import EmbeddingModel
from parsers import LexMLParser, PDFParser
from pipeline.chunking import ArticleChunker, ICAChunker, get_chunker
from pipeline.text_cleaner import TextCleaner
from pipeline.quality import QualityValidator, QualityLevel
from database.qdrant_manager import QdrantManager


def generate_point_id(text: str) -> str:
    """Generate a valid UUID point ID from text."""
    # Create a deterministic UUID based on the text content
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
            self.embedding_model = EmbeddingModel()
            self.db = QdrantManager()

        self.lexml_parser = LexMLParser()
        self.pdf_parser = PDFParser()
        self.chunker = ArticleChunker()
        self.ica_chunker = ICAChunker()
        logger.info(
            f"IngestionPipeline initialized "
            f"(cleaning={enable_cleaning}, embedding={enable_embedding})"
        )

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

        logger.info(f"Encoding {len(all_chunks)} chunks in batch...")
        texts = [c["text"] for c in all_chunks]
        embeddings = self.embedding_model.encode(texts, show_progress=True)

        points = []
        for chunk, embedding in zip(all_chunks, embeddings):
            point_id = generate_point_id(chunk.get("regulation_id", "") + chunk["text"][:100])
            points.append({
                "id": point_id,
                "vector": embedding.tolist(),
                "payload": chunk
            })

        logger.info(f"Upserting {len(points)} points to Qdrant...")
        self.db.upsert_points(points)

        logger.success(f"Ingested {len(points)} chunks from {len(xml_paths)} documents")
        return len(points)

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

        # Quality gate: reject garbage documents
        quality = self.quality_validator.validate(content, doc_id=doc_id)
        if not quality.is_acceptable:
            logger.warning(
                f"Rejected {doc_id} [{quality.level.value}]: {quality.reject_reason}"
            )
            return []

        # Clean text if enabled
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
        2. Batch-encode all texts in a single model call (if embedding enabled)
        3. Batch-upsert all points to Qdrant (if embedding enabled)
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

        logger.info(f"Produced {len(all_chunks)} chunks from {len(json_paths)} documents")

        if not self.enable_embedding:
            logger.info("Embedding disabled — skipping encode and upsert")
            return len(all_chunks)

        logger.info(f"Encoding {len(all_chunks)} chunks in batch...")
        texts = [c["text"] for c in all_chunks]
        embeddings = self.embedding_model.encode(texts, show_progress=True)

        points = []
        for chunk, embedding in zip(all_chunks, embeddings):
            point_id = generate_point_id(chunk.get("regulation_id", "") + chunk["text"][:100])
            points.append({
                "id": point_id,
                "vector": embedding.tolist(),
                "payload": chunk
            })

        logger.info(f"Upserting {len(points)} points to Qdrant...")
        self.db.upsert_points(points)

        logger.success(f"Ingested {len(points)} chunks from {len(json_paths)} JSON documents")
        return len(points)

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

        logger.info(f"Encoding {len(all_sections)} sections in batch...")
        texts = [s["text"] for s in all_sections]
        embeddings = self.embedding_model.encode(texts, show_progress=True)

        points = []
        for section, embedding in zip(all_sections, embeddings):
            points.append({
                "id": section.get("regulation_id") or generate_point_id(section["text"][:200]),
                "vector": embedding.tolist(),
                "payload": section
            })

        logger.info(f"Upserting {len(points)} points to Qdrant...")
        self.db.upsert_points(points)

        logger.success(f"Ingested {len(points)} sections from {len(pdf_paths)} PDFs")
        return len(points)


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    print("IngestionPipeline ready")
