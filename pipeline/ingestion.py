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
from database.qdrant_manager import QdrantManager


def generate_point_id(text: str) -> str:
    """Generate a valid UUID point ID from text."""
    # Create a deterministic UUID based on the text content
    hash_bytes = hashlib.md5(text.encode('utf-8')).digest()
    return str(uuid.UUID(bytes=hash_bytes))


class IngestionPipeline:
    """End-to-end pipeline for ingesting documents."""

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.lexml_parser = LexMLParser()
        self.pdf_parser = PDFParser()
        self.chunker = ArticleChunker()
        self.ica_chunker = ICAChunker()  # Specialized chunker for ICAs
        self.db = QdrantManager()
        logger.info("IngestionPipeline initialized")

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
        """Load a JSON document and return its chunks (no embedding yet)."""
        with open(json_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)

        content = doc.get('content', '')
        if not content or len(content) < 50:
            logger.warning(f"Skipping {json_path}: insufficient content")
            return []

        doc_id = doc.get('urn') or doc.get('slug') or doc.get('url') or Path(json_path).stem
        doc_id = doc_id.replace(':', '_').replace('/', '_').replace('?', '_')[:100]
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
                "chunk_type": "document"
            }
        }

        chunker = get_chunker(doc_type)
        return chunker.chunk(article)

    def ingest_json_documents(self, json_paths: List[str]) -> int:
        """
        Ingest JSON documents (from web scraping).

        Processes in three phases for performance:
        1. Read all JSONs and chunk them
        2. Batch-encode all texts in a single model call
        3. Batch-upsert all points to Qdrant
        """
        all_chunks = []

        for json_path in tqdm(json_paths, desc="Chunking Documents"):
            try:
                chunks = self._load_and_chunk(json_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error chunking {json_path}: {e}")

        if not all_chunks:
            logger.warning("No chunks produced from any document")
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
