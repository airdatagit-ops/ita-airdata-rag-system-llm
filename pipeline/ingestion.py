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
        total_chunks = 0

        for xml_path in tqdm(xml_paths, desc="Ingesting LexML"):
            try:
                # Parse
                articles = self.lexml_parser.parse_xml(xml_path)

                # Chunk
                chunks = []
                for article in articles:
                    chunks.extend(self.chunker.chunk(article))

                # Embed
                texts = [c["text"] for c in chunks]
                embeddings = self.embedding_model.encode(texts, show_progress=False)

                # Prepare points
                points = []
                for chunk, embedding in zip(chunks, embeddings):
                    # Generate valid UUID for point ID
                    point_id = generate_point_id(chunk.get("regulation_id", "") + chunk["text"][:100])
                    points.append({
                        "id": point_id,
                        "vector": embedding.tolist(),
                        "payload": chunk
                    })

                # Upload
                self.db.upsert_points(points)
                total_chunks += len(points)

            except Exception as e:
                logger.error(f"Error ingesting {xml_path}: {e}")

        logger.success(f"Ingested {total_chunks} chunks from {len(xml_paths)} documents")
        return total_chunks

    def ingest_json_documents(self, json_paths: List[str]) -> int:
        """
        Ingest JSON documents (from web scraping).
        
        Each JSON file should have:
        - 'content': The main text content
        - 'title': Document title
        - 'url' or 'urn': Document identifier
        - Other metadata fields
        
        Uses ICAChunker for DECEA documents (ICA, MCA, PCA, etc.)
        and ArticleChunker for other documents.
        """
        total_chunks = 0

        for json_path in tqdm(json_paths, desc="Ingesting Documents"):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)

                content = doc.get('content', '')
                if not content or len(content) < 50:
                    logger.warning(f"Skipping {json_path}: insufficient content")
                    continue

                # Create article-like structure for chunking
                doc_id = doc.get('urn') or doc.get('slug') or doc.get('url') or Path(json_path).stem
                # Clean doc_id to be a valid ID
                doc_id = doc_id.replace(':', '_').replace('/', '_').replace('?', '_')[:100]
                
                # Determine document type for chunking strategy
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

                # Select appropriate chunker based on document type
                chunker = get_chunker(doc_type)
                chunks = chunker.chunk(article)

                if not chunks:
                    continue

                # Embed
                texts = [c["text"] for c in chunks]
                embeddings = self.embedding_model.encode(texts, show_progress=False)

                # Prepare points with valid UUIDs
                points = []
                for chunk, embedding in zip(chunks, embeddings):
                    # Generate valid UUID for point ID
                    point_id = generate_point_id(chunk.get("regulation_id", "") + chunk["text"][:100])
                    points.append({
                        "id": point_id,
                        "vector": embedding.tolist(),
                        "payload": chunk
                    })

                # Upload
                self.db.upsert_points(points)
                total_chunks += len(points)
                logger.debug(f"Ingested {len(points)} chunks from {json_path}")

            except Exception as e:
                logger.error(f"Error ingesting {json_path}: {e}")

        logger.success(f"Ingested {total_chunks} chunks from {len(json_paths)} JSON documents")
        return total_chunks

    def ingest_pdfs(self, pdf_paths: List[str]) -> int:
        """Ingest PDF documents."""
        total_sections = 0

        for pdf_path in tqdm(pdf_paths, desc="Ingesting PDFs"):
            try:
                sections = self.pdf_parser.parse_pdf(pdf_path)
                texts = [s["text"] for s in sections]
                embeddings = self.embedding_model.encode(texts, show_progress=False)

                points = []
                for section, embedding in zip(sections, embeddings):
                    points.append({
                        "id": section.get("regulation_id"),
                        "vector": embedding.tolist(),
                        "payload": section
                    })

                self.db.upsert_points(points)
                total_sections += len(points)

            except Exception as e:
                logger.error(f"Error ingesting {pdf_path}: {e}")

        logger.success(f"Ingested {total_sections} sections from {len(pdf_paths)} PDFs")
        return total_sections


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    print("IngestionPipeline ready")
