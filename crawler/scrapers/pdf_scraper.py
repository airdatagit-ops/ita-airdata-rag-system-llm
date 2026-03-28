"""
Local PDF Scraper for Aviation RAG System.

Scans a directory for PDF files, extracts text via PDFParser, and returns
ScrapedDocument instances.  File I/O and parsing are CPU-bound and run
inside ``asyncio.to_thread``.

Usage:
    from crawler.scrapers import get_scraper

    scraper = get_scraper("pdf", pdf_dir="./data/pdfs")
    docs = await scraper.search(limit=100)
    results = await scraper.fetch_all(docs, concurrency=4)
"""

import asyncio
import re
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from crawler.scrapers.base import BaseScraper, ScrapedDocument
from crawler.scrapers import register_scraper


@register_scraper
class PDFScraper(BaseScraper):
    """Scraper for local PDF files."""

    source_name = "pdf"

    def __init__(self, pdf_dir: str = "./data/pdfs", recursive: bool = True, **_kwargs):
        self.pdf_dir = Path(pdf_dir)
        self.recursive = recursive

    # ── BaseScraper interface ───────────────────────────────────

    async def search(self, *, limit: int = 100, **kwargs) -> List[Dict]:
        """Return metadata dicts for PDF files found in the directory."""
        pdf_dir = Path(kwargs.get("pdf_dir") or self.pdf_dir)
        recursive = kwargs.get("recursive", self.recursive)

        if not pdf_dir.exists():
            logger.warning(f"[pdf] Directory not found: {pdf_dir}")
            return []

        pattern = "**/*.pdf" if recursive else "*.pdf"
        paths = sorted(pdf_dir.glob(pattern))[:limit]

        if not paths:
            logger.warning(f"[pdf] No PDF files found in {pdf_dir}")
            return []

        logger.info(f"[pdf] Found {len(paths)} PDF files in {pdf_dir}")
        return [
            {
                "path": str(p),
                "filename": p.name,
                "stem": p.stem,
                "title": p.stem,
            }
            for p in paths
        ]

    async def fetch_document(
        self, doc: Dict, save_original: bool = True
    ) -> Optional[ScrapedDocument]:
        """Parse a single PDF and return a ScrapedDocument."""
        return await asyncio.to_thread(self._parse_pdf_sync, doc)

    def make_doc_id(self, doc: Dict) -> str:
        stem = doc.get("stem", doc.get("title", "unknown"))
        return f"pdf_{re.sub(r'[^a-zA-Z0-9._-]', '_', stem)}"

    # ── sync implementation ────────────────────────────────────

    def _parse_pdf_sync(self, doc: Dict) -> Optional[ScrapedDocument]:
        from parsers.pdf_parser import PDFParser

        pdf_path = Path(doc["path"])
        try:
            parser = PDFParser()
            sections = parser.parse_pdf(str(pdf_path))
            if not sections:
                logger.warning(f"[pdf] No sections extracted from {pdf_path.name}")
                return None

            content = "\n\n".join(s.get("text", "") for s in sections)
            if len(content.strip()) < 50:
                logger.warning(f"[pdf] Skipping {pdf_path.name}: content too short")
                return None

            title = sections[0].get("title") or pdf_path.stem

            return ScrapedDocument(
                doc_id=self.make_doc_id(doc),
                source=self.source_name,
                title=title,
                content=content,
                metadata={
                    "filename": pdf_path.name,
                    "path": str(pdf_path),
                    "num_sections": len(sections),
                },
                doc_type="PDF",
            )
        except Exception as exc:
            logger.error(f"[pdf] Error processing {pdf_path.name}: {exc}")
            return None
