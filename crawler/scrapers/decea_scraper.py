"""
DECEA Publications Scraper for Aviation RAG System.

Fetches documents from publicacoes.decea.mil.br using HTTP requests.
Sync I/O is wrapped with ``asyncio.to_thread`` to expose an async interface
compatible with :class:`BaseScraper`.

Usage:
    from crawler.scrapers import get_scraper

    scraper = get_scraper("decea")
    docs = await scraper.search(limit=50, doc_types=["ICA"])
    results = await scraper.fetch_all(docs, concurrency=8)
"""

import asyncio
import re
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger

from crawler.scrapers.base import BaseScraper, ScrapedDocument, DEFAULT_USER_AGENT
from crawler.scrapers import register_scraper
from parsers.pdf_parser import extract_text_from_bytes

_DOC_TYPE_RE = re.compile(r'^([A-Za-z]+(?:-[A-Za-z]+)*)(\d.*)$')
_PDF_URL_RE = re.compile(r'https?://[^"\'>\s]+\.pdf[^"\'>\s]*')
_HEADER_VALUES = {'NÚMERO', 'NUMERO', 'TÍTULO', 'TITLE', 'NUMBER'}

_ORIGINALS_FOLDER = {
    "ica": "ica", "mca": "ica", "pca": "ica", "dca": "ica",
    "tca": "ica", "circea": "ica", "nsca": "ica", "fca": "ica",
}


@register_scraper
class DECEAScraper(BaseScraper):
    """Scraper for DECEA Publications Portal (publicacoes.decea.mil.br)."""

    BASE_URL = "https://publicacoes.decea.mil.br"
    INDEX_URL = f"{BASE_URL}/publicacao/indice"
    PUBLICATION_URL = f"{BASE_URL}/publicacao"

    source_name = "decea"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = DEFAULT_USER_AGENT
        logger.info("DECEAScraper initialized (HTTP mode)")

    # ── BaseScraper interface ───────────────────────────────────

    async def search(self, *, limit: int = 100, **kwargs) -> List[Dict]:
        """Search for documents (async wrapper around sync HTTP).

        Keyword Args:
            doc_types: List of document type codes (default ``["ICA"]``).
            keywords: Optional keyword filters.
            slugs: Fetch specific publications by slug.
        """
        return await asyncio.to_thread(
            self._search_sync,
            doc_types=kwargs.get("doc_types"),
            keywords=kwargs.get("keywords"),
            slugs=kwargs.get("slugs"),
            limit=limit,
        )

    async def fetch_document(
        self, doc: Dict, save_original: bool = True
    ) -> Optional[ScrapedDocument]:
        """Fetch PDF and extract text for a single document."""
        return await asyncio.to_thread(self._fetch_document_sync, doc, save_original)

    def make_doc_id(self, doc: Dict) -> str:
        slug = doc.get("slug", "")
        if slug:
            return f"decea_{slug}"
        return super().make_doc_id(doc)

    # ── sync implementation ────────────────────────────────────

    def get_publications_index(self) -> List[Dict]:
        """Fetch all active publications from the index page."""
        logger.info(f"Fetching index: {self.INDEX_URL}")
        resp = self.session.get(self.INDEX_URL, timeout=self.timeout)
        resp.raise_for_status()
        docs = self._parse_index_html(resp.text)
        logger.success(f"Found {len(docs)} publications in index")
        return docs

    def _search_sync(
        self,
        doc_types: List[str] = None,
        keywords: List[str] = None,
        limit: int = 100,
        slugs: List[str] = None,
    ) -> List[Dict]:
        if slugs:
            docs = [d for s in slugs if (d := self._fetch_by_slug(s))]
            return docs[:limit]

        types_upper = {t.upper() for t in (doc_types or ["ICA"])}
        results = []
        for doc in self.get_publications_index():
            if doc["type"] not in types_upper:
                continue
            if keywords and not any(
                k.lower() in doc["title"].lower() for k in keywords
            ):
                continue
            results.append(doc)
            if len(results) >= limit:
                break

        logger.info(f"Filtered to {len(results)} documents (types={types_upper})")
        return results

    def get_pdf_url(self, slug: str) -> Optional[str]:
        """Extract the signed PDF download URL from a publication page."""
        try:
            resp = self.session.get(
                f"{self.PUBLICATION_URL}/{slug}", timeout=self.timeout
            )
            resp.raise_for_status()
            match = _PDF_URL_RE.search(resp.text)
            if match:
                return match.group(0).replace("\\u0026", "&").rstrip("\\")
            logger.warning(f"No PDF link found for {slug}")
            return None
        except requests.RequestException as e:
            logger.warning(f"Failed to get PDF URL for {slug}: {e}")
            return None

    def download_pdf(self, url: str, slug: str = "") -> Optional[bytes]:
        """Download PDF content from a URL."""
        try:
            resp = self.session.get(url, timeout=60)
            content_type = resp.headers.get("Content-Type", "")
            if resp.ok and ("pdf" in content_type.lower() or resp.content[:4] == b"%PDF"):
                logger.debug(f"Downloaded {slug}: {len(resp.content):,} bytes")
                return resp.content
            logger.warning(f"Non-PDF response for {slug}: {resp.status_code}")
            return None
        except requests.RequestException as e:
            logger.error(f"Download failed for {slug}: {e}")
            return None

    def get_document_text(self, doc: Dict, save_original: bool = True) -> Optional[str]:
        """Fetch PDF and extract full text for a single document."""
        slug = doc.get("slug", "")
        fallback = doc.get("description") or doc.get("title")

        pdf_url = self.get_pdf_url(slug)
        if not pdf_url:
            return fallback

        doc["pdf_link"] = pdf_url
        pdf_bytes = self.download_pdf(pdf_url, slug)
        if not pdf_bytes:
            return fallback

        if save_original:
            folder_name = _ORIGINALS_FOLDER.get(doc.get("doc_type", ""), "outros")
            safe_slug = slug.replace("/", "-")
            meta = {
                "slug": slug, "type": doc.get("type"), "title": doc.get("title"),
                "source_url": doc.get("source_url"), "pdf_url": pdf_url,
            }
            self.save_original_file(
                pdf_bytes, folder_name=folder_name, stem=safe_slug,
                extension=".pdf", meta=meta,
            )

        text = extract_text_from_bytes(pdf_bytes)
        return text if text and len(text) > 100 else fallback

    def _fetch_document_sync(
        self, doc: Dict, save_original: bool = True
    ) -> Optional[ScrapedDocument]:
        try:
            content = self.get_document_text(doc, save_original)
            if not content or len(content.strip()) < 50:
                return None

            exclude = {"content"}
            metadata = {k: v for k, v in doc.items() if k not in exclude}

            return ScrapedDocument(
                doc_id=self.make_doc_id(doc),
                source=self.source_name,
                title=doc.get("title", ""),
                content=content,
                metadata=metadata,
                url=doc.get("source_url"),
                doc_type=doc.get("doc_type"),
            )
        except Exception as exc:
            logger.error(f"Error processing {doc.get('slug', '?')}: {exc}")
            return None

    # ── HTML parsing helpers ────────────────────────────────────

    def _parse_index_html(self, html: str) -> List[Dict]:
        soup = BeautifulSoup(html, "html.parser")
        pubs = []

        for tr in soup.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if len(cells) < 4:
                continue

            numero = cells[0].get_text(strip=True)
            if not numero or numero.upper() in _HEADER_VALUES:
                continue

            m = _DOC_TYPE_RE.match(numero)
            if not m:
                continue

            doc_type = m.group(1).upper()
            slug = self._build_slug(numero, doc_type)
            title = cells[1].get_text(strip=True)

            pubs.append({
                "slug": slug,
                "type": doc_type,
                "number": numero,
                "title": title,
                "description": title,
                "date_published": cells[2].get_text(strip=True),
                "origin": cells[3].get_text(strip=True),
                "status": "PUBLISHED",
                "doc_type": doc_type.lower(),
                "source_url": f"{self.PUBLICATION_URL}/{slug}",
                "pdf_link": None,
            })

        return pubs

    @staticmethod
    def _build_slug(numero: str, doc_type: str) -> str:
        """Insert dash between type prefix and number: ICA96-1 -> ICA-96-1."""
        suffix = numero[len(doc_type):]
        if suffix and suffix[0].isdigit():
            return f"{doc_type}-{suffix}".replace(" ", "-")
        return numero.replace(" ", "-")

    def _fetch_by_slug(self, slug: str) -> Optional[Dict]:
        url = f"{self.PUBLICATION_URL}/{slug}"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if not resp.ok or "não conseguimos encontrar" in resp.text.lower():
                return None

            soup = BeautifulSoup(resp.text, "html.parser")
            title_tag = soup.find("title")
            title = (
                title_tag.text.replace(" | Publicações DECEA", "").strip()
                if title_tag
                else slug
            )

            m = _DOC_TYPE_RE.match(slug)
            doc_type = m.group(1).upper() if m else "ICA"

            return {
                "slug": slug, "type": doc_type, "number": slug,
                "title": title, "description": title,
                "date_published": "", "origin": "",
                "status": "PUBLISHED", "doc_type": doc_type.lower(),
                "source_url": url, "pdf_link": None,
            }
        except requests.RequestException:
            return None
