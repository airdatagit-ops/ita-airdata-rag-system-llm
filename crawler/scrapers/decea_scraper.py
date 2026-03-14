"""
DECEA Publications Scraper for Aviation RAG System.

Fetches documents from publicacoes.decea.mil.br using HTTP requests.
Supports parallel download and text extraction via ThreadPoolExecutor.

Usage:
    from parsers.decea_scraper import DECEAScraper

    scraper = DECEAScraper()
    documents = scraper.search(doc_types=["ICA"], limit=50)
    results = scraper.fetch_all(documents, workers=8)
"""

import re
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from loguru import logger

_PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "decea"
ORIGINALS_DIR = _PROJECT_ROOT / "data" / "originals"

_DOC_TYPE_RE = re.compile(r'^([A-Za-z]+(?:-[A-Za-z]+)*)(\d.*)$')
_PDF_URL_RE = re.compile(r'https?://[^"\'>\s]+\.pdf[^"\'>\s]*')
_HEADER_VALUES = {'NÚMERO', 'NUMERO', 'TÍTULO', 'TITLE', 'NUMBER'}

_ORIGINALS_FOLDER = {
    "ica": "ica", "mca": "ica", "pca": "ica", "dca": "ica",
    "tca": "ica", "circea": "ica", "nsca": "ica", "fca": "ica",
}


class DECEAScraper:
    """Scraper for DECEA Publications Portal (publicacoes.decea.mil.br)."""

    BASE_URL = "https://publicacoes.decea.mil.br"
    INDEX_URL = f"{BASE_URL}/publicacao/indice"
    PUBLICATION_URL = f"{BASE_URL}/publicacao"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("DECEAScraper initialized (HTTP mode)")

    # ── public API ──────────────────────────────────────────────

    def get_publications_index(self) -> List[Dict]:
        """Fetch all active publications from the index page (single HTTP call)."""
        logger.info(f"Fetching index: {self.INDEX_URL}")
        resp = self.session.get(self.INDEX_URL, timeout=self.timeout)
        resp.raise_for_status()
        docs = self._parse_index_html(resp.text)
        logger.success(f"Found {len(docs)} publications in index")
        return docs

    def search(
        self,
        doc_types: List[str] = None,
        keywords: List[str] = None,
        limit: int = 100,
        slugs: List[str] = None,
    ) -> List[Dict]:
        """Search for documents, filtering by type, keywords, or specific slugs."""
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
            self._save_original(doc, pdf_bytes, pdf_url)

        text = self._extract_pdf_text(pdf_bytes)
        return text if text and len(text) > 100 else fallback

    def fetch_all(
        self,
        documents: List[Dict],
        workers: int = 8,
        extract_text: bool = True,
        save_original: bool = True,
    ) -> List[Tuple[Dict, Optional[str]]]:
        """Fetch PDF content for multiple documents in parallel."""
        n = len(documents)
        logger.info(f"Fetching {n} documents with {workers} workers")
        results: List[Optional[Tuple[Dict, Optional[str]]]] = [None] * n

        def _process(idx: int, doc: Dict):
            try:
                if extract_text:
                    text = self.get_document_text(doc, save_original)
                else:
                    text = doc.get("description") or doc.get("title", "")
            except Exception as e:
                logger.error(f"Error processing {doc.get('slug')}: {e}")
                text = doc.get("title", "")
            return idx, doc, text

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process, i, d): i for i, d in enumerate(documents)}
            done = 0
            for future in as_completed(futures):
                idx, doc, text = future.result()
                results[idx] = (doc, text)
                done += 1
                if done % 10 == 0 or done == n:
                    logger.info(f"Progress: {done}/{n}")

        return results

    def save_document_json(self, doc: Dict, text: str) -> Path:
        """Save document as JSON for the ingestion pipeline."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        slug = doc.get("slug", "unknown")
        file_slug = slug.replace("/", "-")
        path = DATA_DIR / f"{file_slug}.json"

        data = {
            "id": hashlib.md5(slug.encode()).hexdigest(),
            "slug": slug,
            "type": doc.get("type", ""),
            "number": doc.get("number", ""),
            "title": doc.get("title", ""),
            "description": doc.get("title", ""),
            "date_published": doc.get("date_published", ""),
            "source_url": doc.get("source_url", ""),
            "pdf_link": doc.get("pdf_link", ""),
            "content": text,
            "doc_type": doc.get("doc_type", "ica"),
            "scraped_at": datetime.now().isoformat(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Saved JSON: {path}")
        return path

    # ── internal ────────────────────────────────────────────────

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
        """Insert dash between type prefix and number: ICA96-1 → ICA-96-1."""
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

    def _save_original(self, doc: Dict, content: bytes, url: str):
        try:
            folder_name = _ORIGINALS_FOLDER.get(doc.get("doc_type", ""), "outros")
            folder = ORIGINALS_DIR / folder_name
            folder.mkdir(parents=True, exist_ok=True)

            slug = doc.get("slug", "unknown").replace("/", "-")
            (folder / f"{slug}.pdf").write_bytes(content)

            meta = {
                "slug": doc.get("slug"),
                "type": doc.get("type"),
                "title": doc.get("title"),
                "source_url": doc.get("source_url"),
                "pdf_url": url,
                "downloaded_at": datetime.now().isoformat(),
            }
            with open(folder / f"{slug}.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving original: {e}")

    def _extract_pdf_text(self, pdf_content: bytes) -> Optional[str]:
        """Extract text: PyMuPDF → pdfplumber → OCR fallback."""
        try:
            import fitz
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            if len(text.strip()) > 100:
                return text
        except (ImportError, Exception):
            pass

        try:
            import pdfplumber
            import io
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            if len(text.strip()) > 100:
                return text
        except (ImportError, Exception):
            pass

        try:
            from parsers.ocr_processor import OCRProcessor  # noqa: still in parsers/
            ocr = OCRProcessor(use_gpu=False)
            text = ocr.extract_text_from_pdf_bytes(pdf_content, resolution=300)
            if text and len(text.strip()) > 50:
                return text
        except (ImportError, Exception):
            pass

        return None
