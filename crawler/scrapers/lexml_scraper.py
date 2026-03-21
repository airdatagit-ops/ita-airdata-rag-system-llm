"""
LexML Scraper - Async scraper for Aviation RAG System.

Fetches legal documents from LexML Brasil portal using aiohttp with:
- Parallel downloads (asyncio.gather + Semaphore)
- Token-bucket rate limiting (LEXML_MAX_RATE req/s via env var)
- Exponential backoff retry on transient failures

Usage:
    async with LexMLScraper() as scraper:
        docs = await scraper.search(keywords=["aviação", "ANAC"], limit=100)
        content = await scraper.get_document_text(docs[0])
"""

import asyncio
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote, urljoin

import aiohttp
from aiolimiter import AsyncLimiter
from bs4 import BeautifulSoup
from loguru import logger

from config import config
from parsers.document_tracker import DocumentTracker, get_tracker

_PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "lexml"
ORIGINALS_DIR = _PROJECT_ROOT / "data" / "originals"

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_RETRY_BASE_DELAY = 1.0
_MAX_RETRIES = 3

_DOC_TYPE_FOLDER: Dict[str, str] = {
    "lei": "leis",
    "lei.complementar": "leis",
    "lei.ordinaria": "leis",
    "decreto": "leis",
    "decreto.lei": "leis",
    "ica": "ica",
    "instrucao": "ica",
    "rbac": "rbac",
    "regulamento": "rbac",
    "portaria": "portarias",
    "resolucao": "resolucoes",
    "resolução": "resolucoes",
}


def _originals_folder(doc_type: str) -> str:
    lower = (doc_type or "").lower()
    for key, folder in _DOC_TYPE_FOLDER.items():
        if key in lower:
            return folder
    return "outros"


class LexMLScraper:
    """Async scraper for LexML Brasil web portal (lexml.gov.br)."""

    SEARCH_URL = "https://www.lexml.gov.br/busca/search"
    BASE_URL = "https://www.lexml.gov.br"
    NORMAS_URL = "https://normas.leg.br"

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8",
    }

    def __init__(
        self,
        skip_duplicates: bool = True,
        max_rate: float = None,
        concurrency: int = 10,
        timeout: int = 30,
    ):
        """
        Args:
            skip_duplicates: Skip already-downloaded documents via DocumentTracker.
            max_rate: Max requests per second. Defaults to LEXML_MAX_RATE env var (5).
            concurrency: Max simultaneous TCP connections.
            timeout: Per-request timeout in seconds.
        """
        self.skip_duplicates = skip_duplicates
        self.tracker: Optional[DocumentTracker] = get_tracker() if skip_duplicates else None

        rate = max_rate if max_rate is not None else (config.LEXML_MAX_RATE or 5)
        self._limiter = AsyncLimiter(rate, 1.0)
        self._connector = aiohttp.TCPConnector(limit=concurrency)
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(f"LexMLScraper initialized (max_rate={rate} req/s, concurrency={concurrency})")

    async def __aenter__(self) -> "LexMLScraper":
        self._session = aiohttp.ClientSession(
            headers=self._HEADERS,
            connector=self._connector,
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # ── public API ────────────────────────────────────────────────────────────

    async def search(
        self,
        keywords: List[str] = None,
        doc_type: str = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Search LexML portal and return document metadata up to `limit` results."""
        keywords = keywords or config.lexml_keywords_list
        query = self._build_query(keywords, doc_type)
        logger.info(f"Searching LexML: {query!r} (limit={limit})")

        documents: List[Dict] = []
        offset = 0

        while len(documents) < limit:
            url = self.SEARCH_URL + "?" + query
            if offset > 0:
                url += f";startDoc={offset + 1}"

            html = await self._get_html(url)
            if html is None:
                break

            batch = self._parse_search_results(html)
            if not batch:
                logger.info("No more results")
                break

            documents.extend(batch)
            offset += len(batch)
            logger.info(f"Retrieved {min(len(documents), limit)}/{limit} documents")

        return documents[:limit]

    async def get_document_text(self, doc: Dict, save_original: bool = True) -> Optional[str]:
        """
        Fetch the full text of a document.

        Strategy:
          1. Open LexML page → find Senado or Planalto link
          2. Follow link → extract full law text
          3. Fallback to ementa if no structured text found
        """
        url = doc.get("url") or (
            f"{self.BASE_URL}/urn/{quote(doc['urn'])}" if doc.get("urn") else None
        )
        if not url:
            return doc.get("description")

        html = await self._get_html(url)
        if html is None:
            return doc.get("description")

        source_url = self._find_source_link(html)
        if not source_url:
            logger.warning(f"No source link for: {doc.get('title', url)}")
            return doc.get("description")

        text = await self._fetch_text_from_source(source_url, doc, save_original)
        return text or doc.get("description")

    async def download_document(
        self, doc: Dict, output_path: str, save_original: bool = True
    ) -> bool:
        """Download document content and save as JSON to `output_path`."""
        content = await self.get_document_text(doc, save_original)
        if not content:
            return False

        data = {**doc, "content": content, "source_url": doc.get("url", "")}
        Path(output_path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Saved: {output_path}")
        return True

    def is_duplicate(self, doc: Dict, content: str = None) -> bool:
        if not self.skip_duplicates or not self.tracker:
            return False
        return self.tracker.is_duplicate(doc, content)

    def register_document(self, doc: Dict, content: str = None, file_path: str = None):
        if self.tracker:
            self.tracker.register_document(doc, content, file_path, source="lexml")

    def get_tracker_stats(self) -> Dict:
        return self.tracker.get_stats() if self.tracker else {}

    # ── HTTP with retry ───────────────────────────────────────────────────────

    async def _get_html(self, url: str) -> Optional[str]:
        """GET a URL respecting rate limit, retrying on transient errors."""
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with self._limiter:
                    async with self._session.get(url) as resp:
                        if resp.status == 200:
                            return await resp.text()
                        if resp.status in _RETRYABLE_STATUSES:
                            raise aiohttp.ClientResponseError(
                                resp.request_info, resp.history, status=resp.status
                            )
                        logger.warning(f"HTTP {resp.status}: {url}")
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        f"Attempt {attempt}/{_MAX_RETRIES} failed ({exc}), "
                        f"retrying in {delay:.0f}s: {url}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {_MAX_RETRIES} attempts failed: {url}")
        return None

    # ── text extraction ───────────────────────────────────────────────────────

    async def _fetch_text_from_source(
        self, source_url: str, doc: Dict, save_original: bool
    ) -> Optional[str]:
        html = await self._get_html(source_url)
        if html is None:
            return None

        if save_original:
            self._save_original(doc, html, source_url)

        soup = BeautifulSoup(html, "html.parser")

        if "senado.leg.br" in source_url:
            return await self._extract_senado_text(soup, source_url, doc, save_original)
        if "planalto.gov.br" in source_url:
            return self._extract_planalto_text(soup)
        return self._extract_generic_text(html)

    async def _extract_senado_text(
        self,
        soup: BeautifulSoup,
        base_url: str,
        doc: Dict,
        save_original: bool,
    ) -> Optional[str]:
        """Follow Senado publication link and extract #conteudoPrincipal."""
        pub_url = self._find_senado_publication_link(soup, base_url)
        if not pub_url:
            return None

        html = await self._get_html(pub_url)
        if html is None:
            return None

        if save_original:
            self._save_original(doc, html, pub_url)

        pub_soup = BeautifulSoup(html, "html.parser")
        content = pub_soup.select_one("#conteudoPrincipal")
        if not content:
            return None

        for tag in content.find_all(["nav", "button"]):
            tag.decompose()

        text = content.get_text(separator="\n", strip=True)
        if len(text) > 200:
            logger.info(f"Extracted {len(text):,} chars from Senado")
            return text
        return None

    def _extract_planalto_text(self, soup: BeautifulSoup) -> Optional[str]:
        for selector in ("#textoNorma", ".textoNorma", "#conteudo", ".conteudo", "article"):
            elem = soup.select_one(selector)
            if elem:
                text = elem.get_text(separator="\n", strip=True)
                if len(text) > 200:
                    logger.info(f"Extracted {len(text):,} chars from Planalto")
                    return text
        return None

    def _extract_generic_text(self, html: str) -> Optional[str]:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        for selector in (".texto-norma", ".conteudo-norma", "#texto", "article", "main", "#conteudo"):
            elem = soup.select_one(selector)
            if elem:
                text = elem.get_text(separator="\n", strip=True)
                if len(text) > 100:
                    return text
        body = soup.find("body")
        if body:
            text = re.sub(r"\n{3,}", "\n\n", body.get_text(separator="\n", strip=True))
            if len(text) > 100:
                return text
        return None

    # ── HTML parsing (CPU-bound, sync) ────────────────────────────────────────

    def _build_query(self, keywords: List[str], doc_type: Optional[str]) -> str:
        parts = ["keyword=" + "+".join(keywords)] if keywords else ["lei federal"]
        if doc_type:
            parts.append(f";f1-tipoDocumento={doc_type}")
        return "".join(parts)

    def _parse_search_results(self, html: str) -> List[Dict]:
        try:
            soup = BeautifulSoup(html, "html.parser")
            results = [
                self._parse_result_item(item)
                for item in soup.find_all("div", class_="docHit")
            ]
            return [r for r in results if r]
        except Exception as exc:
            logger.error(f"Error parsing search results: {exc}")
            return []

    def _parse_result_item(self, item) -> Optional[Dict]:
        try:
            doc: Dict = {}
            for row in item.find_all("tr"):
                cols = row.find_all("td")
                if len(cols) < 3:
                    continue
                label = cols[1].get_text(strip=True).replace(":", "").lower()
                value = cols[2]

                if "título" in label or "titulo" in label:
                    link = value.find("a")
                    if link:
                        doc["title"] = link.get_text(strip=True)
                        href = link.get("href", "")
                        if href:
                            doc["url"] = urljoin(self.BASE_URL, href)
                            if "/urn/" in href:
                                doc["urn"] = href.split("/urn/")[-1]
                elif "urn" in label:
                    urn_text = " ".join(value.get_text(strip=True).split())
                    if urn_text.startswith("urn:lex:"):
                        doc["urn"] = urn_text
                elif "data" in label:
                    doc["date"] = value.get_text(strip=True)
                elif "ementa" in label:
                    doc["description"] = value.get_text(strip=True)[:500]
                elif "autoridade" in label:
                    doc["authority"] = value.get_text(strip=True)
                elif "localidade" in label:
                    doc["location"] = value.get_text(strip=True)
                elif "assuntos" in label:
                    doc["subjects"] = value.get_text(strip=True)

            if doc.get("urn"):
                doc.update(self._parse_urn(doc["urn"]))

            return doc if (doc.get("title") or doc.get("urn")) else None
        except Exception as exc:
            logger.warning(f"Error parsing result item: {exc}")
            return None

    def _parse_urn(self, urn: str) -> Dict:
        """Parse urn:lex:br:federal:lei:1993-06-21;8666 into structured metadata."""
        parts = urn.split(":")
        if len(parts) >= 6:
            date_num = parts[5].split(";")
            return {
                "authority": parts[3] if len(parts) > 3 else None,
                "doc_type": parts[4] if len(parts) > 4 else None,
                "publication_date": date_num[0] if date_num else None,
                "number": date_num[1] if len(date_num) > 1 else None,
            }
        return {}

    def _find_source_link(self, html: str) -> Optional[str]:
        """Find Senado (preferred) or Planalto link in LexML page."""
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            if "legis.senado.leg.br/norma" in a["href"]:
                return a["href"]
        for a in soup.find_all("a", href=True):
            if "planalto.gov.br" in a["href"] and "legislacao" in a["href"]:
                return a["href"]
        return None

    def _find_senado_publication_link(
        self, soup: BeautifulSoup, base_url: str
    ) -> Optional[str]:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True).lower()
            if "publicacao" in href or "sigen" in text:
                if href.startswith("/"):
                    return f"https://legis.senado.leg.br{href}"
                if href.startswith("http"):
                    return href
                base = "/".join(base_url.split("/")[:-1])
                return f"{base}/{href}"
        return None

    # ── file I/O ──────────────────────────────────────────────────────────────

    def _save_original(self, doc: Dict, html: str, source_url: str) -> None:
        try:
            folder = ORIGINALS_DIR / _originals_folder(doc.get("doc_type", ""))
            folder.mkdir(parents=True, exist_ok=True)

            doc_type = doc.get("doc_type", "doc")
            number = doc.get("number", "")
            date = doc.get("publication_date", "")

            if number and date:
                stem = f"{doc_type}_{number}_{date}"
            else:
                slug = re.sub(r"[^\w\-]", "", doc.get("title", "doc"))[:50]
                stem = f"{slug}_{hashlib.md5(source_url.encode()).hexdigest()[:8]}"

            (folder / f"{stem}.html").write_text(html, encoding="utf-8")
            meta = {**doc, "source_url": source_url, "downloaded_at": datetime.now().isoformat()}
            (folder / f"{stem}_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.error(f"Error saving original: {exc}")
