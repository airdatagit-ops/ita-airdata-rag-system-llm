"""
SISLAER Scraper — Primary source for aviation legislation.

Document discovery uses the SISLAER search API
(``Busca/RapidaLegislacao``), querying per norma type and paginating
results.  This is vastly faster than scanning codigoRegistro IDs and
avoids the 20 K global cap by issuing one query per type.

An explicit ``strategy="ids"`` kwarg on ``search()`` falls back to ID
range scanning (kept for emergency / debugging use only).

Content priority: inline "Texto integral" > VisualizadorHtml > PDF download.

Usage:
    from crawler.scrapers import get_scraper

    async with get_scraper("sislaer") as scraper:
        docs = await scraper.search(limit=100, doc_types=["ICA"])
        results = await scraper.fetch_all(docs)
"""

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import aiohttp
from aiolimiter import AsyncLimiter
from yarl import URL
from bs4 import BeautifulSoup
from loguru import logger

from config import config
from crawler.scrapers.base import (
    BaseScraper, ScrapedDocument, DEFAULT_USER_AGENT, compute_canonical_id, split_version_year,
)

_FRONTIER_FILE = Path(config.DATA_DIR) / ".sislaer_frontier.json"
from crawler.scrapers import register_scraper
from parsers.pdf_parser import extract_text_from_bytes

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_RETRY_BASE_DELAY = 1.0
_MAX_RETRIES = 3

_PT_MONTHS = {
    "janeiro": "01", "fevereiro": "02", "março": "03", "marco": "03",
    "abril": "04", "maio": "05", "junho": "06", "julho": "07",
    "agosto": "08", "setembro": "09", "outubro": "10",
    "novembro": "11", "dezembro": "12",
}
_DATE_NUMERIC_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
_DATE_SPELLED_RE = re.compile(
    r"\b(\d{1,2})\s+[dD][eE]\s+("
    + "|".join(_PT_MONTHS)
    + r")\s+[dD][eE]\s+(\d{4})\b",
    re.IGNORECASE,
)


def _extract_date_from_text(text: str | None) -> str | None:
    """Extract a ``dd/mm/yyyy`` date from free text like publicacao references."""
    if not text:
        return None
    m = _DATE_NUMERIC_RE.search(text)
    if m:
        return m.group(1)
    m = _DATE_SPELLED_RE.search(text)
    if m:
        day = m.group(1).zfill(2)
        month = _PT_MONTHS[m.group(2).lower()]
        return f"{day}/{month}/{m.group(3)}"
    return None

_TITLE_RE = re.compile(
    r"^(?P<type>[A-ZÇÃa-zçã][A-Za-zÇÃçã\s-]*?)\s+"
    r"(?:Nº\s+)?(?P<number>[\d][\d./-]*[\w/]*\d+)",
)

_DETAIL_LINK_RE = re.compile(r"tw\.irParaDetalheComVoltar\((\d+)\)")

# Maps SISLAER norma codes to human-readable type labels (from /norma/ListarNormaSet).
NORMA_CODE_MAP: Dict[str, int] = {
    "AVISO": 20, "BCA": 17, "BMA": 24, "BOLETIM EXTERNO": 44,
    "COMUNICADO": 38, "CONSTITUIÇÃO FEDERAL": 39, "DCA": 3,
    "DECRETO": 18, "DECRETO - LEI": 25, "FCA": 4, "ICA": 5,
    "IMA": 31, "INSTRUÇÃO NORMATIVA": 43, "LEI": 22,
    "LEI COMPLEMENTAR": 41, "MANUAL - OUTROS": 40,
    "MANUAL ELETRÔNICO": 21, "MCA": 6, "MEDIDA PROVISÓRIA": 42,
    "NOPREP": 30, "NORMAS DO COMPREP": 29, "NOTA": 33,
    "NPA": 36, "NSCA": 1, "OCA": 7, "ORDEM TÉCNICA": 34,
    "ORIENTAÇÃO NORMATIVA": 35, "PCA": 8, "PORTARIA": 19,
    "PORTARIA CONJUNTA": 37, "PTA": 28, "RCA": 16,
    "RESOLUÇÃO": 32, "RICA": 14, "RIMA": 45, "RMA": 26,
    "ROCA": 15, "TCA": 9,
}


class _SessionExpired(Exception):
    """Raised when the SISLAER search session (CSRF token) has expired."""


def _resolve_norma_codes(type_names: Set[str]) -> List[int]:
    """Map human-readable type names to SISLAER norma codes.

    Returns the list of codes that could be resolved. Types not found in
    ``NORMA_CODE_MAP`` are silently skipped (they'll be caught by the ID
    scan fallback if needed).
    """
    codes: List[int] = []
    for name in type_names:
        key = name.strip().upper()
        if key in NORMA_CODE_MAP:
            codes.append(NORMA_CODE_MAP[key])
        else:
            for map_key, code in NORMA_CODE_MAP.items():
                if map_key.startswith(key) or key.startswith(map_key):
                    codes.append(code)
                    break
    return sorted(set(codes))


_KNOWN_TYPES_SORTED = sorted(NORMA_CODE_MAP.keys(), key=len, reverse=True)


def _parse_title(raw: str) -> Dict[str, Optional[str]]:
    """Extract doc_type and number from a SISLAER title.

    Handles structured titles like ``ICA 96-1/2025`` as well as
    type-only titles like ``NSCA`` or ``NSCA/2023``.
    """
    text = raw.strip()
    m = _TITLE_RE.match(text)
    if m:
        return {"doc_type": m.group("type").strip(), "number": m.group("number").strip()}

    upper = text.upper()
    for norma_type in _KNOWN_TYPES_SORTED:
        if upper.startswith(norma_type):
            rest = text[len(norma_type):].strip().lstrip("/").strip()
            number = rest if rest else None
            return {"doc_type": norma_type, "number": number}

    return {"doc_type": None, "number": None}


def _normalize_doc_type(raw: Optional[str]) -> Optional[str]:
    """Normalize common type variations to uppercase canonical form."""
    if not raw:
        return None
    t = raw.strip().upper()
    mapping = {
        "PORTARIA": "Portaria",
        "PORTARIA CONJUNTA": "Portaria Conjunta",
        "INSTRUÇÃO NORMATIVA": "Instrução Normativa",
        "INSTRUCAO NORMATIVA": "Instrução Normativa",
        "LEI": "Lei",
        "LEI COMPLEMENTAR": "Lei Complementar",
        "DECRETO": "Decreto",
        "DECRETO - LEI": "Decreto-Lei",
        "DECRETO-LEI": "Decreto-Lei",
        "MEDIDA PROVISÓRIA": "Medida Provisória",
        "RESOLUÇÃO": "Resolução",
        "RESOLUCAO": "Resolução",
        "CONSTITUIÇÃO FEDERAL": "Constituição Federal",
        "ORDEM TÉCNICA": "Ordem técnica",
        "ORIENTAÇÃO NORMATIVA": "Orientação normativa",
        "MANUAL ELETRÔNICO": "Manual Eletrônico",
    }
    return mapping.get(t, t)


@register_scraper
class SISLAERScraper(BaseScraper):
    """Async scraper for SISLAER TerminalWebCENDOC legislation portal.

    Default discovery uses the Search API (``Busca/RapidaLegislacao``),
    querying per norma type for fast, complete results.  Pass
    ``strategy="ids"`` to ``search()`` for the legacy ID-scan fallback.
    """

    source_name = "sislaer"

    _HEADERS = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9",
    }

    _DEFAULT_DOC_TYPES = {
        "ICA", "DCA", "FCA", "MCA", "NSCA", "PCA", "RCA", "TCA",
        "OCA", "ROCA", "RICA", "RIMA", "RMA", "NPA", "PTA",
        "LEI", "DECRETO", "DECRETO - LEI",
        "RESOLUÇÃO", "INSTRUÇÃO NORMATIVA",
        "MEDIDA PROVISÓRIA", "LEI COMPLEMENTAR",
        "CONSTITUIÇÃO FEDERAL",
        "ORDEM TÉCNICA", "ORIENTAÇÃO NORMATIVA",
        "MANUAL ELETRÔNICO", "MANUAL - OUTROS",
        "AVISO", "COMUNICADO", "NOTA",
        "BCA", "BMA", "IMA", "BOLETIM EXTERNO",
        "NOPREP", "NORMAS DO COMPREP",
    }

    def __init__(
        self,
        max_rate: float = None,
        concurrency: int = None,
        timeout: int = None,
        start_id: int = None,
        end_id: int = None,
    ):
        rate = max_rate or config.SISLAER_MAX_RATE
        self._concurrency = concurrency or config.SISLAER_CONCURRENCY
        self._timeout_sec = timeout or config.SISLAER_TIMEOUT
        self._start_id = start_id or config.SISLAER_START_ID
        self._end_id = end_id or config.SISLAER_END_ID or None
        self._limiter = AsyncLimiter(rate, 1.0)
        self._session: Optional[aiohttp.ClientSession] = None
        self._base_url = config.SISLAER_BASE_URL

        # Search API session state (cookie + anti-forgery token)
        self._csrf_token: Optional[str] = None

        # Per-instance counters for the detail-page anti-bot gate.
        # Reported once at the end of a run instead of per-doc warnings.
        self._validation_stats = {
            "attempts": 0,
            "successes": 0,
            "still_shell": 0,    # gave up after _DETALHE_UNLOCK_MAX_CYCLES
            "total_cycles": 0,   # cumulative GET-POST cycles across all attempts
        }

        logger.info(
            f"SISLAERScraper initialized (rate={rate}/s, concurrency={self._concurrency}, "
            f"ids={self._start_id}-{self._end_id or 'auto'})"
        )

    async def __aenter__(self) -> "SISLAERScraper":
        self._session = aiohttp.ClientSession(
            headers=self._HEADERS,
            connector=aiohttp.TCPConnector(limit=self._concurrency),
            timeout=aiohttp.ClientTimeout(total=self._timeout_sec),
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        s = self._validation_stats
        if s["attempts"]:
            avg_cycles = s["total_cycles"] / max(s["attempts"], 1)
            logger.info(
                "[sislaer] anti-bot gate stats: "
                f"attempts={s['attempts']} successes={s['successes']} "
                f"still_shell={s['still_shell']} avg_cycles={avg_cycles:.1f}"
            )

    # ── Search API session management ─────────────────────────

    async def _ensure_search_session(self) -> str:
        """GET the search page to obtain cookie + CSRF token.

        Returns the anti-forgery token for subsequent POST requests.
        Reuses existing token if available.
        """
        if self._csrf_token:
            return self._csrf_token

        url = f"{self._base_url}/Busca/Legislacao"
        async with self._limiter:
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Failed to load search page: HTTP {resp.status}")
                html = await resp.text()

        m = re.search(
            r'name="__RequestVerificationToken"[^>]*value="([^"]+)"', html,
        )
        if not m:
            raise RuntimeError("Could not extract CSRF token from search page")

        self._csrf_token = m.group(1)
        logger.debug("[sislaer] Search session established (CSRF token obtained)")
        return self._csrf_token

    async def _refresh_search_session(self) -> str:
        """Force a new search session (e.g. after token expiry)."""
        self._csrf_token = None
        return await self._ensure_search_session()

    # ── Search API: paginated query by norma type ─────────────

    async def _search_by_api(
        self, *, norma_codes: List[int], limit: int,
    ) -> List[Dict]:
        """Discover documents via the SISLAER search API.

        Performs one ``POST Busca/RapidaLegislacao`` per norma code, then
        paginates through ``POST Resultado/CarregarPaginaLayoutDetalhe``
        to collect all ``codigoRegistro`` values.
        """
        token = await self._ensure_search_session()
        all_found: List[Dict] = []
        seen_ids: Set[int] = set()

        for code in norma_codes:
            if len(all_found) >= limit:
                break

            guid = str(uuid.uuid4())
            form_data = {
                "__RequestVerificationToken": token,
                "Guid": guid,
                "TipoBuscaRapida": "0",
                "IniciadoCom": "false",
                "PalavraChave": "",
                "CodigosNorma": str(code),
                "Numero": "",
                "Ano": "",
                "CodigosOrgao": "",
            }

            try:
                page1_html = await self._post_search(
                    f"{self._base_url}/Busca/RapidaLegislacao?bibliotecas=",
                    form_data,
                )
            except _SessionExpired:
                token = await self._refresh_search_session()
                form_data["__RequestVerificationToken"] = token
                page1_html = await self._post_search(
                    f"{self._base_url}/Busca/RapidaLegislacao?bibliotecas=",
                    form_data,
                )

            if not page1_html:
                continue

            # Update CSRF token from the response page
            new_tok = re.search(
                r'name="__RequestVerificationToken"[^>]*value="([^"]+)"',
                page1_html,
            )
            if new_tok:
                token = new_tok.group(1)
                self._csrf_token = token

            total, per_page, total_pages = self._parse_result_meta(page1_html)
            page_docs = self._extract_result_ids(page1_html, guid)

            norma_label = next(
                (k for k, v in NORMA_CODE_MAP.items() if v == code), str(code)
            )
            logger.info(
                f"[sislaer] search API: {norma_label} → {total} docs "
                f"({total_pages} pages)"
            )

            for doc in page_docs:
                rid = doc["codigoRegistro"]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    all_found.append(doc)

            remaining_pages = min(total_pages, (limit - len(all_found)) // per_page + 2)
            for page_num in range(2, remaining_pages + 1):
                if len(all_found) >= limit:
                    break

                page_url = (
                    f"{self._base_url}/Resultado/CarregarPaginaLayoutDetalhe"
                    f"?paginaInicial={page_num}&guid={guid}"
                )
                try:
                    page_html = await self._post_search(
                        page_url,
                        {"__RequestVerificationToken": token},
                        xhr=True,
                    )
                except _SessionExpired:
                    token = await self._refresh_search_session()
                    page_html = await self._post_search(
                        page_url,
                        {"__RequestVerificationToken": token},
                        xhr=True,
                    )

                if not page_html:
                    break

                page_docs = self._extract_result_ids(page_html)
                if not page_docs:
                    break

                for doc in page_docs:
                    rid = doc["codigoRegistro"]
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        all_found.append(doc)

                if page_num % 20 == 0:
                    logger.info(
                        f"[sislaer] {norma_label}: page {page_num}/{total_pages}, "
                        f"{len(all_found)} total docs so far"
                    )

        result = all_found[:limit]
        logger.info(f"[sislaer] Search API complete: {len(result)} documents found")
        return result

    async def _post_search(
        self, url: str, data: Dict, *, xhr: bool = False,
    ) -> Optional[str]:
        """POST to a search/pagination endpoint with rate limiting and retry."""
        headers = {}
        if xhr:
            headers["X-Requested-With"] = "XMLHttpRequest"

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with self._limiter:
                    async with self._session.post(
                        url, data=data, headers=headers,
                    ) as resp:
                        if resp.status == 200:
                            return await resp.text()
                        if resp.status in (400, 403):
                            raise _SessionExpired()
                        if resp.status in _RETRYABLE_STATUSES:
                            raise aiohttp.ClientResponseError(
                                resp.request_info, resp.history,
                                status=resp.status,
                            )
                        return None
            except _SessionExpired:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        f"[sislaer] POST attempt {attempt}/{_MAX_RETRIES} "
                        f"failed ({exc}), retrying in {delay:.0f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"[sislaer] POST all attempts failed: {url}")
        return None

    @staticmethod
    def _parse_result_meta(html: str) -> tuple:
        """Extract (total_registros, tamanho_pagina, total_paginas) from results page."""
        total = per_page = pages = 0
        m = re.search(r'data-total-registros="(\d+)"', html)
        if m:
            total = int(m.group(1))
        m = re.search(r'data-tamanho-pagina="(\d+)"', html)
        if m:
            per_page = int(m.group(1))
        m = re.search(r'data-total-paginas="(\d+)"', html)
        if m:
            pages = int(m.group(1))
        return total, per_page or 20, pages

    @staticmethod
    def _extract_result_ids(
        html: str, guid: str = None,
    ) -> List[Dict]:
        """Extract codigoRegistro + title from a results page HTML.

        Handles two HTML layouts:
          - Full page (page 1): ``<a ... title="ICA ...">``
          - XHR fragment (page 2+): ``<img alt="ICA ..." class="capa-ficha">``
        Falls back to pairing href IDs with img alt values.
        """
        # Strategy 1: title attribute on the <a> tag (full-page results)
        pattern_title = r'acervo/detalhe/(\d+)\?[^"]*"[^>]*title="([^"]*)"'
        matches = re.findall(pattern_title, html)

        if not matches:
            # Strategy 2: pair detalhe IDs with <img alt="..."> (XHR pages)
            ids = re.findall(r'acervo/detalhe/(\d+)', html)
            alts = re.findall(r'alt="([^"]+)"[^>]*class="capa-ficha"', html)
            unique_ids = list(dict.fromkeys(ids))
            matches = list(zip(unique_ids, alts))

        seen: Set[int] = set()
        results: List[Dict] = []
        for code_str, title in matches:
            code = int(code_str)
            if code in seen:
                continue
            seen.add(code)
            parsed = _parse_title(title)
            results.append({
                "codigoRegistro": code,
                "title": title,
                "doc_type": parsed.get("doc_type"),
                "number": parsed.get("number"),
            })
        return results

    # ── auto-discovery (ID scan) ──────────────────────────────

    @staticmethod
    def _find_doc_title(soup: BeautifulSoup) -> Optional[str]:
        """Return the document title h1 (the second one), or None."""
        h1s = soup.find_all("h1")
        if len(h1s) >= 2:
            text = h1s[1].get_text(strip=True)
            if text and len(text) > 3:
                return text
        return None

    async def _id_exists(self, reg_id: int) -> bool:
        """Check whether a codigoRegistro returns a page with actual document content."""
        html = await self._get_html(f"{self._base_url}/acervo/detalhe/{reg_id}")
        if not html or len(html) < 500:
            return False
        soup = BeautifulSoup(html, "html.parser")
        return self._find_doc_title(soup) is not None

    @staticmethod
    def _load_frontier() -> Optional[int]:
        """Load cached end_id from disk."""
        try:
            if _FRONTIER_FILE.exists():
                data = json.loads(_FRONTIER_FILE.read_text())
                end_id = data.get("end_id")
                if end_id and isinstance(end_id, int):
                    logger.info(f"[sislaer] Loaded cached frontier: {end_id}")
                    return end_id
        except Exception:
            pass
        return None

    @staticmethod
    def _save_frontier(end_id: int) -> None:
        """Persist discovered end_id to disk for future runs."""
        try:
            _FRONTIER_FILE.parent.mkdir(parents=True, exist_ok=True)
            _FRONTIER_FILE.write_text(json.dumps({"end_id": end_id}))
            logger.info(f"[sislaer] Saved frontier to {_FRONTIER_FILE}: {end_id}")
        except Exception as exc:
            logger.warning(f"[sislaer] Could not save frontier: {exc}")

    async def _discover_end_id(self) -> int:
        """Find the highest valid codigoRegistro via binary search.

        First checks a cached frontier file. If not found, runs a binary
        search starting from 55,000. The result is saved for future runs.
        """
        cached = self._load_frontier()
        if cached:
            return cached

        hint = 55_000
        lo, hi = self._start_id, hint

        max_upper = hint * 4
        if await self._id_exists(hi):
            while await self._id_exists(hi) and hi < max_upper:
                lo = hi
                hi = min(hi * 2, max_upper)
                logger.debug(f"[sislaer] auto-discover: expanding to {hi}")

        logger.info(f"[sislaer] auto-discover: binary search in [{lo}, {hi}]")
        while lo < hi - 1:
            mid = (lo + hi) // 2
            exists = await self._id_exists(mid)
            logger.debug(f"[sislaer] binary search: id={mid} exists={exists}")
            if exists:
                lo = mid
            else:
                hi = mid

        max_id = lo
        gap = 0
        for probe_id in range(lo + 1, lo + 101):
            if await self._id_exists(probe_id):
                max_id = probe_id
                gap = 0
            else:
                gap += 1
                if gap >= 20:
                    break

        logger.info(f"[sislaer] Auto-discovered end ID: {max_id}")
        self._save_frontier(max_id)
        return max_id

    async def _search_by_ids(
        self, *, allowed: Set[str], limit: int,
    ) -> List[Dict]:
        """Discover documents by scanning codigoRegistro IDs."""
        if self._end_id is None:
            self._end_id = await self._discover_end_id()

        sem = asyncio.Semaphore(self._concurrency)
        found: List[Dict] = []
        total_ids = self._end_id - self._start_id + 1

        async def _probe(reg_id: int):
            if len(found) >= limit:
                return
            async with sem:
                html = await self._get_html(
                    f"{self._base_url}/acervo/detalhe/{reg_id}"
                )
            if not html or len(html) < 500:
                return

            soup = BeautifulSoup(html, "html.parser")
            title = self._find_doc_title(soup)
            if not title:
                return

            parsed = _parse_title(title)
            doc_type_raw = parsed["doc_type"]
            if not doc_type_raw:
                return
            if allowed and doc_type_raw.upper() not in allowed:
                return

            found.append({
                "codigoRegistro": reg_id,
                "title": title,
                "doc_type": doc_type_raw,
                "number": parsed["number"],
            })

        batch_size = 500
        for batch_start in range(self._start_id, self._end_id + 1, batch_size):
            if len(found) >= limit:
                break
            batch_end = min(batch_start + batch_size, self._end_id + 1)
            tasks = [_probe(i) for i in range(batch_start, batch_end)]
            await asyncio.gather(*tasks)

            progress = min(batch_end - self._start_id, total_ids)
            logger.info(
                f"[sislaer] ID scan progress: {progress}/{total_ids} IDs, "
                f"{len(found)} docs found"
            )

        result = found[:limit]
        logger.info(f"[sislaer] ID scan complete: {len(result)} documents matched")
        return result

    # ── BaseScraper interface ────────────────────────────────────

    async def search(self, *, limit: int = 100, **kwargs) -> List[Dict]:
        """Discover documents via the SISLAER search API.

        By default, queries the Search API per norma type — fast and
        complete.  Pass ``strategy="ids"`` for the legacy ID-scan
        fallback (slower, kept for debugging).

        Keyword Args:
            doc_types: List of type names to include (e.g. ``["ICA", "DCA"]``).
                       Defaults to ``config.SISLAER_DOC_TYPES``.
            strategy:  ``"api"`` (default) or ``"ids"`` (legacy fallback).
        """
        strategy = kwargs.get("strategy", "api")
        raw_types = kwargs.get("doc_types") or config.SISLAER_DOC_TYPES.split(",")
        allowed = {t.strip().upper() for t in raw_types}

        if strategy == "ids":
            logger.info(f"[sislaer] Using ID scan (explicit strategy, limit={limit})")
            return await self._search_by_ids(allowed=allowed, limit=limit)

        norma_codes = _resolve_norma_codes(allowed)
        if not norma_codes:
            norma_codes = sorted(NORMA_CODE_MAP.values())

        logger.info(
            f"[sislaer] Using Search API for {len(norma_codes)} norma types "
            f"(limit={limit})"
        )
        return await self._search_by_api(norma_codes=norma_codes, limit=limit)

    async def fetch_document(
        self, doc: Dict, save_original: bool = True
    ) -> Optional[ScrapedDocument]:
        """Fetch and parse a full document detail page.

        The detail page is gated by a per-doc anti-bot check. Once
        unlocked, the validation cookie lives in an ephemeral session
        owned by this call — viewer/PDF follow-up requests for the
        same doc must use that session, otherwise they hit the gate
        again on the main session.
        """
        reg_id = doc["codigoRegistro"]
        url = f"{self._base_url}/acervo/detalhe/{reg_id}"

        eph: Optional[aiohttp.ClientSession] = None
        try:
            html = await self._raw_get(url)
            if html is None:
                return None
            if self._is_detalhe_validation_page(html):
                unlocked = await self._unlock_detail(url, html)
                if unlocked is None:
                    return None
                html, eph = unlocked

            if len(html) < 500:
                return None

            soup = BeautifulSoup(html, "html.parser")
            detail = self._parse_detail(soup, reg_id)
            if not detail:
                return None

            content = await self._get_content(
                soup, reg_id, save_original, session=eph,
            )
            if not content or len(content.strip()) < 50:
                return None
        finally:
            if eph is not None:
                await eph.close()

        parsed = _parse_title(detail["title"])
        doc_type = _normalize_doc_type(parsed["doc_type"]) or doc.get("doc_type")
        raw_number = parsed["number"] or doc.get("number") or ""

        number, version_year = split_version_year(raw_number)
        canonical = compute_canonical_id(doc_type, number)
        doc_id = f"{canonical}/{version_year}" if canonical and version_year else (canonical or f"sislaer_{reg_id}")

        situacao = (detail.get("situacao") or "").lower()
        status = "revoked" if "revogado" in situacao else None

        authority = detail.get("authority")
        relations = detail.get("relations", [])

        pub_date = (
            detail.get("ato_publicacao")
            or _extract_date_from_text(detail.get("publicacao"))
            or detail.get("portaria_aprovacao")
        )
        source_ref = f"sislaer:{reg_id}"
        metadata = {
            "situacao": detail.get("situacao"),
            "publication_date": pub_date,
            "revocation_date": detail.get("revocation_date"),
            "portaria_aprovacao": detail.get("portaria_aprovacao"),
            "ato_publicacao": detail.get("ato_publicacao"),
            "publicacao": detail.get("publicacao"),
            "ementa": detail.get("ementa"),
            "observacoes": detail.get("observacoes"),
            "natureza": detail.get("natureza"),
            "relations": relations,
        }

        if save_original:
            self.save_original_file(
                html,
                folder_name="sislaer",
                stem=f"sislaer_{reg_id}",
                extension=".html",
                meta={"codigoRegistro": reg_id, "title": detail["title"], "url": url},
            )

        return ScrapedDocument(
            doc_id=doc_id,
            source=self.source_name,
            title=detail["title"],
            content=content,
            metadata=metadata,
            url=url,
            doc_type=doc_type,
            canonical_id=canonical,
            source_ref=source_ref,
            status=status,
            number=number,
            authority=authority,
            version_year=version_year,
        )

    def make_doc_id(self, doc: Dict) -> str:
        return f"sislaer_{doc['codigoRegistro']}"

    # ── HTTP with retry ──────────────────────────────────────────

    # Anti-bot validation gate. On the first GET to /acervo/detalhe/{id}
    # the portal returns a small HTML shell ("Aguarde, estamos validando
    # sua requisição...") with an in-page ``AntiForgeryToken``. The
    # bundled ``validacao.js`` POSTs that token to
    # ``/acervo/validaacessodetalhe`` and reloads the page; the POST
    # always returns ``{"Resultado": true}`` and sets a per-request
    # ``SBW.REQUEST.VALIDATION`` cookie.
    #
    # Empirically (probed against the live site):
    #   * The cookie sometimes lets the GET through immediately, but
    #     frequently the next GET still returns the shell — at which
    #     point a fresh POST (with the new token from the new shell)
    #     is required. With a 1 s gap between cycles, 2 cycles cover
    #     ~95% of docs and 8 cycles cover essentially everything.
    #   * The shell↔real-page transition is independent across docs,
    #     so as long as each doc has its own cookie jar (ephemeral
    #     session) we can run dozens in parallel — measured 32/32 OK
    #     at 4 docs/s with concurrency=32.
    _DETALHE_VALIDATION_MARKER = "estamos validando"
    _DETALHE_VALIDATE_URL = "/acervo/validaacessodetalhe"
    # Max GET-POST cycles per doc. Empirically the 99th percentile is
    # ≤4; 8 leaves ample headroom for slow corner cases.
    _DETALHE_UNLOCK_MAX_CYCLES = 8
    # Pause between cycles. Going below ~0.5 s gains nothing but
    # extra load; the backend needs a moment to propagate the cookie.
    _DETALHE_UNLOCK_CYCLE_DELAY_S = 1.0
    _ANTIFORGERY_RE = re.compile(r"AntiForgeryToken\s*=\s*'([^']+)'")

    async def _get_html(
        self, url: str, *, session: Optional[aiohttp.ClientSession] = None
    ) -> Optional[str]:
        """Fetch a URL, transparently handling the anti-bot gate.

        ``session`` defaults to the main session. Pass an ephemeral
        session (created by :meth:`_unlock_detail`) when fetching
        follow-up resources for the same doc (viewer HTML, PDF), so
        the validation cookie travels with the request.
        """
        sess = session or self._session
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                html = await self._raw_get(url, session=sess)
                if html is None:
                    return None
                if self._is_detalhe_validation_page(html):
                    # Promote the main session through the gate by
                    # discarding the ephemeral jar after copying the
                    # html out — used only when caller didn't pass a
                    # dedicated session.
                    unlocked = await self._unlock_detail(url, html)
                    if unlocked is None:
                        return None
                    html, eph = unlocked
                    await eph.close()
                return html
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        f"[sislaer] Attempt {attempt}/{_MAX_RETRIES} failed ({exc}), "
                        f"retrying in {delay:.0f}s: {url}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"[sislaer] All {_MAX_RETRIES} attempts failed: {url}")
        return None

    async def _unlock_detail(
        self, url: str, shell_html: str
    ) -> Optional[Tuple[str, aiohttp.ClientSession]]:
        """Bypass the per-doc anti-bot gate.

        Returns ``(html, ephemeral_session)`` on success; the caller
        owns the session and must close it after using it for any
        follow-up requests on the same doc (viewer HTML, PDF). The
        validation cookie is held in the ephemeral session's jar.

        Algorithm: alternating POST validate / GET cycles in an
        isolated cookie jar. The POST always returns
        ``{"Resultado": true}`` but the next GET frequently still
        serves the shell; a fresh POST against the new shell's token
        drives the cookie state forward. Empirically 1-2 cycles cover
        most docs and ≤4 cycles covers ~99% — we cap at
        ``_DETALHE_UNLOCK_MAX_CYCLES``.

        The ephemeral session is isolated so concurrent workers never
        race on the shared cookie jar; the main session's TCP
        connector is reused (``connector_owner=False``) so the
        keep-alive pool is preserved — no extra TLS handshakes.
        """
        self._validation_stats["attempts"] += 1
        eph = self._make_ephemeral_session()
        try:
            current_html = shell_html
            cycles_used = 0
            for cycle in range(1, self._DETALHE_UNLOCK_MAX_CYCLES + 1):
                cycles_used = cycle
                m = self._ANTIFORGERY_RE.search(current_html)
                if not m:
                    break
                token = m.group(1)
                ok = await self._post_validate(eph, url, token)
                if not ok:
                    break
                await asyncio.sleep(self._DETALHE_UNLOCK_CYCLE_DELAY_S)
                refreshed = await self._raw_get(url, referer=url, session=eph)
                if refreshed is None:
                    break
                if not self._is_detalhe_validation_page(refreshed):
                    self._validation_stats["successes"] += 1
                    self._validation_stats["total_cycles"] += cycles_used
                    return refreshed, eph
                current_html = refreshed
            await eph.close()
            self._validation_stats["still_shell"] += 1
            self._validation_stats["total_cycles"] += cycles_used
            return None
        except BaseException:
            await eph.close()
            raise

    async def _post_validate(
        self, session: aiohttp.ClientSession, url: str, token: str,
    ) -> bool:
        """POST the AntiForgeryToken; returns True iff Resultado:true."""
        validate_url = f"{self._base_url}{self._DETALHE_VALIDATE_URL}"
        headers = {
            "RequestVerificationToken": token,
            "X-Requested-With": "XMLHttpRequest",
            "Referer": url,
        }
        try:
            async with self._limiter:
                async with session.post(validate_url, headers=headers) as resp:
                    if resp.status != 200:
                        return False
                    try:
                        data = await resp.json(content_type=None)
                    except aiohttp.ContentTypeError:
                        return False
                    return bool(data.get("Resultado"))
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.debug(f"[sislaer] POST validate failed: {exc}")
            return False

    def _make_ephemeral_session(self) -> aiohttp.ClientSession:
        """Build an ephemeral session that inherits the main session's
        cookies (e.g. ``ASP.NET_SessionId``) but isolates anything the
        backend writes during the gate (the per-doc
        ``SBW.REQUEST.VALIDATION``). When the session closes the
        ephemeral cookies are dropped — the main jar stays clean and
        concurrent workers cannot race on the validation cookie.
        """
        eph = aiohttp.ClientSession(
            connector=self._session.connector,
            connector_owner=False,
            cookie_jar=aiohttp.CookieJar(unsafe=True),
            headers=self._HEADERS,
            timeout=aiohttp.ClientTimeout(total=self._timeout_sec),
        )
        seed = {c.key: c.value for c in self._session.cookie_jar}
        if seed:
            eph.cookie_jar.update_cookies(seed, response_url=URL(self._base_url))
        return eph

    async def _raw_get(
        self,
        url: str,
        *,
        referer: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Optional[str]:
        """Single GET without anti-bot retry. Caller handles retries."""
        sess = session or self._session
        headers = {"Referer": referer} if referer else None
        async with self._limiter:
            async with sess.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.text()
                if resp.status in _RETRYABLE_STATUSES:
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history, status=resp.status
                    )
                return None

    @classmethod
    def _is_detalhe_validation_page(cls, html: str) -> bool:
        return cls._DETALHE_VALIDATION_MARKER in (html or "").lower()

    # ── detail page parsing ──────────────────────────────────────

    def _parse_detail(self, soup: BeautifulSoup, reg_id: int) -> Optional[Dict]:
        title = self._find_doc_title(soup)
        if not title:
            return None

        detail: Dict = {"title": title}

        authority_link = soup.find("a", href=re.compile(r"autoresClick"))
        if authority_link:
            detail["authority"] = authority_link.get_text(strip=True)

        field_map = {
            "situação": "situacao",
            "situacao": "situacao",
            "portaria de aprovação": "portaria_aprovacao",
            "portaria de aprovacao": "portaria_aprovacao",
            "ato de publicação": "ato_publicacao",
            "ato de publicacao": "ato_publicacao",
            "publicação": "publicacao",
            "publicacao": "publicacao",
            "natureza / esfera": "natureza",
            "observações": "observacoes",
            "observacoes": "observacoes",
        }

        for text_node in soup.find_all(string=True):
            label = text_node.strip().lower().rstrip(":")
            if label in field_map:
                parent = text_node.parent
                if parent:
                    sibling = parent.find_next_sibling()
                    if sibling:
                        detail[field_map[label]] = sibling.get_text(strip=True)

        ementa_divs = soup.find_all("div", id=re.compile(r"Ementa.*html", re.I))
        if ementa_divs:
            detail["ementa"] = ementa_divs[-1].get_text(strip=True)
        else:
            for text_node in soup.find_all(string=re.compile(r"Portaria\s*/\s*Ementa", re.I)):
                parent = text_node.parent
                if parent:
                    sibling = parent.find_next_sibling()
                    if sibling:
                        detail["ementa"] = sibling.get_text(strip=True)
                        break

        relations = []
        for section_label, rel_type in [
            ("Alterações", "amends"),
            ("Correlações", "correlates"),
        ]:
            for text_node in soup.find_all(string=re.compile(rf"^\s*{section_label}\s*$", re.I)):
                container = text_node.parent
                if not container:
                    continue
                sibling = container.find_next_sibling()
                if not sibling:
                    continue
                for link in sibling.find_all("a", href=True):
                    href = link.get("href", "")
                    m = _DETAIL_LINK_RE.search(href)
                    if m:
                        target_text = link.get_text(strip=True)
                        actual_type = rel_type
                        if "Revogad" in target_text:
                            actual_type = "revoked_by" if "por" in target_text.lower() else "revokes"
                        relations.append({
                            "target_ref": m.group(1),
                            "type": actual_type,
                            "label": target_text,
                        })

        detail["relations"] = relations

        # BCA revocation date (from <span class="rotulo" title="BCA - REVOGAÇÃO">)
        for span in soup.find_all("span", class_="rotulo"):
            if "REVOGA" in (span.get("title") or "").upper():
                parent_p = span.find_parent("p")
                if parent_p:
                    for a in parent_p.find_all("a", href=True):
                        m = re.search(r"(\d{2})-(\d{2})-(\d{4})", a["href"])
                        if m:
                            detail["revocation_date"] = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
                            break
                break

        return detail

    # ── content extraction ───────────────────────────────────────

    async def _get_content(
        self,
        soup: BeautifulSoup,
        reg_id: int,
        save_original: bool,
        *,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Optional[str]:
        """Extract content in priority order: inline text > HTML viewer > PDF.

        ``session`` should be the ephemeral session that holds the
        validation cookie for this doc. Without it, viewer/PDF GETs
        trigger the anti-bot gate again on the main session.
        """
        sess = session or self._session

        # 1) Inline "Texto integral" expanded div
        for div in soup.find_all("div", id=re.compile(r"TextoIntegral.*html", re.I)):
            text = div.get_text(separator="\n", strip=True)
            if text and len(text) > 100:
                logger.debug(f"[sislaer] {reg_id}: inline text ({len(text)} chars)")
                return text

        # 2) VisualizadorHtml link
        for a in soup.find_all("a", href=re.compile(r"VisualizadorHtml")):
            html_url = a.get("href", "")
            if not html_url.startswith("http"):
                html_url = f"https://www.sislaer.fab.mil.br{html_url}"
            viewer_html = await self._get_html(html_url, session=sess)
            if viewer_html and len(viewer_html) > 200:
                viewer_soup = BeautifulSoup(viewer_html, "html.parser")
                for tag in viewer_soup(["script", "style", "nav", "header", "footer"]):
                    tag.decompose()
                body = viewer_soup.find("body") or viewer_soup
                text = body.get_text(separator="\n", strip=True)
                if text and len(text) > 100:
                    logger.debug(f"[sislaer] {reg_id}: viewer HTML ({len(text)} chars)")
                    return text

        # 3) PDF download
        for a in soup.find_all("a", href=re.compile(r"Busca/Download")):
            pdf_url = a.get("href", "")
            if not pdf_url.startswith("http"):
                pdf_url = f"https://www.sislaer.fab.mil.br{pdf_url}"
            try:
                async with self._limiter:
                    async with sess.get(pdf_url) as resp:
                        if resp.status != 200:
                            continue
                        pdf_bytes = await resp.read()
                if pdf_bytes and pdf_bytes[:4] == b"%PDF":
                    if save_original:
                        self.save_original_file(
                            pdf_bytes,
                            folder_name="sislaer",
                            stem=f"sislaer_{reg_id}",
                            extension=".pdf",
                        )
                    text = extract_text_from_bytes(pdf_bytes)
                    if text and len(text) > 100:
                        logger.debug(f"[sislaer] {reg_id}: PDF ({len(text)} chars)")
                        return text
            except Exception as exc:
                logger.warning(f"[sislaer] PDF download failed for {reg_id}: {exc}")

        # 4) Fallback: ementa text from detail page
        ementa_divs = soup.find_all("div", id=re.compile(r"Ementa.*html", re.I))
        if ementa_divs:
            text = ementa_divs[-1].get_text(separator="\n", strip=True)
            if text and len(text) > 50:
                logger.debug(f"[sislaer] {reg_id}: ementa fallback ({len(text)} chars)")
                return text

        return None
