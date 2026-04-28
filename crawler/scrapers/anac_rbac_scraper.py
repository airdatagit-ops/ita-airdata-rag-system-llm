"""
ANAC RBAC Scraper — Extrai Regulamentos Brasileiros da Aviação Civil (RBAC).

Acessa a listagem de RBACs da ANAC, extrai os links de todos os documentos
da tabela e coleta o conteúdo textual de cada página individual.

O TSPD da ANAC bloqueia User-Agents de browsers reais mas permite requisições
com o User-Agent padrão do Python/aiohttp. Por isso este scraper usa aiohttp
sem User-Agent customizado — simples, rápido e sem dependência de Selenium.

Saída: data/originals/anac/rbac-XX.txt (plain text, UTF-8)

Preparado para integração futura com o pipeline:
    chunking → embedding → sqlite → qdrant

Usage:
    from crawler.scrapers import get_scraper

    async with get_scraper("anac_rbac") as scraper:
        docs = await scraper.search()
        results = await scraper.fetch_all(docs)
"""

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
from aiolimiter import AsyncLimiter
from bs4 import BeautifulSoup
from loguru import logger

from crawler.scrapers import register_scraper
from crawler.scrapers.base import (
    BaseScraper,
    ScrapedDocument,
    compute_canonical_id,
    ORIGINALS_DIR,
)

# ── Constantes ──────────────────────────────────────────────────────────────

_INDEX_URL = "https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac"
_RBAC_URL_RE = re.compile(
    r"https?://www\.anac\.gov\.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-[\w-]+$",
    re.IGNORECASE,
)

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_RETRY_BASE_DELAY = 1.0
_MAX_RETRIES = 3

_OUTPUT_DIR = ORIGINALS_DIR / "anac"


@register_scraper
class ANACRBACscraper(BaseScraper):
    """Scraper para RBACs da ANAC.

    Usa aiohttp sem User-Agent customizado. O TSPD da ANAC bloqueia
    User-Agents de browsers reais mas permite o UA padrão do Python/aiohttp.
    """

    source_name = "anac_rbac"

    def __init__(self, timeout: int = 30, rate: float = 3.0):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._limiter = AsyncLimiter(rate, 1.0)
        self._session: Optional[aiohttp.ClientSession] = None
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("ANACRBACscraper initialized")

    def make_doc_id(self, doc: Dict) -> str:
        """ID estável e consistente com o retornado por fetch_document."""
        return f"anac_rbac_{doc['rbac_id']}"

    # ── Context manager ─────────────────────────────────────────────────────

    async def __aenter__(self):
        # Sem headers customizados: User-Agent padrão do aiohttp passa pelo TSPD
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

    # ── BaseScraper interface ────────────────────────────────────────────────

    async def search(self, *, limit: int = 0, **kwargs) -> List[Dict]:
        """Retorna lista de dicts com url, rbac_id e title de cada RBAC.

        Args:
            limit: Máximo de resultados (0 = todos).

        Returns:
            Lista de dicts: ``{"url": str, "rbac_id": str, "title": str}``
        """
        html = await self._get(_INDEX_URL)
        docs = self._extract_links(html)
        if limit > 0:
            docs = docs[:limit]
        logger.info(f"[anac_rbac] search() retornou {len(docs)} RBACs")
        return docs

    async def fetch_document(
        self, doc: Dict, save_original: bool = True
    ) -> Optional[ScrapedDocument]:
        """Baixa e extrai conteúdo de um RBAC individual.

        Args:
            doc: Dict com ``url``, ``rbac_id`` e ``title``.
            save_original: Se True, salva o arquivo .txt em disco.

        Returns:
            ScrapedDocument ou None em caso de falha.
        """
        url = doc["url"]
        rbac_id = doc["rbac_id"]
        title = doc.get("title", rbac_id.upper())

        logger.info(f"[anac_rbac] Fetching {rbac_id} — {url}")

        try:
            html = await self._get(url)
        except Exception as exc:
            logger.error(f"[anac_rbac] Error on {rbac_id}: {exc}")
            return None

        content = self._extract_content(html, rbac_id)
        if not content:
            logger.warning(f"[anac_rbac] Conteúdo vazio para {rbac_id}, pulando")
            return None

        if save_original:
            self._save_txt(rbac_id, content, url)

        number = rbac_id.replace("rbac-", "")

        return ScrapedDocument(
            doc_id=f"anac_rbac_{rbac_id}",
            source="anac_rbac",
            title=title,
            content=content,
            url=url,
            authority="ANAC",
            doc_type="RBAC",
            number=number,
            canonical_id=compute_canonical_id("RBAC", number),
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _get(self, url: str) -> str:
        """GET com rate limiting e retry + backoff exponencial."""
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with self._limiter:
                    async with self._session.get(url) as resp:
                        if resp.status in _RETRYABLE_STATUSES:
                            raise aiohttp.ClientResponseError(
                                resp.request_info,
                                resp.history,
                                status=resp.status,
                            )
                        resp.raise_for_status()
                        return await resp.text(encoding="utf-8", errors="replace")
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == _MAX_RETRIES:
                    raise
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"[anac_rbac] Retry {attempt}/{_MAX_RETRIES} para {url}: {exc}"
                )
                await asyncio.sleep(delay)
        return ""  # nunca alcançado

    def _extract_links(self, html: str) -> List[Dict]:
        """Parseia a tabela da listagem e extrai todos os links de RBAC.

        Estratégia:
        - Localiza ``<table id="tabela-normas">`` no HTML renderizado
        - Para cada linha, busca o primeiro link que:
          * corresponde a ``_RBAC_URL_RE`` (URL limpa, sem ``?visao=tabela``)
          * não aponta para arquivo direto (``@@display-file``)
        - Extrai o título da coluna ``td.tbNumero``
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", id="tabela-normas")
        if not table:
            logger.warning("[anac_rbac] Tabela 'tabela-normas' não encontrada no HTML")
            return []

        results: List[Dict] = []
        seen_urls: set = set()

        tbody = table.find("tbody")
        if not tbody:
            logger.warning("[anac_rbac] <tbody> não encontrado na tabela")
            return []

        for tr in tbody.find_all("tr"):
            # Título: coluna tbNumero
            td_num = tr.find("td", class_="tbNumero")
            title = td_num.get_text(strip=True) if td_num else ""

            # URL: primeiro link válido na linha (sem display-file, sem ?visao)
            url: Optional[str] = None
            for a in tr.find_all("a", href=True):
                href = a["href"]
                if _RBAC_URL_RE.search(href) and "display-file" not in href:
                    url = href
                    break

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            slug = url.rstrip("/").split("/")[-1]  # ex: "rbac-11", "rbac-e-94"
            results.append({"url": url, "rbac_id": slug, "title": title})

        logger.info(f"[anac_rbac] {len(results)} links extraídos da tabela")
        return results

    def _extract_content(self, html: str, rbac_id: str) -> str:
        """Extrai e limpa o texto da div de conteúdo do RBAC.

        Usa o seletor ``div#content-core`` identificado por inspeção da
        estrutura real do HTML renderizado das páginas de RBAC.

        Fallback: busca por ``div#content`` como alternativa.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Seletor primário confirmado por inspeção
        content_div = soup.find("div", id="content-core")

        if not content_div:
            # Fallback para seletor alternativo
            content_div = soup.find("div", id="content")
            if content_div:
                logger.debug(
                    f"[anac_rbac] {rbac_id}: usando fallback div#content"
                )

        if not content_div:
            logger.warning(
                f"[anac_rbac] {rbac_id}: div#content-core não encontrado"
            )
            return ""

        text = content_div.get_text(separator="\n", strip=True)

        # Normalizar múltiplas linhas em branco (máx 2 consecutivas)
        text = re.sub(r"\n{3,}", "\n\n", text)

        if not text.strip():
            logger.warning(f"[anac_rbac] {rbac_id}: texto extraído está vazio")
            return ""

        return text

    def _save_txt(self, rbac_id: str, content: str, source_url: str) -> Path:
        """Salva conteúdo em data/originals/anac/rbac-XX.txt."""
        header = (
            f"# Source: {source_url}\n"
            f"# Downloaded: {datetime.now(timezone.utc).isoformat()}\n"
            f"# RBAC: {rbac_id.upper()}\n"
            "# ---\n\n"
        )
        path = _OUTPUT_DIR / f"{rbac_id}.txt"
        path.write_text(header + content, encoding="utf-8")
        logger.success(f"[anac_rbac] Saved {path.name} ({len(content):,} chars)")
        return path
