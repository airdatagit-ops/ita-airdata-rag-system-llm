"""
ANAC RBAC Scraper — Extrai Regulamentos Brasileiros da Aviação Civil (RBAC).

Acessa a listagem de RBACs da ANAC, extrai os links de todos os documentos
da tabela e coleta o conteúdo textual de cada página individual.

O site da ANAC é protegido por sistema anti-bot (TSPD), o que impede o uso
direto de aiohttp. Por isso este scraper utiliza Selenium (Chrome headless)
para renderizar as páginas com JavaScript antes de extrair o conteúdo.

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
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from loguru import logger
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from crawler.scrapers import register_scraper
from crawler.scrapers.base import (
    BaseScraper,
    ScrapedDocument,
    DEFAULT_USER_AGENT,
    compute_canonical_id,
    ORIGINALS_DIR,
)

# ── Constantes ──────────────────────────────────────────────────────────────

_INDEX_URL = "https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac"
_RBAC_URL_RE = re.compile(
    r"https?://www\.anac\.gov\.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-[\w-]+$",
    re.IGNORECASE,
)

_RETRYABLE_EXCEPTIONS = (WebDriverException, OSError)
_RETRY_BASE_DELAY = 1.0
_MAX_RETRIES = 3

# Timeout para WebDriverWait aguardar elemento aparecer no DOM.
# O TSPD pode demorar até ~15s para resolver o desafio JS.
_ELEMENT_WAIT_TIMEOUT = 30.0

# Seletores CSS que indicam que a página de conteúdo carregou.
# Aguardamos qualquer um deles aparecer.
_READY_SELECTORS = ["#tabela-normas", "#content-core"]

_OUTPUT_DIR = ORIGINALS_DIR / "anac"


def _make_driver(page_timeout: int = 30) -> webdriver.Chrome:
    """Cria um Chrome headless com webdriver-manager."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(f"--user-agent={DEFAULT_USER_AGENT}")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options,
    )
    driver.set_page_load_timeout(page_timeout)
    return driver


@register_scraper
class ANACRBACscraper(BaseScraper):
    """Scraper para RBACs da ANAC.

    Usa Selenium (Chrome headless) para contornar o sistema anti-bot TSPD
    da ANAC, que bloqueia requisições HTTP diretas (aiohttp/requests).

    Notes:
        - Todas as navegações do Selenium são serializadas via threading.Lock
          para garantir thread-safety com driver único.
        - O método fetch_all() herdado de BaseScraper controla a concorrência
          via asyncio.Semaphore; as chamadas chegam ao Selenium de forma
          serializada pelo lock.
    """

    source_name = "anac_rbac"

    def __init__(self, timeout: int = 30, rate: float = 3.0):
        # rate mantido por compatibilidade de interface, não usado diretamente
        # (a taxa é limitada naturalmente pelo tempo de carregamento do Selenium)
        self._page_timeout = timeout
        self._driver: Optional[webdriver.Chrome] = None
        self._lock = threading.Lock()
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("ANACRBACscraper initialized (Selenium/Chrome mode)")

    def make_doc_id(self, doc: Dict) -> str:
        """ID estável e consistente com o retornado por fetch_document."""
        return f"anac_rbac_{doc['rbac_id']}"

    # ── Context manager ─────────────────────────────────────────────────────

    async def __aenter__(self):
        """Inicializa o driver Chrome headless em thread dedicada."""
        self._driver = await asyncio.to_thread(_make_driver, self._page_timeout)
        logger.debug("[anac_rbac] Chrome driver started")
        return self

    async def __aexit__(self, *_):
        """Encerra o driver Chrome."""
        if self._driver:
            await asyncio.to_thread(self._driver.quit)
            self._driver = None
            logger.debug("[anac_rbac] Chrome driver quit")

    # ── BaseScraper interface ────────────────────────────────────────────────

    async def search(self, *, limit: int = 0, **kwargs) -> List[Dict]:
        """Retorna lista de dicts com url, rbac_id e title de cada RBAC.

        Args:
            limit: Máximo de resultados (0 = todos).

        Returns:
            Lista de dicts: ``{"url": str, "rbac_id": str, "title": str}``
        """
        html = await asyncio.to_thread(self._get_sync, _INDEX_URL)
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
        return await asyncio.to_thread(self._fetch_document_sync, doc, save_original)

    # ── Sync implementation (executado em threads via asyncio.to_thread) ─────

    def _get_sync(self, url: str) -> str:
        """Carrega URL com Selenium e retorna o page_source.

        Aguarda ativamente (WebDriverWait) um dos ``_READY_SELECTORS`` aparecer
        no DOM antes de retornar, garantindo que o desafio TSPD já foi resolvido
        e o conteúdo real está carregado.

        Aplica retry com backoff exponencial para erros de carregamento.
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                with self._lock:
                    self._driver.get(url)
                    self._wait_for_content(url)
                    return self._driver.page_source
            except _RETRYABLE_EXCEPTIONS as exc:
                if attempt == _MAX_RETRIES:
                    logger.error(
                        f"[anac_rbac] Falha ao carregar {url} após {_MAX_RETRIES} tentativas: {exc}"
                    )
                    raise
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"[anac_rbac] Retry {attempt}/{_MAX_RETRIES} para {url}: {exc}"
                )
                time.sleep(delay)
        return ""  # nunca alcançado

    def _wait_for_content(self, url: str) -> None:
        """Aguarda um dos _READY_SELECTORS aparecer no DOM.

        O TSPD executa um desafio JavaScript antes de servir o conteúdo real;
        sem esse wait, o page_source pode ser capturado ainda na página de
        desafio, que não contém a tabela nem o conteúdo do RBAC.
        """
        wait = WebDriverWait(self._driver, _ELEMENT_WAIT_TIMEOUT)
        for selector in _READY_SELECTORS:
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                logger.debug(f"[anac_rbac] Elemento '{selector}' encontrado para {url}")
                return
            except TimeoutException:
                continue
        # Nenhum seletor encontrado — loga diagnóstico e continua com o que tiver
        logger.warning(
            f"[anac_rbac] Nenhum seletor de conteúdo encontrado após {_ELEMENT_WAIT_TIMEOUT}s "
            f"para {url}. Título: {self._driver.title!r}"
        )

    def _fetch_document_sync(
        self, doc: Dict, save_original: bool = True
    ) -> Optional[ScrapedDocument]:
        """Implementação síncrona de fetch_document para uso em thread."""
        url = doc["url"]
        rbac_id = doc["rbac_id"]
        title = doc.get("title", rbac_id.upper())

        logger.info(f"[anac_rbac] Fetching {rbac_id} — {url}")

        try:
            html = self._get_sync(url)
        except Exception as exc:
            logger.error(f"[anac_rbac] Error on {rbac_id}: {exc}")
            return None

        content = self._extract_content(html, rbac_id)
        if not content:
            logger.warning(f"[anac_rbac] Conteúdo vazio para {rbac_id}, pulando")
            return None

        if save_original:
            self._save_txt(rbac_id, content, url)

        # Extrai número (ex: "11" de "rbac-11", "e-94" de "rbac-e-94")
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

    # ── Helpers ─────────────────────────────────────────────────────────────

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
