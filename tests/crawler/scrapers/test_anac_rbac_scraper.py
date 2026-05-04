"""Tests for crawler.scrapers.anac_rbac_scraper."""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from crawler.scrapers.anac_rbac_scraper import ANACRBACscraper, _RBAC_URL_RE, _ARQUIVO_NORMA_RE
from crawler.scrapers.base import BaseScraper, ScrapedDocument


# ── HTML fixtures ─────────────────────────────────────────────────────────────

INDEX_HTML_HTML_LINKS = """
<html><body>
<table id="tabela-normas">
  <tbody>
    <tr>
      <td class="tbNumero">RBAC 11</td>
      <td><a href="https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-11">RBAC 11</a></td>
    </tr>
    <tr>
      <td class="tbNumero">RBAC 21</td>
      <td><a href="https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-21">RBAC 21</a></td>
    </tr>
    <tr>
      <td class="tbNumero">RBAC 91</td>
      <td><a href="https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-91">RBAC 91</a></td>
    </tr>
  </tbody>
</table>
</body></html>
"""

INDEX_HTML_PDF_LINKS = """
<html><body>
<table id="tabela-normas">
  <tbody>
    <tr>
      <td class="tbNumero">RBAC 11</td>
      <td>
        <a href="/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-11/@@display-file/arquivo_norma/RBAC11.pdf">PDF</a>
      </td>
    </tr>
    <tr>
      <td class="tbNumero">RBAC 121</td>
      <td>
        <a href="/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-121/@@display-file/arquivo_norma/RBAC121.pdf">PDF</a>
      </td>
    </tr>
  </tbody>
</table>
</body></html>
"""

INDEX_HTML_MIXED = """
<html><body>
<table id="tabela-normas">
  <tbody>
    <tr>
      <td class="tbNumero">RBAC 11</td>
      <td>
        <a href="https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-11">HTML</a>
        <a href="/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-11/@@display-file/arquivo_norma/RBAC11.pdf">PDF</a>
      </td>
    </tr>
    <tr>
      <td class="tbNumero">RBAC 91</td>
      <td>
        <a href="https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-91">HTML</a>
      </td>
    </tr>
  </tbody>
</table>
</body></html>
"""

INDEX_HTML_NO_TABLE = """
<html><body>
  <p>Nenhuma tabela encontrada.</p>
</body></html>
"""

INDEX_HTML_NO_TBODY = """
<html><body>
<table id="tabela-normas">
  <tr><td class="tbNumero">RBAC 11</td><td><a href="https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-11">link</a></td></tr>
</table>
</body></html>
"""

INDEX_HTML_DUPLICATE_URLS = """
<html><body>
<table id="tabela-normas">
  <tbody>
    <tr>
      <td class="tbNumero">RBAC 11</td>
      <td><a href="https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-11">link</a></td>
    </tr>
    <tr>
      <td class="tbNumero">RBAC 11 (duplicado)</td>
      <td><a href="https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-11">link duplicado</a></td>
    </tr>
  </tbody>
</table>
</body></html>
"""

RBAC_CONTENT_CORE_HTML = """
<html><body>
  <div id="content-core">
    <h1>RBAC Nº 11</h1>
    <p>Requisitos Gerais para Aeronaves Civis.</p>
    <p>Art. 1º Este Regulamento estabelece os requisitos gerais para aeronaves.</p>
    <p>Art. 2º Todas as aeronaves devem cumprir os requisitos estabelecidos.</p>
  </div>
</body></html>
"""

RBAC_CONTENT_FALLBACK_HTML = """
<html><body>
  <div id="content">
    <h1>RBAC Nº 21</h1>
    <p>Certificação de Produtos Aeronáuticos.</p>
  </div>
</body></html>
"""

RBAC_NO_CONTENT_HTML = """
<html><body>
  <p>Página sem conteúdo estruturado.</p>
</body></html>
"""

RBAC_EMPTY_CONTENT_HTML = """
<html><body>
  <div id="content-core">
  </div>
</body></html>
"""

RBAC_MULTILINE_CONTENT_HTML = """
<html><body>
  <div id="content-core">
    <p>Linha 1</p>



    <p>Linha 2</p>
  </div>
</body></html>
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_async_response(status: int = 200, text: str = "", content: bytes = b"") -> MagicMock:
    resp = MagicMock()
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    resp.status = status
    resp.text = AsyncMock(return_value=text)
    resp.read = AsyncMock(return_value=content)
    resp.raise_for_status = MagicMock()
    resp.request_info = MagicMock()
    resp.history = []
    return resp


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def scraper():
    s = ANACRBACscraper()
    s._session = MagicMock()
    return s


@pytest.fixture
async def async_scraper():
    async with ANACRBACscraper() as s:
        yield s


# ── herança de BaseScraper ─────────────────────────────────────────────────────

class TestInheritance:
    def test_is_base_scraper(self, scraper):
        assert isinstance(scraper, BaseScraper)

    def test_source_name(self, scraper):
        assert scraper.source_name == "anac_rbac"

    def test_make_doc_id(self, scraper):
        assert scraper.make_doc_id({"rbac_id": "rbac-11"}) == "anac_rbac_rbac-11"

    def test_make_doc_id_e_suffix(self, scraper):
        assert scraper.make_doc_id({"rbac_id": "rbac-e-94"}) == "anac_rbac_rbac-e-94"


# ── context manager assíncrono ────────────────────────────────────────────────

class TestAsyncContextManager:
    async def test_enter_creates_session(self):
        async with ANACRBACscraper() as s:
            assert s._session is not None
            assert isinstance(s._session, aiohttp.ClientSession)

    async def test_exit_closes_session(self):
        async with ANACRBACscraper() as s:
            session = s._session
        assert session.closed

    async def test_session_is_none_before_enter(self):
        s = ANACRBACscraper()
        assert s._session is None


# ── expressões regulares ──────────────────────────────────────────────────────

class TestRbacUrlRegex:
    @pytest.mark.parametrize("url", [
        "https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-11",
        "https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-91",
        "https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-e-94",
        "https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-121",
        "http://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-11",
    ])
    def test_matches_valid_rbac_urls(self, url):
        assert _RBAC_URL_RE.search(url) is not None

    @pytest.mark.parametrize("url", [
        "https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/",
        "https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-11/subarticle",
        "https://www.google.com/rbac/rbac-11",
        "https://www.anac.gov.br/rbac/rbac-11",
        "https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-11?visao=tabela",
    ])
    def test_rejects_invalid_urls(self, url):
        assert _RBAC_URL_RE.search(url) is None


class TestArquivaNormaRegex:
    @pytest.mark.parametrize("path,expected_group", [
        (
            "/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-11/@@display-file/arquivo_norma/RBAC11.pdf",
            "rbac-11",
        ),
        (
            "/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-121/@@display-file/arquivo_norma/RBAC121EMD1.pdf",
            "rbac-121",
        ),
        (
            "/rbac/rbac-e-94/@@display-file/arquivo_norma/RBACE94.pdf",
            "rbac-e-94",
        ),
    ])
    def test_matches_and_captures_slug(self, path, expected_group):
        m = _ARQUIVO_NORMA_RE.search(path)
        assert m is not None
        assert m.group(1) == expected_group

    def test_rejects_html_links(self):
        url = "https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-11"
        assert _ARQUIVO_NORMA_RE.search(url) is None

    def test_rejects_non_pdf(self):
        path = "/rbac/rbac-11/@@display-file/arquivo_norma/RBAC11.docx"
        assert _ARQUIVO_NORMA_RE.search(path) is None


# ── _extract_links ────────────────────────────────────────────────────────────

class TestExtractLinks:
    def test_parses_html_links(self, scraper):
        docs = scraper._extract_links(INDEX_HTML_HTML_LINKS)
        assert len(docs) == 3

    def test_html_link_fields(self, scraper):
        docs = scraper._extract_links(INDEX_HTML_HTML_LINKS)
        doc = docs[0]
        assert doc["rbac_id"] == "rbac-11"
        assert doc["url"] == "https://www.anac.gov.br/assuntos/legislacao/legislacao-1/rbha-e-rbac/rbac/rbac-11"
        assert doc["title"] == "RBAC 11"
        assert doc["pdf_direct"] is False

    def test_parses_pdf_direct_links(self, scraper):
        docs = scraper._extract_links(INDEX_HTML_PDF_LINKS)
        assert len(docs) == 2

    def test_pdf_direct_link_fields(self, scraper):
        docs = scraper._extract_links(INDEX_HTML_PDF_LINKS)
        doc = docs[0]
        assert doc["rbac_id"] == "rbac-11"
        assert doc["pdf_direct"] is True
        assert "@@display-file/arquivo_norma" in doc["url"]
        assert doc["url"].startswith("https://www.anac.gov.br")

    def test_mixed_prefers_pdf_over_html(self, scraper):
        """Quando a linha tem ambos os links, PDF direto tem prioridade."""
        docs = scraper._extract_links(INDEX_HTML_MIXED)
        rbac_11 = next(d for d in docs if d["rbac_id"] == "rbac-11")
        assert rbac_11["pdf_direct"] is True

    def test_mixed_html_fallback_when_no_pdf(self, scraper):
        """Quando a linha só tem link HTML, pdf_direct deve ser False."""
        docs = scraper._extract_links(INDEX_HTML_MIXED)
        rbac_91 = next(d for d in docs if d["rbac_id"] == "rbac-91")
        assert rbac_91["pdf_direct"] is False

    def test_returns_empty_when_no_table(self, scraper):
        docs = scraper._extract_links(INDEX_HTML_NO_TABLE)
        assert docs == []

    def test_returns_empty_when_no_tbody(self, scraper):
        """Sem <tbody> explícito, a lista deve ser vazia."""
        docs = scraper._extract_links(INDEX_HTML_NO_TBODY)
        assert docs == []

    def test_deduplicates_same_url(self, scraper):
        docs = scraper._extract_links(INDEX_HTML_DUPLICATE_URLS)
        assert len(docs) == 1
        assert docs[0]["rbac_id"] == "rbac-11"

    def test_returns_empty_on_empty_html(self, scraper):
        docs = scraper._extract_links("<html></html>")
        assert docs == []


# ── _extract_content ──────────────────────────────────────────────────────────

class TestExtractContent:
    def test_extracts_from_content_core(self, scraper):
        text = scraper._extract_content(RBAC_CONTENT_CORE_HTML, "rbac-11")
        assert "RBAC Nº 11" in text
        assert "Art. 1º" in text

    def test_fallback_to_content_div(self, scraper):
        text = scraper._extract_content(RBAC_CONTENT_FALLBACK_HTML, "rbac-21")
        assert "RBAC Nº 21" in text
        assert "Certificação" in text

    def test_returns_empty_when_no_div(self, scraper):
        text = scraper._extract_content(RBAC_NO_CONTENT_HTML, "rbac-99")
        assert text == ""

    def test_returns_empty_when_content_is_blank(self, scraper):
        text = scraper._extract_content(RBAC_EMPTY_CONTENT_HTML, "rbac-11")
        assert text == ""

    def test_normalizes_multiple_blank_lines(self, scraper):
        text = scraper._extract_content(RBAC_MULTILINE_CONTENT_HTML, "rbac-11")
        assert "\n\n\n" not in text

    def test_returns_empty_on_empty_html(self, scraper):
        text = scraper._extract_content("<html></html>", "rbac-11")
        assert text == ""


# ── search ────────────────────────────────────────────────────────────────────

class TestSearch:
    async def test_returns_list_of_dicts(self, scraper):
        with patch.object(scraper, "_get", AsyncMock(return_value=INDEX_HTML_HTML_LINKS)):
            docs = await scraper.search()
        assert len(docs) == 3
        assert all(isinstance(d, dict) for d in docs)

    async def test_respects_limit(self, scraper):
        with patch.object(scraper, "_get", AsyncMock(return_value=INDEX_HTML_HTML_LINKS)):
            docs = await scraper.search(limit=2)
        assert len(docs) == 2

    async def test_limit_zero_returns_all(self, scraper):
        with patch.object(scraper, "_get", AsyncMock(return_value=INDEX_HTML_HTML_LINKS)):
            docs = await scraper.search(limit=0)
        assert len(docs) == 3

    async def test_no_table_returns_empty(self, scraper):
        with patch.object(scraper, "_get", AsyncMock(return_value=INDEX_HTML_NO_TABLE)):
            docs = await scraper.search()
        assert docs == []


# ── fetch_document (HTML) ──────────────────────────────────────────────────────

class TestFetchDocumentHtml:
    async def test_returns_scraped_document(self, scraper):
        doc = {"url": "https://www.anac.gov.br/rbac/rbac-11", "rbac_id": "rbac-11", "title": "RBAC 11", "pdf_direct": False}
        with patch.object(scraper, "_get", AsyncMock(return_value=RBAC_CONTENT_CORE_HTML)):
            result = await scraper.fetch_document(doc, save_original=False)
        assert isinstance(result, ScrapedDocument)
        assert result.doc_id == "anac_rbac_rbac-11"
        assert result.source == "anac_rbac"
        assert result.title == "RBAC 11"
        assert result.authority == "ANAC"
        assert result.doc_type == "RBAC"
        assert result.number == "11"
        assert "RBAC Nº 11" in result.content

    async def test_returns_none_on_empty_content(self, scraper):
        doc = {"url": "https://www.anac.gov.br/rbac/rbac-11", "rbac_id": "rbac-11", "title": "RBAC 11", "pdf_direct": False}
        with patch.object(scraper, "_get", AsyncMock(return_value=RBAC_NO_CONTENT_HTML)):
            result = await scraper.fetch_document(doc, save_original=False)
        assert result is None

    async def test_returns_none_on_exception(self, scraper):
        doc = {"url": "https://www.anac.gov.br/rbac/rbac-11", "rbac_id": "rbac-11", "title": "RBAC 11", "pdf_direct": False}
        with patch.object(scraper, "_get", AsyncMock(side_effect=aiohttp.ClientError("timeout"))):
            result = await scraper.fetch_document(doc, save_original=False)
        assert result is None

    async def test_canonical_id_set(self, scraper):
        doc = {"url": "https://www.anac.gov.br/rbac/rbac-11", "rbac_id": "rbac-11", "title": "RBAC 11", "pdf_direct": False}
        with patch.object(scraper, "_get", AsyncMock(return_value=RBAC_CONTENT_CORE_HTML)):
            result = await scraper.fetch_document(doc, save_original=False)
        assert result is not None
        assert result.canonical_id == "rbac_11"

    async def test_uses_rbac_id_as_title_fallback(self, scraper):
        doc = {"url": "https://www.anac.gov.br/rbac/rbac-11", "rbac_id": "rbac-11", "pdf_direct": False}
        with patch.object(scraper, "_get", AsyncMock(return_value=RBAC_CONTENT_CORE_HTML)):
            result = await scraper.fetch_document(doc, save_original=False)
        assert result is not None
        assert result.title == "RBAC-11"


# ── fetch_document (PDF direto) ───────────────────────────────────────────────

class TestFetchDocumentPdf:
    async def test_returns_scraped_document_from_pdf(self, scraper):
        doc = {
            "url": "https://www.anac.gov.br/rbac/rbac-11/@@display-file/arquivo_norma/RBAC11.pdf",
            "rbac_id": "rbac-11",
            "title": "RBAC 11",
            "pdf_direct": True,
        }
        fake_text = "RBAC 11\nConteúdo completo do regulamento. " * 20
        with (
            patch.object(scraper, "_get_bytes", AsyncMock(return_value=b"%PDF-1.4 fake content")),
            patch("crawler.scrapers.anac_rbac_scraper.extract_text_from_bytes", return_value=fake_text),
        ):
            result = await scraper.fetch_document(doc, save_original=False)
        assert isinstance(result, ScrapedDocument)
        assert result.doc_id == "anac_rbac_rbac-11"
        assert result.source == "anac_rbac"
        assert result.doc_type == "RBAC"

    async def test_returns_none_on_invalid_pdf_magic_bytes(self, scraper):
        doc = {
            "url": "https://www.anac.gov.br/rbac/rbac-11/@@display-file/arquivo_norma/RBAC11.pdf",
            "rbac_id": "rbac-11",
            "title": "RBAC 11",
            "pdf_direct": True,
        }
        with patch.object(scraper, "_get_bytes", AsyncMock(return_value=b"<html>not a pdf</html>")):
            result = await scraper.fetch_document(doc, save_original=False)
        assert result is None

    async def test_returns_none_on_empty_bytes(self, scraper):
        doc = {
            "url": "https://www.anac.gov.br/rbac/rbac-11/@@display-file/arquivo_norma/RBAC11.pdf",
            "rbac_id": "rbac-11",
            "title": "RBAC 11",
            "pdf_direct": True,
        }
        with patch.object(scraper, "_get_bytes", AsyncMock(return_value=b"")):
            result = await scraper.fetch_document(doc, save_original=False)
        assert result is None

    async def test_returns_none_when_pdf_has_no_extractable_text(self, scraper):
        doc = {
            "url": "https://www.anac.gov.br/rbac/rbac-11/@@display-file/arquivo_norma/RBAC11.pdf",
            "rbac_id": "rbac-11",
            "title": "RBAC 11",
            "pdf_direct": True,
        }
        with (
            patch.object(scraper, "_get_bytes", AsyncMock(return_value=b"%PDF-1.4 scanned")),
            patch("crawler.scrapers.anac_rbac_scraper.extract_text_from_bytes", return_value=""),
        ):
            result = await scraper.fetch_document(doc, save_original=False)
        assert result is None

    async def test_returns_none_on_exception(self, scraper):
        doc = {
            "url": "https://www.anac.gov.br/rbac/rbac-11/@@display-file/arquivo_norma/RBAC11.pdf",
            "rbac_id": "rbac-11",
            "title": "RBAC 11",
            "pdf_direct": True,
        }
        with patch.object(scraper, "_get_bytes", AsyncMock(side_effect=aiohttp.ClientError("network error"))):
            result = await scraper.fetch_document(doc, save_original=False)
        assert result is None


# ── retry e backoff ───────────────────────────────────────────────────────────

class TestGetRetry:
    async def test_retries_on_retryable_status(self, scraper):
        """Deve retentar em status 503 e ter sucesso na segunda chamada."""
        ok_resp = _make_async_response(status=200, text="<html>ok</html>")
        err_resp = _make_async_response(status=503)
        err_resp.raise_for_status = MagicMock()

        call_count = 0

        async def fake_get(url):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                error = aiohttp.ClientResponseError(MagicMock(), [], status=503)
                raise error
            return ok_resp

        with (
            patch.object(scraper._limiter, "acquire", AsyncMock()),
            patch("asyncio.sleep", AsyncMock()),
        ):
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(side_effect=fake_get)
            ctx.__aexit__ = AsyncMock(return_value=False)

            # Testar via mock do session.get direto
            responses = [
                _make_async_response(status=503),
                _make_async_response(status=200, text="conteudo ok"),
            ]
            responses[0].raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(MagicMock(), [], status=503))
            scraper._session.get.side_effect = iter(responses)

            with patch("asyncio.sleep", AsyncMock()):
                text = await scraper._get("http://example.com")
            assert text == "conteudo ok"

    async def test_raises_after_max_retries(self, scraper):
        """Deve levantar exceção após esgotar todas as tentativas."""
        resp = _make_async_response(status=503)
        resp.raise_for_status = MagicMock(
            side_effect=aiohttp.ClientResponseError(MagicMock(), [], status=503)
        )
        scraper._session.get.return_value = resp

        with (
            patch("asyncio.sleep", AsyncMock()),
            pytest.raises(aiohttp.ClientResponseError),
        ):
            await scraper._get("http://example.com")


# ── fetch_all (via BaseScraper) ───────────────────────────────────────────────

class TestFetchAll:
    async def test_fetches_multiple_documents(self, scraper):
        fake_text = "Conteúdo do RBAC. " * 30
        docs = [
            {"url": f"https://anac.gov.br/rbac/rbac-{n}", "rbac_id": f"rbac-{n}", "title": f"RBAC {n}", "pdf_direct": False}
            for n in [11, 21, 91]
        ]
        with patch.object(scraper, "_get", AsyncMock(return_value=RBAC_CONTENT_CORE_HTML)):
            results = await scraper.fetch_all(docs, concurrency=2, save_original=False)
        assert len(results) == 3
        assert all(isinstance(r, ScrapedDocument) for r in results)

    async def test_skips_failed_documents(self, scraper):
        docs = [
            {"url": "https://anac.gov.br/rbac/rbac-11", "rbac_id": "rbac-11", "title": "RBAC 11", "pdf_direct": False},
            {"url": "https://anac.gov.br/rbac/rbac-99", "rbac_id": "rbac-99", "title": "RBAC 99", "pdf_direct": False},
        ]

        async def side_effect(doc, save_original=True):
            if doc["rbac_id"] == "rbac-99":
                return None
            return ScrapedDocument(
                doc_id="anac_rbac_rbac-11",
                source="anac_rbac",
                title="RBAC 11",
                content="conteudo " * 20,
            )

        with patch.object(scraper, "fetch_document", side_effect=side_effect):
            results = await scraper.fetch_all(docs, concurrency=2)
        assert len(results) == 1
        assert results[0].doc_id == "anac_rbac_rbac-11"
