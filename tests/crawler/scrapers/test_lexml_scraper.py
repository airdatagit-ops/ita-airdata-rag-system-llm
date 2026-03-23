"""Tests for crawler.scrapers.lexml_scraper (async)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from crawler.scrapers.lexml_scraper import LexMLScraper

# ── HTML fixtures ─────────────────────────────────────────────────────────────

SEARCH_HTML = """
<html><body>
  <div class="docHit">
    <table>
      <tr><td></td><td>Título:</td><td><a href="/urn/urn:lex:br:federal:lei:2001-01-01;10000">Lei 10000</a></td></tr>
      <tr><td></td><td>URN:</td><td>urn:lex:br:federal:lei:2001-01-01;10000</td></tr>
      <tr><td></td><td>Data:</td><td>01/01/2001</td></tr>
      <tr><td></td><td>Ementa:</td><td>Dispõe sobre aviação civil.</td></tr>
      <tr><td></td><td>Autoridade:</td><td>Federal</td></tr>
    </table>
  </div>
  <div class="docHit">
    <table>
      <tr><td></td><td>Título:</td><td><a href="/urn/urn:lex:br:federal:decreto:2002-06-15;4321">Decreto 4321</a></td></tr>
      <tr><td></td><td>URN:</td><td>urn:lex:br:federal:decreto:2002-06-15;4321</td></tr>
    </table>
  </div>
</body></html>
"""

LEXML_PAGE_HTML = """
<html><body>
  <a href="https://legis.senado.leg.br/norma/12345">Ver no Senado</a>
</body></html>
"""

SENADO_PAGE_HTML = """
<html><body>
  <a href="/publicacao/56789">Ver publicação</a>
</body></html>
"""

SENADO_PUB_HTML = """
<html><body>
  <div id="conteudoPrincipal">
    <p>LEI Nº 10.000, DE 1º DE JANEIRO DE 2001.</p>
    <p>Dispõe sobre a política de aviação civil no Brasil e dá outras providências.</p>
    <p>Art. 1º Esta Lei estabelece as normas gerais para a aviação civil brasileira.</p>
    <p>Art. 2º Compete à ANAC regulamentar e fiscalizar as atividades de aviação civil.</p>
  </div>
</body></html>
"""

PLANALTO_HTML = """
<html><body>
  <div id="textoNorma">
    <p>DECRETO Nº 4.321, DE 15 DE JUNHO DE 2002.</p>
    <p>Regulamenta a Lei nº 10.000, de 2001, que dispõe sobre a política de aviação civil no Brasil.</p>
    <p>Art. 1º Este Decreto regulamenta os procedimentos de aviação civil no território nacional.</p>
    <p>Art. 2º A Agência Nacional de Aviação Civil - ANAC é responsável pela fiscalização.</p>
    <p>Art. 3º As infrações a este Decreto sujeitam os infratores às penalidades previstas em lei.</p>
  </div>
</body></html>
"""

NO_CONTENT_HTML = "<html><body><p>Página não encontrada.</p></body></html>"


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_response(status: int = 200, text: str = "") -> MagicMock:
    """Build a minimal aiohttp response mock usable as async context manager."""
    resp = MagicMock()
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    resp.status = status
    resp.text = AsyncMock(return_value=text)
    resp.request_info = MagicMock()
    resp.history = []
    return resp


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
async def scraper():
    """Async scraper with duplicate-tracking disabled and a very high rate limit."""
    async with LexMLScraper(skip_duplicates=False, max_rate=1000) as s:
        yield s


# ── async context manager ─────────────────────────────────────────────────────


class TestAsyncContextManager:
    async def test_enter_creates_session(self):
        async with LexMLScraper(skip_duplicates=False) as s:
            assert s._session is not None

    async def test_exit_closes_session(self):
        async with LexMLScraper(skip_duplicates=False) as s:
            session = s._session
        assert session.closed or s._session is None

    async def test_double_use_works(self):
        s = LexMLScraper(skip_duplicates=False, max_rate=1000)
        async with s:
            assert s._session is not None
        assert s._session is None


# ── _parse_urn ────────────────────────────────────────────────────────────────


class TestParseUrn:
    @pytest.mark.parametrize("urn,expected", [
        (
            "urn:lex:br:federal:lei:1993-06-21;8666",
            {"authority": "federal", "doc_type": "lei", "publication_date": "1993-06-21", "number": "8666"},
        ),
        (
            "urn:lex:br:federal:decreto:2002-06-15;4321",
            {"authority": "federal", "doc_type": "decreto", "publication_date": "2002-06-15", "number": "4321"},
        ),
        (
            "urn:lex:br:federal:portaria:2020-03-10;1234",
            {"authority": "federal", "doc_type": "portaria", "publication_date": "2020-03-10", "number": "1234"},
        ),
    ])
    def test_parses_standard_urns(self, scraper, urn, expected):
        result = scraper._parse_urn(urn)
        for key, value in expected.items():
            assert result[key] == value

    def test_returns_empty_for_malformed_urn(self, scraper):
        assert scraper._parse_urn("not-a-valid-urn") == {}

    def test_returns_empty_for_short_urn(self, scraper):
        assert scraper._parse_urn("urn:lex:br") == {}


# ── _parse_search_results ─────────────────────────────────────────────────────


class TestParseSearchResults:
    def test_parses_all_results(self, scraper):
        docs = scraper._parse_search_results(SEARCH_HTML)
        assert len(docs) == 2

    def test_first_doc_fields(self, scraper):
        doc = scraper._parse_search_results(SEARCH_HTML)[0]
        assert doc["title"] == "Lei 10000"
        assert "federal" in doc["url"]
        assert doc["urn"] == "urn:lex:br:federal:lei:2001-01-01;10000"
        assert doc["date"] == "01/01/2001"
        assert doc["description"] == "Dispõe sobre aviação civil."
        # authority comes from _parse_urn (URN part[3]) which is lowercase "federal"
        assert doc["authority"] == "federal"

    def test_urn_parsed_into_metadata(self, scraper):
        doc = scraper._parse_search_results(SEARCH_HTML)[0]
        assert doc["doc_type"] == "lei"
        assert doc["number"] == "10000"
        assert doc["publication_date"] == "2001-01-01"

    def test_returns_empty_for_no_results(self, scraper):
        assert scraper._parse_search_results("<html><body></body></html>") == []

    def test_returns_empty_on_malformed_html(self, scraper):
        assert scraper._parse_search_results("not html at all <<<") == []


# ── _build_query ──────────────────────────────────────────────────────────────


class TestBuildQuery:
    def test_keywords_only(self, scraper):
        q = scraper._build_query(["aviação", "ANAC"], None)
        assert q == "keyword=aviação+ANAC"

    def test_with_doc_type(self, scraper):
        q = scraper._build_query(["aviação"], "Legislação")
        assert q == "keyword=aviação;f1-tipoDocumento=Legislação"

    def test_fallback_on_empty_keywords(self, scraper):
        q = scraper._build_query([], None)
        assert q == "lei federal"


# ── _get_html (retry) ─────────────────────────────────────────────────────────


class TestGetHtmlWithRetry:
    async def test_returns_html_on_200(self, scraper):
        resp = _make_response(200, "<html>ok</html>")
        scraper._session.get = MagicMock(return_value=resp)
        result = await scraper._get_html("http://example.com")
        assert result == "<html>ok</html>"

    async def test_returns_none_on_404(self, scraper):
        resp = _make_response(404)
        scraper._session.get = MagicMock(return_value=resp)
        result = await scraper._get_html("http://example.com")
        assert result is None

    async def test_retries_on_500_then_succeeds(self, scraper):
        fail = _make_response(500)
        ok = _make_response(200, "<html>ok</html>")
        scraper._session.get = MagicMock(side_effect=[fail, ok])

        with patch("crawler.scrapers.lexml_scraper.asyncio.sleep", new_callable=AsyncMock):
            result = await scraper._get_html("http://example.com")

        assert result == "<html>ok</html>"

    async def test_returns_none_after_all_retries_fail(self, scraper):
        fail = _make_response(500)
        scraper._session.get = MagicMock(side_effect=[fail, fail, fail])

        with patch("crawler.scrapers.lexml_scraper.asyncio.sleep", new_callable=AsyncMock):
            result = await scraper._get_html("http://example.com")

        assert result is None

    async def test_retries_on_timeout(self, scraper):
        scraper._session.get = MagicMock(side_effect=asyncio.TimeoutError)

        with patch("crawler.scrapers.lexml_scraper.asyncio.sleep", new_callable=AsyncMock):
            result = await scraper._get_html("http://example.com")

        assert result is None


# ── search ────────────────────────────────────────────────────────────────────


class TestSearch:
    async def test_returns_documents(self, scraper):
        results = _make_response(200, SEARCH_HTML)
        empty = _make_response(200, "<html><body></body></html>")
        scraper._session.get = MagicMock(side_effect=[results, empty])
        docs = await scraper.search(keywords=["aviação"], limit=10)
        assert len(docs) == 2

    async def test_respects_limit(self, scraper):
        resp = _make_response(200, SEARCH_HTML)
        # second page returns nothing to stop pagination
        empty = _make_response(200, "<html><body></body></html>")
        scraper._session.get = MagicMock(side_effect=[resp, empty])
        docs = await scraper.search(keywords=["aviação"], limit=1)
        assert len(docs) == 1

    async def test_stops_on_empty_page(self, scraper):
        empty = _make_response(200, "<html><body></body></html>")
        scraper._session.get = MagicMock(return_value=empty)
        docs = await scraper.search(keywords=["aviação"], limit=50)
        assert docs == []

    async def test_stops_on_http_error(self, scraper):
        scraper._session.get = MagicMock(return_value=_make_response(404))
        docs = await scraper.search(keywords=["aviação"], limit=50)
        assert docs == []


# ── get_document_text ─────────────────────────────────────────────────────────


class TestGetDocumentText:
    async def test_follows_senado_link_and_extracts_text(self, scraper):
        responses = [
            _make_response(200, LEXML_PAGE_HTML),   # LexML page
            _make_response(200, SENADO_PAGE_HTML),   # Senado norma page
            _make_response(200, SENADO_PUB_HTML),    # Senado publication
        ]
        scraper._session.get = MagicMock(side_effect=responses)
        doc = {"url": "http://lexml.gov.br/urn/test", "title": "Lei 10000"}
        text = await scraper.get_document_text(doc, save_original=False)
        assert text is not None
        assert "LEI" in text
        assert len(text) > 50

    async def test_follows_planalto_link(self, scraper):
        lexml_html = """<html><body>
            <a href="https://www.planalto.gov.br/legislacao/decreto4321">Planalto</a>
        </body></html>"""
        responses = [
            _make_response(200, lexml_html),
            _make_response(200, PLANALTO_HTML),
        ]
        scraper._session.get = MagicMock(side_effect=responses)
        doc = {"url": "http://lexml.gov.br/urn/test", "title": "Decreto 4321"}
        text = await scraper.get_document_text(doc, save_original=False)
        assert text is not None
        assert "DECRETO" in text

    async def test_falls_back_to_description_when_no_source_link(self, scraper):
        no_links_html = "<html><body><p>Sem links.</p></body></html>"
        scraper._session.get = MagicMock(return_value=_make_response(200, no_links_html))
        doc = {"url": "http://lexml.gov.br/urn/test", "description": "Ementa do documento"}
        text = await scraper.get_document_text(doc, save_original=False)
        assert text == "Ementa do documento"

    async def test_falls_back_to_description_on_http_error(self, scraper):
        scraper._session.get = MagicMock(return_value=_make_response(500))
        with patch("crawler.scrapers.lexml_scraper.asyncio.sleep", new_callable=AsyncMock):
            doc = {"url": "http://lexml.gov.br/urn/test", "description": "Fallback"}
            text = await scraper.get_document_text(doc, save_original=False)
        assert text == "Fallback"

    async def test_returns_none_when_no_url_or_urn(self, scraper):
        doc = {"title": "Documento sem URL"}
        text = await scraper.get_document_text(doc, save_original=False)
        assert text is None


# ── parallel downloads ────────────────────────────────────────────────────────


class TestParallelDownloads:
    async def test_gather_respects_semaphore(self, scraper):
        """Semaphore with limit=1 should serialize coroutines without deadlock."""
        call_order = []

        async def mock_get_text(doc, save_original=True):
            call_order.append(doc["title"])
            await asyncio.sleep(0)
            return f"content of {doc['title']}"

        scraper.get_document_text = mock_get_text
        docs = [{"title": f"Doc {i}"} for i in range(5)]
        sem = asyncio.Semaphore(1)

        async def _one(doc):
            async with sem:
                return await scraper.get_document_text(doc)

        results = await asyncio.gather(*[_one(d) for d in docs])
        assert len(results) == 5
        assert all(r is not None for r in results)

    async def test_gather_handles_individual_failures(self, scraper):
        """One failing download should not cancel others."""
        call_count = 0

        async def mock_get_text(doc, save_original=True):
            nonlocal call_count
            call_count += 1
            if doc.get("fail"):
                raise aiohttp.ClientError("simulated failure")
            return "content"

        scraper.get_document_text = mock_get_text
        docs = [{"title": "ok"}, {"title": "fail", "fail": True}, {"title": "ok2"}]

        results = await asyncio.gather(
            *[scraper.get_document_text(d) for d in docs],
            return_exceptions=True,
        )
        assert call_count == 3
        assert results[0] == "content"
        assert isinstance(results[1], aiohttp.ClientError)
        assert results[2] == "content"
