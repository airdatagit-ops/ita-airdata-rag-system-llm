"""Tests for crawler.scrapers.decea_scraper."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crawler.scrapers.decea_scraper import DECEAScraper, _DOC_TYPE_RE
from crawler.scrapers.base import BaseScraper, ScrapedDocument


INDEX_HTML = """<html><body>
<table>
<tr><th>NÚMERO</th><th>TÍTULO</th><th>EM VIGOR DESDE</th><th>ORIGEM</th></tr>
<tr><td>ICA96-1</td><td>Cartas aeronáuticas</td><td>30/04/2025</td><td>SDOP</td></tr>
<tr><td>ICA100-47</td><td>Habilitação instrutores</td><td>05/05/2025</td><td>SDOP</td></tr>
<tr><td>MCA53-11</td><td>Manual telecom</td><td>01/01/2025</td><td>SDTE</td></tr>
</table>
</body></html>"""

PUB_HTML = (
    r'<a href="https://static.decea.mil.br/files/2025/ica-96-1.pdf'
    r'?X-Amz-Content-Sha256=UNSIGNED\u0026X-Amz-Expires=900">PDF</a>'
)


@pytest.fixture
def scraper():
    s = DECEAScraper()
    s.session = MagicMock()
    return s


# ── inherits BaseScraper ────────────────────────────────────


class TestInheritance:
    def test_is_base_scraper(self, scraper):
        assert isinstance(scraper, BaseScraper)

    def test_source_name(self, scraper):
        assert scraper.source_name == "decea"

    def test_make_doc_id_with_slug(self, scraper):
        assert scraper.make_doc_id({"slug": "ICA-96-1"}) == "decea_ICA-96-1"

    def test_make_doc_id_fallback(self, scraper):
        assert scraper.make_doc_id({"title": "Some Title"}) == "Some_Title"


# ── slug & regex ────────────────────────────────────────────


class TestBuildSlug:
    @pytest.mark.parametrize("numero,doc_type,expected", [
        ("ICA96-1", "ICA", "ICA-96-1"),
        ("CIRCEA100-123", "CIRCEA", "CIRCEA-100-123"),
        ("DCA205-7", "DCA", "DCA-205-7"),
        ("AIC-A07/26", "AIC-A", "AIC-A-07/26"),
        ("MCA53-11", "MCA", "MCA-53-11"),
    ])
    def test_generates_slug(self, numero, doc_type, expected):
        assert DECEAScraper._build_slug(numero, doc_type) == expected


class TestDocTypeRegex:
    @pytest.mark.parametrize("text,expected_type,expected_num", [
        ("ICA96-1", "ICA", "96-1"),
        ("AIC-A07/26", "AIC-A", "07/26"),
        ("CIRCEA100-123", "CIRCEA", "100-123"),
        ("MCA53-11", "MCA", "53-11"),
        ("NSCA5-7", "NSCA", "5-7"),
    ])
    def test_matches(self, text, expected_type, expected_num):
        m = _DOC_TYPE_RE.match(text)
        assert m is not None
        assert m.group(1) == expected_type
        assert m.group(2) == expected_num

    def test_rejects_pure_text(self):
        assert _DOC_TYPE_RE.match("TÍTULO") is None

    def test_rejects_pure_number(self):
        assert _DOC_TYPE_RE.match("12345") is None


# ── index parsing ───────────────────────────────────────────


class TestParseIndex:
    def test_parses_all_rows(self, scraper):
        docs = scraper._parse_index_html(INDEX_HTML)
        assert len(docs) == 3

    def test_ica_fields(self, scraper):
        doc = scraper._parse_index_html(INDEX_HTML)[0]
        assert doc["slug"] == "ICA-96-1"
        assert doc["type"] == "ICA"
        assert doc["number"] == "ICA96-1"
        assert doc["title"] == "Cartas aeronáuticas"
        assert doc["date_published"] == "30/04/2025"
        assert doc["doc_type"] == "ica"

    def test_mca_type(self, scraper):
        doc = scraper._parse_index_html(INDEX_HTML)[2]
        assert doc["type"] == "MCA"
        assert doc["slug"] == "MCA-53-11"

    def test_skips_header_rows(self, scraper):
        html = "<table><tr><th>NÚMERO</th><th>TÍTULO</th><th>D</th><th>O</th></tr></table>"
        assert scraper._parse_index_html(html) == []

    def test_skips_short_rows(self, scraper):
        html = "<table><tr><td>ICA96-1</td><td>Title</td></tr></table>"
        assert scraper._parse_index_html(html) == []

    def test_empty_html(self, scraper):
        assert scraper._parse_index_html("<html></html>") == []


# ── sync search ─────────────────────────────────────────────


class TestSearch:
    def _mock_index(self, scraper):
        scraper.get_publications_index = MagicMock(return_value=[
            {"type": "ICA", "title": "Cartas aeronáuticas", "slug": "ICA-96-1"},
            {"type": "ICA", "title": "Habilitação técnica", "slug": "ICA-100-47"},
            {"type": "MCA", "title": "Manual telecom", "slug": "MCA-53-11"},
        ])

    def test_filters_by_type(self, scraper):
        self._mock_index(scraper)
        result = scraper._search_sync(doc_types=["ICA"])
        assert len(result) == 2
        assert all(d["type"] == "ICA" for d in result)

    def test_filters_by_keyword(self, scraper):
        self._mock_index(scraper)
        result = scraper._search_sync(doc_types=["ICA"], keywords=["cartas"])
        assert len(result) == 1
        assert result[0]["slug"] == "ICA-96-1"

    def test_respects_limit(self, scraper):
        scraper.get_publications_index = MagicMock(return_value=[
            {"type": "ICA", "title": f"Doc {i}", "slug": f"ICA-{i}"}
            for i in range(50)
        ])
        assert len(scraper._search_sync(limit=5)) == 5

    def test_default_type_is_ica(self, scraper):
        self._mock_index(scraper)
        result = scraper._search_sync()
        assert all(d["type"] == "ICA" for d in result)


# ── async search ────────────────────────────────────────────


class TestAsyncSearch:
    async def test_delegates_to_sync(self, scraper):
        scraper.get_publications_index = MagicMock(return_value=[
            {"type": "ICA", "title": "Doc 1", "slug": "ICA-1"},
        ])
        result = await scraper.search(limit=10, doc_types=["ICA"])
        assert len(result) == 1


# ── PDF URL extraction ──────────────────────────────────────


class TestGetPdfUrl:
    def test_extracts_and_unescapes(self, scraper):
        resp = MagicMock()
        resp.text = PUB_HTML
        scraper.session.get.return_value = resp

        url = scraper.get_pdf_url("ICA-96-1")
        assert url is not None
        assert "static.decea.mil.br" in url
        assert "\\u0026" not in url
        assert "&" in url

    def test_returns_none_when_no_pdf(self, scraper):
        resp = MagicMock()
        resp.text = "<html>No PDF here</html>"
        scraper.session.get.return_value = resp
        assert scraper.get_pdf_url("INVALID") is None

    def test_handles_request_error(self, scraper):
        import requests as req
        scraper.session.get.side_effect = req.RequestException("timeout")
        assert scraper.get_pdf_url("ICA-96-1") is None


# ── download ────────────────────────────────────────────────


class TestDownloadPdf:
    def test_valid_pdf(self, scraper):
        resp = MagicMock(ok=True, content=b"%PDF-1.4 data")
        resp.headers = {"Content-Type": "application/pdf"}
        scraper.session.get.return_value = resp
        assert scraper.download_pdf("http://x/f.pdf") == b"%PDF-1.4 data"

    def test_rejects_non_pdf(self, scraper):
        resp = MagicMock(ok=True, content=b"<html>")
        resp.headers = {"Content-Type": "text/html"}
        scraper.session.get.return_value = resp
        assert scraper.download_pdf("http://x/f.pdf") is None

    def test_handles_http_error(self, scraper):
        resp = MagicMock(ok=False, content=b"", status_code=404)
        resp.headers = {}
        scraper.session.get.return_value = resp
        assert scraper.download_pdf("http://x/f.pdf") is None


# ── fetch_document (async) ──────────────────────────────────


class TestFetchDocument:
    async def test_returns_scraped_document(self, scraper):
        with patch.object(scraper, "get_document_text", return_value="A" * 200):
            doc = {"slug": "ICA-1", "title": "Test", "doc_type": "ica"}
            result = await scraper.fetch_document(doc, save_original=False)
            assert isinstance(result, ScrapedDocument)
            assert result.doc_id == "decea_ICA-1"
            assert result.source == "decea"
            assert result.content == "A" * 200

    async def test_returns_none_on_no_content(self, scraper):
        with patch.object(scraper, "get_document_text", return_value=None):
            result = await scraper.fetch_document({"slug": "X", "title": "T"})
            assert result is None


# ── fetch_all (parallel via BaseScraper) ─────────────────────


class TestFetchAll:
    async def test_fetches_all_documents(self, scraper):
        with patch.object(scraper, "get_document_text", return_value="text " * 50):
            docs = [
                {"slug": f"ICA-{i}", "title": f"D{i}", "doc_type": "ica"}
                for i in range(5)
            ]
            results = await scraper.fetch_all(docs, concurrency=2)
            assert len(results) == 5
            assert all(isinstance(r, ScrapedDocument) for r in results)

    async def test_handles_errors_gracefully(self, scraper):
        with patch.object(scraper, "get_document_text", side_effect=Exception("fail")):
            docs = [{"slug": "ICA-1", "title": "Fallback", "doc_type": "ica"}]
            results = await scraper.fetch_all(docs, concurrency=1)
            assert len(results) == 0
