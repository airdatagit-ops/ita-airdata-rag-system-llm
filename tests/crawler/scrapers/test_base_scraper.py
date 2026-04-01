"""Tests for crawler.scrapers.base and crawler.scrapers registry."""

import asyncio
from typing import Dict, List, Optional

import pytest

from crawler.scrapers.base import BaseScraper, ScrapedDocument
from crawler.scrapers import get_scraper, list_scrapers, _REGISTRY


# ── concrete test scraper ────────────────────────────────────────────────────

class _StubScraper(BaseScraper):
    source_name = "_test_stub"

    def __init__(self, docs=None, content="stub content " * 20):
        self._docs = docs or []
        self._content = content

    async def search(self, *, limit=100, **kwargs):
        return self._docs[:limit]

    async def fetch_document(self, doc, save_original=True):
        if doc.get("fail"):
            raise RuntimeError("intentional failure")
        if not self._content or len(self._content.strip()) < 50:
            return None
        return ScrapedDocument(
            doc_id=self.make_doc_id(doc),
            source=self.source_name,
            title=doc.get("title", ""),
            content=self._content,
            metadata=doc,
        )


# ── ScrapedDocument ──────────────────────────────────────────────────────────

class TestScrapedDocument:
    def test_required_fields(self):
        d = ScrapedDocument(doc_id="1", source="test", title="T", content="C")
        assert d.doc_id == "1"
        assert d.metadata == {}
        assert d.url is None

    def test_optional_fields(self):
        d = ScrapedDocument(
            doc_id="1", source="test", title="T", content="C",
            url="http://x", urn="urn:lex:br", doc_type="lei",
        )
        assert d.url == "http://x"
        assert d.urn == "urn:lex:br"


# ── BaseScraper.make_doc_id ─────────────────────────────────────────────────

class TestMakeDocId:
    def test_sanitizes_title(self):
        s = _StubScraper()
        assert s.make_doc_id({"title": "My Doc / 2025"}) == "My_Doc___2025"

    def test_truncates_long_title(self):
        s = _StubScraper()
        doc_id = s.make_doc_id({"title": "A" * 200})
        assert len(doc_id) <= 100


# ── BaseScraper.fetch_all ────────────────────────────────────────────────────

class TestFetchAll:
    async def test_returns_all_successful(self):
        docs = [{"title": f"Doc {i}"} for i in range(5)]
        s = _StubScraper(docs=docs)
        results = await s.fetch_all(docs, concurrency=2)
        assert len(results) == 5
        assert all(isinstance(r, ScrapedDocument) for r in results)

    async def test_skips_failed_documents(self):
        docs = [{"title": "ok"}, {"title": "fail", "fail": True}, {"title": "ok2"}]
        s = _StubScraper(docs=docs)
        results = await s.fetch_all(docs, concurrency=2)
        assert len(results) == 2
        titles = {r.title for r in results}
        assert "ok" in titles
        assert "ok2" in titles

    async def test_skips_none_results(self):
        s = _StubScraper(content="short")
        docs = [{"title": "Doc 1"}]
        results = await s.fetch_all(docs, concurrency=1)
        assert len(results) == 0

    async def test_respects_concurrency(self):
        active = 0
        max_active = 0

        original_fetch = _StubScraper.fetch_document

        async def tracking_fetch(self, doc, save_original=True):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            result = await original_fetch(self, doc, save_original)
            active -= 1
            return result

        docs = [{"title": f"Doc {i}"} for i in range(10)]
        s = _StubScraper(docs=docs)
        s.fetch_document = lambda doc, save_original=True: tracking_fetch(s, doc, save_original)

        await s.fetch_all(docs, concurrency=3)
        assert max_active <= 3


# ── BaseScraper.save_original_file ───────────────────────────────────────────

class TestSaveOriginalFile:
    def test_saves_text_file(self, tmp_path):
        import json
        from unittest.mock import patch
        from crawler.scrapers import base

        with patch.object(base, "ORIGINALS_DIR", tmp_path):
            BaseScraper.save_original_file(
                "<html>test</html>",
                folder_name="leis",
                stem="test_doc",
                extension=".html",
                meta={"title": "Test"},
            )

        html_path = tmp_path / "leis" / "test_doc.html"
        meta_path = tmp_path / "leis" / "test_doc_meta.json"
        assert html_path.exists()
        assert meta_path.exists()
        assert html_path.read_text() == "<html>test</html>"
        meta = json.loads(meta_path.read_text())
        assert meta["title"] == "Test"
        assert "downloaded_at" in meta

    def test_saves_binary_file(self, tmp_path):
        from unittest.mock import patch
        from crawler.scrapers import base

        with patch.object(base, "ORIGINALS_DIR", tmp_path):
            BaseScraper.save_original_file(
                b"%PDF-1.4 content",
                folder_name="ica",
                stem="doc",
                extension=".pdf",
            )

        assert (tmp_path / "ica" / "doc.pdf").exists()
        assert (tmp_path / "ica" / "doc.pdf").read_bytes() == b"%PDF-1.4 content"


# ── async context manager ───────────────────────────────────────────────────

class TestAsyncContextManager:
    async def test_enter_and_exit(self):
        s = _StubScraper()
        async with s as scraper:
            assert scraper is s


# ── registry ────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_registered_scrapers_exist(self):
        names = list_scrapers()
        assert "decea" in names
        assert "lexml" in names
        assert "pdf" in names
        assert "sislaer" in names

    def test_get_scraper_decea(self):
        s = get_scraper("decea")
        assert s.source_name == "decea"
        assert isinstance(s, BaseScraper)

    def test_get_scraper_pdf(self):
        s = get_scraper("pdf", pdf_dir="/tmp/test")
        assert s.source_name == "pdf"

    def test_get_scraper_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scraper"):
            get_scraper("nonexistent_source")
