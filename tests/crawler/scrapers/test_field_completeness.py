"""Field-completeness tests across all scrapers.

Ensures that ScrapedDocument fields produced by each scraper are never
unexpectedly None or empty, and that cross-source canonical IDs are
consistent.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crawler.scrapers.base import (
    ScrapedDocument,
    compute_canonical_id,
    split_version_year,
)
from crawler.scrapers.sislaer_scraper import _parse_title, _normalize_doc_type
from crawler.scrapers.lexml_scraper import LexMLScraper
from crawler.scrapers.decea_scraper import DECEAScraper


# ── split_version_year edge cases ────────────────────────────────────────────


class TestSplitVersionYearEdgeCases:

    def test_none_returns_empty_tuple(self):
        assert split_version_year(None) == ("", None)

    def test_empty_string_returns_empty_tuple(self):
        assert split_version_year("") == ("", None)

    def test_standard_year_suffix(self):
        assert split_version_year("96-1/2025") == ("96-1", "2025")

    def test_no_slash(self):
        assert split_version_year("8666") == ("8666", None)

    def test_non_year_suffix(self):
        assert split_version_year("1082/GM3") == ("1082/GM3", None)

    def test_three_digit_suffix(self):
        assert split_version_year("100-1/123") == ("100-1/123", None)

    def test_five_digit_suffix(self):
        assert split_version_year("100-1/12345") == ("100-1/12345", None)


# ── compute_canonical_id edge cases ──────────────────────────────────────────


class TestComputeCanonicalIdEdgeCases:

    def test_none_type_returns_none(self):
        assert compute_canonical_id(None, "96-1") is None

    def test_none_number_returns_none(self):
        assert compute_canonical_id("ICA", None) is None

    def test_empty_number_returns_none(self):
        assert compute_canonical_id("ICA", "") is None

    def test_both_none_returns_none(self):
        assert compute_canonical_id(None, None) is None


# ── cross-source canonical ID consistency ────────────────────────────────────


class TestCrossSourceCanonicalConsistency:
    """SISLAER and DECEA must produce the same canonical_id for the same doc."""

    @pytest.mark.parametrize("sislaer_number,decea_number,doc_type,expected", [
        ("96-1", "ICA96-1", "ICA", "ica_96-1"),
        ("100-1", "ICA100-1", "ICA", "ica_100-1"),
        ("53-1", "ICA53-1", "ICA", "ica_53-1"),
        ("55-82", "ICA55-82", "ICA", "ica_55-82"),
        ("53-11", "MCA53-11", "MCA", "mca_53-11"),
    ])
    def test_sislaer_decea_same_canonical(
        self, sislaer_number, decea_number, doc_type, expected
    ):
        sislaer_canonical = compute_canonical_id(doc_type, sislaer_number)
        decea_canonical = compute_canonical_id(doc_type, decea_number)
        assert sislaer_canonical == decea_canonical == expected


# ── SISLAER _parse_title completeness ────────────────────────────────────────


class TestParseTitleCompleteness:
    """_parse_title must handle structured titles AND type-only titles."""

    def test_standard_ica(self):
        r = _parse_title("ICA 96-1/2025")
        assert r["doc_type"] == "ICA"
        assert r["number"] == "96-1/2025"

    def test_dca_with_year(self):
        r = _parse_title("DCA 100-1/2018")
        assert r["doc_type"] == "DCA"
        assert r["number"] == "100-1/2018"

    def test_type_only_nsca(self):
        r = _parse_title("NSCA")
        assert r["doc_type"] == "NSCA"
        assert r["number"] is None

    def test_type_only_nsca_with_year(self):
        r = _parse_title("NSCA/2023")
        assert r["doc_type"] == "NSCA"
        assert r["number"] == "2023"

    def test_type_only_mca(self):
        r = _parse_title("MCA")
        assert r["doc_type"] == "MCA"
        assert r["number"] is None

    def test_lei_with_number(self):
        r = _parse_title("Lei Nº 8666")
        assert r["doc_type"] is not None
        assert r["number"] == "8666"

    def test_decreto_with_number(self):
        r = _parse_title("Decreto 4321")
        assert r["doc_type"] is not None
        assert r["number"] == "4321"

    def test_empty_title(self):
        r = _parse_title("")
        assert r["doc_type"] is None
        assert r["number"] is None

    def test_unrecognized_title(self):
        r = _parse_title("Random Garbage 123")
        # Should at least not crash; result depends on regex match
        assert isinstance(r, dict)


# ── SISLAER ScrapedDocument field completeness ───────────────────────────────


class TestSislaerFieldCompleteness:
    """SISLAER's fetch_document should populate key ScrapedDocument fields."""

    def _build_detail(self, title="ICA 96-1/2025", **overrides):
        detail = {
            "title": title,
            "situacao": "Em vigor",
            "authority": "DECEA",
            "relations": [],
            "ato_publicacao": "01/01/2025",
            "publicacao": None,
            "portaria_aprovacao": None,
            "ementa": "Test ementa",
            "observacoes": None,
            "natureza": "regulatoria",
        }
        detail.update(overrides)
        return detail

    def _make_scraped_doc(self, title="ICA 96-1/2025", reg_id=12345, **detail_kw):
        """Simulate what fetch_document does, without network."""
        from crawler.scrapers.sislaer_scraper import _normalize_doc_type

        detail = self._build_detail(title, **detail_kw)
        parsed = _parse_title(detail["title"])
        doc_type = _normalize_doc_type(parsed["doc_type"]) or None
        raw_number = parsed["number"] or ""

        number, version_year = split_version_year(raw_number)
        canonical = compute_canonical_id(doc_type, number)
        doc_id = (
            f"{canonical}/{version_year}" if canonical and version_year
            else (canonical or f"sislaer_{reg_id}")
        )

        situacao = (detail.get("situacao") or "").lower()
        status = "revoked" if "revogado" in situacao else None

        return ScrapedDocument(
            doc_id=doc_id,
            source="sislaer",
            title=detail["title"],
            content="x" * 200,
            url=f"https://sislaer.fab.mil.br/acervo/detalhe/{reg_id}",
            doc_type=doc_type,
            canonical_id=canonical,
            source_ref=f"sislaer:{reg_id}",
            status=status,
            number=number,
            authority=detail.get("authority"),
            version_year=version_year,
        )

    def test_ica_all_fields_populated(self):
        doc = self._make_scraped_doc("ICA 96-1/2025")
        assert doc.doc_type == "ICA"
        assert doc.number == "96-1"
        assert doc.version_year == "2025"
        assert doc.canonical_id == "ica_96-1"
        assert doc.doc_id == "ica_96-1/2025"
        assert doc.authority == "DECEA"
        assert doc.source == "sislaer"

    def test_dca_all_fields_populated(self):
        doc = self._make_scraped_doc("DCA 100-1/2018")
        assert doc.doc_type == "DCA"
        assert doc.number == "100-1"
        assert doc.version_year == "2018"
        assert doc.canonical_id == "dca_100-1"
        assert doc.doc_id == "dca_100-1/2018"

    def test_nsca_type_only_uses_fallback_id(self):
        doc = self._make_scraped_doc("NSCA", reg_id=48618)
        assert doc.doc_type == "NSCA"
        assert doc.number == ""
        assert doc.canonical_id is None
        assert doc.doc_id == "sislaer_48618"

    def test_nsca_with_year_suffix(self):
        doc = self._make_scraped_doc("NSCA/2023", reg_id=47605)
        assert doc.doc_type == "NSCA"
        assert doc.number == "2023"
        assert doc.canonical_id == "nsca_2023"

    def test_revoked_status(self):
        doc = self._make_scraped_doc("ICA 100-1/2016", situacao="Revogado")
        assert doc.status == "revoked"

    def test_active_status(self):
        doc = self._make_scraped_doc("ICA 100-1/2018", situacao="Em vigor")
        assert doc.status is None  # not revoked, temporal extractor decides

    def test_lei_type_normalization(self):
        doc = self._make_scraped_doc("Lei Nº 8666")
        assert doc.doc_type is not None
        assert doc.number == "8666"


# ── DECEA ScrapedDocument field completeness ─────────────────────────────────


class TestDeceaFieldCompleteness:
    """DECEA's _fetch_document_sync should populate key ScrapedDocument fields."""

    def _make_decea_doc(
        self, slug="ICA-96-1", number="ICA96-1", doc_type="ica", origin="SDOP"
    ):
        """Simulate what _fetch_document_sync does, without network."""
        raw_number = number or ""
        dtype = (doc_type or "").upper()
        authority = origin or "DECEA"

        num, version_year = split_version_year(raw_number)
        canonical = compute_canonical_id(dtype, num)
        doc_id = (
            (f"{canonical}/{version_year}" if version_year else canonical)
            if canonical else f"decea_{slug}"
        )

        return ScrapedDocument(
            doc_id=doc_id,
            source="decea",
            title="Test document",
            content="x" * 200,
            url=f"https://publicacoes.decea.mil.br/publicacao/{slug}",
            doc_type=dtype,
            canonical_id=canonical,
            source_ref=f"decea:{slug}",
            number=num,
            authority=authority,
            version_year=version_year,
        )

    def test_ica_all_fields_populated(self):
        doc = self._make_decea_doc("ICA-96-1", "ICA96-1", "ica")
        assert doc.doc_type == "ICA"
        assert doc.number == "ICA96-1"
        assert doc.canonical_id == "ica_96-1"
        assert doc.doc_id == "ica_96-1"
        assert doc.authority == "SDOP"

    def test_mca_canonical_consistency(self):
        doc = self._make_decea_doc("MCA-53-11", "MCA53-11", "mca")
        assert doc.doc_type == "MCA"
        assert doc.canonical_id == "mca_53-11"

    def test_none_number_uses_slug_fallback(self):
        doc = self._make_decea_doc("UNK-DOC", None, "ica")
        assert doc.doc_id == "decea_UNK-DOC"
        assert doc.canonical_id is None
        assert doc.number == ""

    def test_none_doc_type_uses_slug_fallback(self):
        doc = self._make_decea_doc("UNK-DOC", "ICA96-1", None)
        assert doc.doc_id == "decea_UNK-DOC"

    def test_none_origin_defaults_to_decea(self):
        doc = self._make_decea_doc("ICA-96-1", "ICA96-1", "ica", origin=None)
        assert doc.authority == "DECEA"


# ── LexML ScrapedDocument field completeness ─────────────────────────────────


class TestLexmlFieldCompleteness:
    """LexML's fetch_document should populate key ScrapedDocument fields."""

    @pytest.fixture
    async def scraper(self):
        async with LexMLScraper(skip_duplicates=False, max_rate=1000) as s:
            yield s

    async def test_standard_lei_all_fields(self, scraper):
        async def mock_text(doc, save_original=True):
            return "A" * 200

        scraper.get_document_text = mock_text
        doc = {
            "url": "http://example.com",
            "title": "Lei 10000",
            "urn": "urn:lex:br:federal:lei:2001-01-01;10000",
            "doc_type": "lei",
            "number": "10000",
            "authority": "federal",
        }
        result = await scraper.fetch_document(doc, save_original=False)
        assert result is not None
        assert result.doc_type == "lei"
        assert result.number == "10000"
        assert result.authority == "federal"
        assert result.canonical_id == "lei_10000"
        assert result.doc_id == "lei_10000"
        assert result.source == "lexml"

    async def test_urn_without_number_segment(self, scraper):
        """URN like urn:lex:br:federal:lei:2001-01-01 has no ;number."""
        async def mock_text(doc, save_original=True):
            return "A" * 200

        scraper.get_document_text = mock_text
        doc = {
            "url": "http://example.com",
            "title": "Lei sem número",
            "urn": "urn:lex:br:federal:lei:2001-01-01",
            "doc_type": "lei",
            "number": None,
            "authority": "federal",
        }
        result = await scraper.fetch_document(doc, save_original=False)
        assert result is not None
        assert result.number == ""
        assert result.doc_type == "lei"
        assert result.canonical_id is None
        assert "urn_lex_br_federal_lei_2001-01-01" in result.doc_id

    async def test_none_doc_type_and_number(self, scraper):
        """Edge case: all parsed fields are None."""
        async def mock_text(doc, save_original=True):
            return "A" * 200

        scraper.get_document_text = mock_text
        doc = {
            "url": "http://example.com",
            "title": "Unknown doc",
            "urn": "urn:lex:br:federal:unknown:2001-01-01",
            "doc_type": None,
            "number": None,
            "authority": None,
        }
        result = await scraper.fetch_document(doc, save_original=False)
        assert result is not None
        assert result.number == ""
        assert result.doc_type == ""
        assert result.authority == ""

    async def test_make_doc_id_with_none_number(self, scraper):
        """make_doc_id must not crash when number is None."""
        doc = {
            "urn": "urn:lex:br:federal:lei:2001-01-01",
            "doc_type": "lei",
            "number": None,
        }
        doc_id = scraper.make_doc_id(doc)
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0


# ── publication date extraction ──────────────────────────────────────────────


class TestPublicationDateExtraction:
    """_publication_date in collect.py must find dates from all scraper metadata formats.

    Raw ``publicacao`` is excluded from the fallback chain because it can
    contain free text.  The SISLAER scraper extracts the date from it into
    ``publication_date``.
    """

    @staticmethod
    def _publication_date(doc: ScrapedDocument) -> str | None:
        """Mirror the logic from scripts/collect.py."""
        meta = doc.metadata or {}
        return (
            meta.get("date")
            or meta.get("publication_date")
            or meta.get("date_published")
            or meta.get("ato_publicacao")
            or meta.get("portaria_aprovacao")
        )

    def test_sislaer_ato_publicacao(self):
        doc = ScrapedDocument(
            doc_id="test", source="sislaer", title="T", content="x" * 100,
            metadata={"ato_publicacao": "01/01/2025", "publicacao": None},
        )
        assert self._publication_date(doc) == "01/01/2025"

    def test_sislaer_portaria_aprovacao_fallback(self):
        doc = ScrapedDocument(
            doc_id="test", source="sislaer", title="T", content="x" * 100,
            metadata={
                "ato_publicacao": None,
                "publicacao": None,
                "portaria_aprovacao": "10/12/2020",
            },
        )
        assert self._publication_date(doc) == "10/12/2020"

    def test_lexml_publication_date(self):
        doc = ScrapedDocument(
            doc_id="test", source="lexml", title="T", content="x" * 100,
            metadata={"publication_date": "2001-01-01"},
        )
        assert self._publication_date(doc) == "2001-01-01"

    def test_all_none_returns_none(self):
        doc = ScrapedDocument(
            doc_id="test", source="sislaer", title="T", content="x" * 100,
            metadata={
                "ato_publicacao": None,
                "publicacao": None,
                "portaria_aprovacao": None,
            },
        )
        assert self._publication_date(doc) is None

    def test_raw_publicacao_not_used_directly(self):
        """Free-text publicacao should NOT leak into the date chain."""
        doc = ScrapedDocument(
            doc_id="test", source="sislaer", title="T", content="x" * 100,
            metadata={
                "ato_publicacao": None,
                "publicacao": "PUB BCA de 20/11/2019 página 016749",
                "portaria_aprovacao": None,
            },
        )
        assert self._publication_date(doc) is None


# ── SISLAER publicacao date extraction ───────────────────────────────────────


class TestSislaerPublicacaoDateExtraction:
    """The scraper should extract clean dates from free-text publicacao."""

    @pytest.mark.parametrize("text,expected", [
        ("PUB BCA de 20/11/2019 página 016749", "20/11/2019"),
        ("PUB BCA de 22/09/2016", "22/09/2016"),
        ("PUB 03, 01/03/2002.", "01/03/2002"),
        ("PUB BCA de 14/12/2022 página 017797", "14/12/2022"),
        ("BCA Nº 072 DE 02 DE MAIO DE 2019", "02/05/2019"),
        ("BCA N° 214, DE 24 DE NOVEMBRO DE 2025.", "24/11/2025"),
        ("BCA Nº 124, de 7 de julho de 2021", "07/07/2021"),
        ("PUB  BCA no 110, de 5 de julho de 2016.", "05/07/2016"),
        ("Publicada no BCA nº 153, de 19 de agosto de 2021", "19/08/2021"),
        ("BCA 171, de 22 de setembro de 2020.", "22/09/2020"),
        ("BCA Nº 036, de 24 de fevereiro de 2023.", "24/02/2023"),
        ("BCA N° 159, DE 29 DE AGOSTO DE 2023", "29/08/2023"),
    ])
    def test_extracts_date(self, text, expected):
        from crawler.scrapers.sislaer_scraper import _extract_date_from_text
        assert _extract_date_from_text(text) == expected

    def test_no_date_returns_none(self):
        from crawler.scrapers.sislaer_scraper import _extract_date_from_text
        assert _extract_date_from_text("texto sem data") is None

    def test_empty_string(self):
        from crawler.scrapers.sislaer_scraper import _extract_date_from_text
        assert _extract_date_from_text("") is None

    def test_none_input(self):
        from crawler.scrapers.sislaer_scraper import _extract_date_from_text
        assert _extract_date_from_text(None) is None
