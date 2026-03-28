"""Tests for the SISLAER scraper."""

import pytest

from crawler.scrapers.base import compute_canonical_id, ScrapedDocument
from crawler.scrapers.sislaer_scraper import (
    _parse_title,
    _normalize_doc_type,
    SISLAERScraper,
)


class TestParseTitle:

    def test_ica_with_year(self):
        result = _parse_title("ICA 55-82/2012")
        assert result["doc_type"] == "ICA"
        assert result["number"] == "55-82/2012"

    def test_portaria_with_complex_number(self):
        result = _parse_title("Portaria DIRAP Nº 552/3VP/2025")
        assert result["doc_type"] == "Portaria DIRAP"
        assert result["number"] == "552/3VP/2025"

    def test_nsca(self):
        result = _parse_title("NSCA 13-1/2022")
        assert result["doc_type"] == "NSCA"
        assert result["number"] == "13-1/2022"

    def test_roca(self):
        result = _parse_title("ROCA 21-80/2017")
        assert result["doc_type"] == "ROCA"
        assert result["number"] == "21-80/2017"

    def test_portaria_simple(self):
        result = _parse_title("Portaria 1.082/GM3/1981")
        assert result["doc_type"] == "Portaria"
        assert result["number"] == "1.082/GM3/1981"

    def test_portaria_with_number_prefix(self):
        result = _parse_title("Portaria 249/1947")
        assert result["doc_type"] == "Portaria"
        assert result["number"] == "249/1947"

    def test_no_match_returns_none(self):
        result = _parse_title("Some random title without a pattern")
        assert result["doc_type"] is None
        assert result["number"] is None

    def test_empty_string(self):
        result = _parse_title("")
        assert result["doc_type"] is None


class TestNormalizeDocType:

    def test_uppercase_portaria(self):
        assert _normalize_doc_type("PORTARIA") == "Portaria"

    def test_instrucao_normativa(self):
        assert _normalize_doc_type("INSTRUÇÃO NORMATIVA") == "Instrução Normativa"

    def test_ica_passes_through(self):
        assert _normalize_doc_type("ICA") == "ICA"

    def test_none(self):
        assert _normalize_doc_type(None) is None


class TestComputeCanonicalId:

    def test_basic(self):
        assert compute_canonical_id("ICA", "100-12") == "ica_100-12"

    def test_with_year(self):
        assert compute_canonical_id("ICA", "55-82/2012") == "ica_55-82/2012"

    def test_lei(self):
        assert compute_canonical_id("Lei", "8666") == "lei_8666"

    def test_portaria(self):
        assert compute_canonical_id("Portaria", "1082/GM3/1981") == "portaria_1082/gm3/1981"

    def test_none_type(self):
        assert compute_canonical_id(None, "123") is None

    def test_none_number(self):
        assert compute_canonical_id("ICA", None) is None

    def test_empty_strings(self):
        assert compute_canonical_id("", "") is None


class TestSISLAERScrapedDocument:

    def test_canonical_id_field(self):
        doc = ScrapedDocument(
            doc_id="sislaer_2000",
            source="sislaer",
            title="ICA 55-82/2012",
            content="test content",
            canonical_id="ica_55-82/2012",
        )
        assert doc.canonical_id == "ica_55-82/2012"

    def test_canonical_id_default_none(self):
        doc = ScrapedDocument(
            doc_id="test", source="test", title="T", content="C",
        )
        assert doc.canonical_id is None


class TestMakeDocId:

    def test_generates_sislaer_prefix(self):
        scraper = SISLAERScraper.__new__(SISLAERScraper)
        doc_id = scraper.make_doc_id({"codigoRegistro": 2000})
        assert doc_id == "sislaer_2000"


class TestRegistration:

    def test_sislaer_in_registry(self):
        from crawler.scrapers import list_scrapers
        assert "sislaer" in list_scrapers()

    def test_get_sislaer_scraper(self):
        from crawler.scrapers import get_scraper
        s = get_scraper("sislaer")
        assert s.source_name == "sislaer"
