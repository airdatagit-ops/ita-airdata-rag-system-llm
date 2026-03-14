"""Tests for the ingestion pipeline (all external deps mocked)."""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from pipeline.ingestion import IngestionPipeline, generate_point_id


# ---------------------------------------------------------------------------
# generate_point_id
# ---------------------------------------------------------------------------

class TestGeneratePointId:

    def test_returns_valid_uuid(self):
        pid = generate_point_id("some text")
        parts = pid.split("-")
        assert len(parts) == 5

    def test_deterministic(self):
        assert generate_point_id("abc") == generate_point_id("abc")

    def test_different_inputs_differ(self):
        assert generate_point_id("a") != generate_point_id("b")


# ---------------------------------------------------------------------------
# Fixture: pipeline with mocked dependencies
# ---------------------------------------------------------------------------

PATCHES = [
    "pipeline.ingestion.EmbeddingModel",
    "pipeline.ingestion.SparseEncoder",
    "pipeline.ingestion.QdrantManager",
    "pipeline.ingestion.LexMLParser",
    "pipeline.ingestion.PDFParser",
    "pipeline.ingestion.ArticleChunker",
    "pipeline.ingestion.ICAChunker",
]


@pytest.fixture
def pipeline():
    with patch.multiple("pipeline.ingestion", **{cls.rsplit(".", 1)[-1]: MagicMock() for cls in PATCHES}):
        p = IngestionPipeline()
        p.dense_model = MagicMock()
        p.sparse_model = None
        p.db = MagicMock()
        p.lexml_parser = MagicMock()
        p.pdf_parser = MagicMock()
        p.chunker = MagicMock()
        yield p


# ---------------------------------------------------------------------------
# _load_and_chunk
# ---------------------------------------------------------------------------

_VALID_CONTENT = (
    "Art 1 Esta instrucao estabelece os procedimentos para operacao de aeronaves "
    "no espaco aereo brasileiro conforme as normas vigentes do Comando da Aeronautica "
    "e regulamentacoes do DECEA aplicaveis a todos os operadores e pilotos certificados "
    "que atuam em territorio nacional ou sob jurisdicao brasileira"
)


def _make_json(content=_VALID_CONTENT, slug="ICA-1-1", doc_type="ICA", **extra):
    doc = {"content": content, "slug": slug, "type": doc_type, "title": "T", **extra}
    return json.dumps(doc)


class TestLoadAndChunk:

    @patch("pipeline.ingestion.get_chunker")
    def test_returns_chunks(self, mock_get_chunker, pipeline):
        chunker = MagicMock()
        chunker.chunk.return_value = [{"text": "chunk1", "regulation_id": "ICA-1-1"}]
        mock_get_chunker.return_value = chunker

        data = _make_json()
        with patch("builtins.open", mock_open(read_data=data)):
            chunks = pipeline._load_and_chunk("fake.json")

        assert len(chunks) == 1
        assert chunks[0]["text"] == "chunk1"

    @patch("pipeline.ingestion.get_chunker")
    def test_skips_short_content(self, mock_get_chunker, pipeline):
        data = _make_json(content="short")
        with patch("builtins.open", mock_open(read_data=data)):
            chunks = pipeline._load_and_chunk("fake.json")
        assert chunks == []
        mock_get_chunker.assert_not_called()

    @patch("pipeline.ingestion.get_chunker")
    def test_sanitizes_doc_id(self, mock_get_chunker, pipeline):
        chunker = MagicMock()
        chunker.chunk.return_value = [{"text": "c", "regulation_id": "cleaned"}]
        mock_get_chunker.return_value = chunker

        data = _make_json(slug=None, urn="urn:lex:br:federal:lei:2023;1234")
        with patch("builtins.open", mock_open(read_data=data)):
            pipeline._load_and_chunk("fake.json")

        article = chunker.chunk.call_args[0][0]
        assert ":" not in article["regulation_id"]
        assert "/" not in article["regulation_id"]


# ---------------------------------------------------------------------------
# ingest_json_documents
# ---------------------------------------------------------------------------

class TestIngestJsonDocuments:

    @patch("pipeline.ingestion.get_chunker")
    def test_batch_encode_and_upsert(self, mock_get_chunker, pipeline, tmp_path):
        """Verifies single batch encode + single upsert call."""
        chunker = MagicMock()
        chunker.chunk.return_value = [{"text": "chunk text", "regulation_id": "d1"}]
        mock_get_chunker.return_value = chunker

        files = []
        for i in range(3):
            f = tmp_path / f"doc{i}.json"
            f.write_text(_make_json(slug=f"DOC-{i}"))
            files.append(str(f))

        pipeline.dense_model.encode.return_value = np.random.rand(3, 4)

        result = pipeline.ingest_json_documents(files)

        assert result == 3
        pipeline.dense_model.encode.assert_called_once()
        assert len(pipeline.dense_model.encode.call_args[0][0]) == 3
        pipeline.db.upsert_points.assert_called_once()
        assert len(pipeline.db.upsert_points.call_args[0][0]) == 3

    @patch("pipeline.ingestion.get_chunker")
    def test_returns_zero_for_empty_content(self, mock_get_chunker, pipeline, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text(_make_json(content="short"))

        result = pipeline.ingest_json_documents([str(f)])

        assert result == 0
        pipeline.dense_model.encode.assert_not_called()
        pipeline.db.upsert_points.assert_not_called()

    @patch("pipeline.ingestion.get_chunker")
    def test_skips_bad_files_gracefully(self, mock_get_chunker, pipeline, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json")

        good = tmp_path / "good.json"
        good.write_text(_make_json(slug="GOOD-1"))

        chunker = MagicMock()
        chunker.chunk.return_value = [{"text": "ok", "regulation_id": "g"}]
        mock_get_chunker.return_value = chunker
        pipeline.dense_model.encode.return_value = np.random.rand(1, 4)

        result = pipeline.ingest_json_documents([str(bad), str(good)])
        assert result == 1

    @patch("pipeline.ingestion.get_chunker")
    def test_point_ids_are_deterministic(self, mock_get_chunker, pipeline, tmp_path):
        chunker = MagicMock()
        chunker.chunk.return_value = [{"text": "same text", "regulation_id": "X"}]
        mock_get_chunker.return_value = chunker
        pipeline.dense_model.encode.return_value = np.random.rand(1, 4)

        f = tmp_path / "doc.json"
        f.write_text(_make_json(slug="X"))

        pipeline.ingest_json_documents([str(f)])
        points_a = pipeline.db.upsert_points.call_args[0][0]

        pipeline.db.reset_mock()
        pipeline.ingest_json_documents([str(f)])
        points_b = pipeline.db.upsert_points.call_args[0][0]

        assert points_a[0]["id"] == points_b[0]["id"]


# ---------------------------------------------------------------------------
# ingest_lexml
# ---------------------------------------------------------------------------

class TestIngestLexml:

    def test_batch_encode_and_upsert(self, pipeline):
        pipeline.lexml_parser.parse_xml.return_value = [{"text": "art1"}]
        pipeline.chunker.chunk.return_value = [{"text": "c1", "regulation_id": "r1"}]
        pipeline.dense_model.encode.return_value = np.random.rand(2, 4)

        result = pipeline.ingest_lexml(["a.xml", "b.xml"])

        assert result == 2
        pipeline.dense_model.encode.assert_called_once()
        pipeline.db.upsert_points.assert_called_once()

    def test_returns_zero_when_no_chunks(self, pipeline):
        pipeline.lexml_parser.parse_xml.side_effect = Exception("parse error")
        assert pipeline.ingest_lexml(["bad.xml"]) == 0


# ---------------------------------------------------------------------------
# ingest_pdfs
# ---------------------------------------------------------------------------

class TestIngestPdfs:

    def test_batch_encode_and_upsert(self, pipeline):
        pipeline.pdf_parser.parse_pdf.return_value = [
            {"text": "sec1", "regulation_id": "p1"},
            {"text": "sec2", "regulation_id": "p2"},
        ]
        pipeline.dense_model.encode.return_value = np.random.rand(2, 4)

        result = pipeline.ingest_pdfs(["doc.pdf"])

        assert result == 2
        pipeline.dense_model.encode.assert_called_once()
        pipeline.db.upsert_points.assert_called_once()

    def test_returns_zero_on_parse_error(self, pipeline):
        pipeline.pdf_parser.parse_pdf.side_effect = Exception("bad pdf")
        assert pipeline.ingest_pdfs(["bad.pdf"]) == 0

    def test_generates_id_when_missing(self, pipeline):
        pipeline.pdf_parser.parse_pdf.return_value = [{"text": "no id section"}]
        pipeline.dense_model.encode.return_value = np.random.rand(1, 4)

        pipeline.ingest_pdfs(["doc.pdf"])
        points = pipeline.db.upsert_points.call_args[0][0]
        assert points[0]["id"] is not None
