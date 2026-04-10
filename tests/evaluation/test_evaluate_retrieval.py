"""Unit tests for retrieval evaluation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from evaluation.evaluate_retrieval import (
    RetrievalEvaluator,
    _compute_ndcg,
    _extract_doc_id,
    _matches_expected,
    _normalize_id,
    print_report,
)


@pytest.fixture
def golden_set_file(tmp_path):
    """Create a minimal golden set CSV."""
    gs = tmp_path / "golden_set.csv"
    gs.write_text(
        "query_id,query,expected_doc_id,relevance,category,notes\n"
        "Q1,pergunta sobre voo,DOC-1,relevant,retrieval,\n"
        "Q1,pergunta sobre voo,DOC-2,moderate,retrieval,\n"
        "Q2,outra pergunta,DOC-3,relevant,retrieval,\n"
        "Q3,documento inexistente,NOT_IN_DB,irrelevant,coverage,\n",
        encoding="utf-8",
    )
    return str(gs)


@pytest.fixture
def evaluator(golden_set_file):
    return RetrievalEvaluator(golden_set_path=golden_set_file)


class TestNormalizeId:
    def test_converts_golden_format_to_canonical(self):
        assert _normalize_id("ICA-96-1-art563") == "ica_96-1-art563"

    def test_already_canonical_unchanged(self):
        assert _normalize_id("ica_96-1/2025-art563") == "ica_96-1/2025-art563"

    def test_simple_doc_id(self):
        assert _normalize_id("DOC-1") == "doc_1"

    def test_empty_string(self):
        assert _normalize_id("") == ""


class TestExtractDocId:
    def test_strips_article(self):
        assert _extract_doc_id("ica_96-1/2025-art563") == "ica_96-1/2025"

    def test_strips_article_sub_index(self):
        assert _extract_doc_id("ica_7-58-art2-0") == "ica_7-58"

    def test_no_article(self):
        assert _extract_doc_id("ica_7-58") == "ica_7-58"

    def test_golden_format_strips_article(self):
        assert _extract_doc_id("ICA-96-1-art10") == "ica_96-1"


class TestMatchesExpected:
    def test_canonical_article_match(self):
        assert _matches_expected("ica_96-1/2025-art563", "ICA-96-1-art563") is True

    def test_article_mismatch(self):
        assert _matches_expected("ica_96-1/2025-art999", "ICA-96-1-art563") is False

    def test_doc_level_matches_any_chunk(self):
        assert _matches_expected("ica_7-58/2020-art2-0", "ICA-7-58") is True

    def test_doc_level_mismatch(self):
        assert _matches_expected("ica_100-47/2023-art5", "ICA-7-58") is False

    def test_both_simple(self):
        assert _matches_expected("DOC-1", "DOC-1") is True

    def test_version_agnostic(self):
        """Different versions of the same document should match."""
        assert _matches_expected("ica_96-1/2018-art563", "ICA-96-1-art563") is True
        assert _matches_expected("ica_96-1/2025-art563", "ICA-96-1-art563") is True


class TestComputeNDCG:
    def test_perfect_ranking(self):
        result = _compute_ndcg(["A", "B"], relevant=["A"], moderate=["B"], k=2)
        assert result == pytest.approx(1.0)

    def test_reversed_ranking(self):
        result = _compute_ndcg(["B", "A"], relevant=["A"], moderate=["B"], k=2)
        assert 0 < result < 1.0

    def test_no_relevant_docs(self):
        result = _compute_ndcg(["X", "Y"], relevant=["A"], moderate=[], k=2)
        assert result == pytest.approx(0.0)

    def test_empty_expected(self):
        result = _compute_ndcg(["X"], relevant=[], moderate=[], k=1)
        assert result == pytest.approx(0.0)


class TestRetrievalEvaluator:
    def test_load_golden_set(self, evaluator):
        assert len(evaluator.golden_set) == 3
        assert "Q1" in evaluator.golden_set
        assert len(evaluator.golden_set["Q1"]) == 2

    def test_get_expected_docs(self, evaluator):
        relevant, moderate = evaluator._get_expected_docs("Q1")
        assert relevant == ["DOC-1"]
        assert moderate == ["DOC-2"]

    def test_missing_golden_set(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RetrievalEvaluator(golden_set_path=str(tmp_path / "nope.csv"))

    @patch("evaluation.evaluate_retrieval.QdrantManager")
    @patch("evaluation.evaluate_retrieval.create_embedding_model")
    def test_evaluate_hit(self, MockCreateEmbed, MockQdrant, evaluator):
        mock_embed = MockCreateEmbed.return_value
        mock_embed.encode.return_value = np.random.rand(3, 1024)

        mock_point = MagicMock()
        mock_point.payload = {"regulation_id": "DOC-1"}
        mock_point.score = 0.9

        mock_qdrant = MockQdrant.return_value
        mock_qdrant.search.return_value = [mock_point]

        result = evaluator.evaluate(k=1, workers=1)

        assert result.total_queries == 3
        assert result.retrieval_queries == 2
        assert result.coverage_queries == 1

        q1 = next(r for r in result.query_results if r.query_id == "Q1")
        assert q1.hit is True
        assert q1.first_relevant_rank == 1

    @patch("evaluation.evaluate_retrieval.QdrantManager")
    @patch("evaluation.evaluate_retrieval.create_embedding_model")
    def test_evaluate_hit_with_canonical_id(self, MockCreateEmbed, MockQdrant, evaluator):
        """Retrieved canonical IDs should match golden set IDs."""
        mock_embed = MockCreateEmbed.return_value
        mock_embed.encode.return_value = np.random.rand(3, 1024)

        mock_point = MagicMock()
        mock_point.payload = {"regulation_id": "doc_1"}
        mock_point.score = 0.9

        mock_qdrant = MockQdrant.return_value
        mock_qdrant.search.return_value = [mock_point]

        result = evaluator.evaluate(k=1, workers=1)

        q1 = next(r for r in result.query_results if r.query_id == "Q1")
        assert q1.hit is True
        assert q1.first_relevant_rank == 1

    @patch("evaluation.evaluate_retrieval.QdrantManager")
    @patch("evaluation.evaluate_retrieval.create_embedding_model")
    def test_evaluate_miss(self, MockCreateEmbed, MockQdrant, evaluator):
        mock_embed = MockCreateEmbed.return_value
        mock_embed.encode.return_value = np.random.rand(3, 1024)

        mock_point = MagicMock()
        mock_point.payload = {"regulation_id": "WRONG-DOC"}
        mock_point.score = 0.5

        mock_qdrant = MockQdrant.return_value
        mock_qdrant.search.return_value = [mock_point]

        result = evaluator.evaluate(k=1, workers=1)

        q2 = next(r for r in result.query_results if r.query_id == "Q2")
        assert q2.hit is False
        assert q2.first_relevant_rank is None

    @patch("evaluation.evaluate_retrieval.QdrantManager")
    @patch("evaluation.evaluate_retrieval.create_embedding_model")
    def test_evaluate_coverage(self, MockCreateEmbed, MockQdrant, evaluator):
        mock_embed = MockCreateEmbed.return_value
        mock_embed.encode.return_value = np.random.rand(3, 1024)

        mock_qdrant = MockQdrant.return_value
        mock_qdrant.search.return_value = []

        result = evaluator.evaluate(k=1, workers=1)

        assert result.coverage_correct_rate == 1.0

    @patch("evaluation.evaluate_retrieval.QdrantManager")
    @patch("evaluation.evaluate_retrieval.create_embedding_model")
    def test_evaluate_parallel(self, MockCreateEmbed, MockQdrant, evaluator):
        mock_embed = MockCreateEmbed.return_value
        mock_embed.encode.return_value = np.random.rand(3, 1024)

        mock_qdrant = MockQdrant.return_value
        mock_qdrant.search.return_value = []

        result = evaluator.evaluate(k=1, workers=2)
        assert result.total_queries == 3

    @patch("evaluation.evaluate_retrieval.QdrantManager")
    @patch("evaluation.evaluate_retrieval.create_embedding_model")
    def test_save_results(self, MockCreateEmbed, MockQdrant, evaluator, tmp_path):
        mock_embed = MockCreateEmbed.return_value
        mock_embed.encode.return_value = np.random.rand(3, 1024)

        mock_qdrant = MockQdrant.return_value
        mock_qdrant.search.return_value = []

        result = evaluator.evaluate(k=1, workers=1)
        summary, details, json_path = evaluator.save_results(result, str(tmp_path))

        assert Path(summary).exists()
        assert Path(details).exists()
        assert Path(json_path).exists()

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["k"] == 1
        assert len(data["query_results"]) == 3


class TestPrintReport:
    @patch("evaluation.evaluate_retrieval.QdrantManager")
    @patch("evaluation.evaluate_retrieval.create_embedding_model")
    def test_print_report_runs(self, MockCreateEmbed, MockQdrant, evaluator, capsys):
        mock_embed = MockCreateEmbed.return_value
        mock_embed.encode.return_value = np.random.rand(3, 1024)

        mock_qdrant = MockQdrant.return_value
        mock_qdrant.search.return_value = []

        result = evaluator.evaluate(k=1, workers=1)
        print_report(result)

        captured = capsys.readouterr()
        assert "RETRIEVAL QUALITY" in captured.out
        assert "COVERAGE" in captured.out
