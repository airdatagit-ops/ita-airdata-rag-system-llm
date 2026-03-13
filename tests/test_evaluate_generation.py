"""Unit tests for generation evaluation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from evaluation.evaluate_generation import (
    GenerationEvaluator,
    analyze_response,
    print_report,
)


@pytest.fixture
def golden_set_file(tmp_path):
    gs = tmp_path / "golden_set.csv"
    gs.write_text(
        "query_id,query,expected_doc_id,relevance,category,notes\n"
        "Q1,pergunta sobre voo,DOC-1,relevant,retrieval,\n"
        "Q2,outra pergunta,DOC-2,relevant,retrieval,\n"
        "Q3,cobertura,NOT_IN_DB,irrelevant,coverage,\n",
        encoding="utf-8",
    )
    return str(gs)


@pytest.fixture
def evaluator(golden_set_file):
    return GenerationEvaluator(golden_set_path=golden_set_file)


class TestAnalyzeResponse:
    def test_detects_empty_response(self):
        r = analyze_response("Q1", "q", "Não encontrei essa informação.", [], 0, 0)
        assert r.is_empty is True

    def test_detects_non_empty(self):
        r = analyze_response("Q1", "q", "O artigo 5 estabelece regras.", [], 0, 0)
        assert r.is_empty is False

    def test_detects_citation(self):
        r = analyze_response("Q1", "q", "Conforme ICA-96-1-art563, ...", [], 0, 0)
        assert r.has_citation is True
        assert "ICA-96-1-art563" in r.citations_found

    def test_no_citation(self):
        r = analyze_response("Q1", "q", "A regra estabelece que ...", [], 0, 0)
        assert r.has_citation is False

    def test_detects_hedging(self):
        r = analyze_response("Q1", "q", "Possivelmente o artigo trata disso.", [], 0, 0)
        assert r.has_hedging is True

    def test_no_hedging(self):
        r = analyze_response("Q1", "q", "O artigo 10 define os tipos.", [], 0, 0)
        assert r.has_hedging is False

    def test_counts_tokens(self):
        r = analyze_response("Q1", "q", "um dois três quatro cinco", [], 0, 0)
        assert r.response_tokens == 5

    def test_multiple_citations(self):
        text = "Conforme ICA-96-1-art10 e ICA-100-47-art5, as regras são..."
        r = analyze_response("Q1", "q", text, [], 0, 0)
        assert len(r.citations_found) == 2

    def test_preserves_metadata(self):
        r = analyze_response("Q1", "q", "resp", ["DOC-1", "DOC-2"], 100, 200)
        assert r.query_id == "Q1"
        assert r.retrieved_doc_ids == ["DOC-1", "DOC-2"]
        assert r.search_time_ms == 100
        assert r.llm_time_ms == 200


class TestGenerationEvaluator:
    def test_loads_only_retrieval_queries(self, evaluator):
        assert len(evaluator.queries) == 2
        assert "Q3" not in evaluator.queries

    def test_missing_golden_set(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            GenerationEvaluator(golden_set_path=str(tmp_path / "nope.csv"))

    @patch("evaluation.evaluate_generation.LlamaModel")
    @patch("evaluation.evaluate_generation.QdrantManager")
    @patch("evaluation.evaluate_generation.EmbeddingModel")
    def test_evaluate_with_results(self, MockEmbed, MockQdrant, MockLlm, evaluator):
        MockEmbed.return_value.encode.return_value = np.random.rand(2, 1024)

        mock_point = MagicMock()
        mock_point.payload = {"regulation_id": "DOC-1", "text": "Artigo sobre voo."}
        mock_point.score = 0.9
        MockQdrant.return_value.search.return_value = [mock_point]

        MockLlm.return_value.generate_with_context.return_value = (
            "Conforme ICA-96-1-art10, o artigo estabelece as regras."
        )

        result = evaluator.evaluate(k=5)

        assert result.total_queries == 2
        assert result.citation_rate > 0
        assert result.empty_rate == 0

    @patch("evaluation.evaluate_generation.LlamaModel")
    @patch("evaluation.evaluate_generation.QdrantManager")
    @patch("evaluation.evaluate_generation.EmbeddingModel")
    def test_evaluate_no_results(self, MockEmbed, MockQdrant, MockLlm, evaluator):
        MockEmbed.return_value.encode.return_value = np.random.rand(2, 1024)
        MockQdrant.return_value.search.return_value = []

        result = evaluator.evaluate(k=5)

        assert result.empty_rate == 1.0
        for a in result.analyses:
            assert a.is_empty is True

    @patch("evaluation.evaluate_generation.LlamaModel")
    @patch("evaluation.evaluate_generation.QdrantManager")
    @patch("evaluation.evaluate_generation.EmbeddingModel")
    def test_evaluate_sample(self, MockEmbed, MockQdrant, MockLlm, evaluator):
        MockEmbed.return_value.encode.return_value = np.random.rand(1, 1024)
        MockQdrant.return_value.search.return_value = []

        result = evaluator.evaluate(k=1, sample=1)
        assert result.total_queries == 1

    @patch("evaluation.evaluate_generation.LlamaModel")
    @patch("evaluation.evaluate_generation.QdrantManager")
    @patch("evaluation.evaluate_generation.EmbeddingModel")
    def test_save_results(self, MockEmbed, MockQdrant, MockLlm, evaluator, tmp_path):
        MockEmbed.return_value.encode.return_value = np.random.rand(2, 1024)
        MockQdrant.return_value.search.return_value = []

        result = evaluator.evaluate(k=1)
        summary, details, json_path = evaluator.save_results(result, str(tmp_path))

        assert Path(summary).exists()
        assert Path(details).exists()
        assert Path(json_path).exists()

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "empty_rate" in data["metrics"]
        assert len(data["analyses"]) == 2


class TestPrintReport:
    @patch("evaluation.evaluate_generation.LlamaModel")
    @patch("evaluation.evaluate_generation.QdrantManager")
    @patch("evaluation.evaluate_generation.EmbeddingModel")
    def test_print_report_runs(self, MockEmbed, MockQdrant, MockLlm, evaluator, capsys):
        MockEmbed.return_value.encode.return_value = np.random.rand(2, 1024)

        mock_point = MagicMock()
        mock_point.payload = {"regulation_id": "D1", "text": "t"}
        mock_point.score = 0.9
        MockQdrant.return_value.search.return_value = [mock_point]

        MockLlm.return_value.generate_with_context.return_value = "Possivelmente algo."

        result = evaluator.evaluate(k=1)
        print_report(result)

        captured = capsys.readouterr()
        assert "GENERATION QUALITY" in captured.out
        assert "HEDGING" in captured.out
