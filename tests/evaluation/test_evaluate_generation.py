"""Unit tests for generation evaluation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from evaluation.evaluate_generation import (
    GenerationEvaluator,
    analyze_response,
    compute_grounding,
    extract_citations,
    print_report,
)
from evaluation.llm_judge import JudgeVerdict, _parse_verdict


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

    @patch("evaluation.evaluate_generation.create_llm")
    @patch("evaluation.evaluate_generation.QdrantManager")
    @patch("evaluation.evaluate_generation.create_embedding_model")
    def test_evaluate_with_results(self, MockCreateEmbed, MockQdrant, MockCreateLlm, evaluator):
        MockCreateEmbed.return_value.encode.return_value = np.random.rand(2, 1024)

        mock_point = MagicMock()
        mock_point.payload = {"regulation_id": "DOC-1", "text": "Artigo sobre voo."}
        mock_point.score = 0.9
        MockQdrant.return_value.search.return_value = [mock_point]

        MockCreateLlm.return_value.generate_with_context.return_value = (
            "Conforme ICA-96-1-art10, o artigo estabelece as regras."
        )

        result = evaluator.evaluate(k=5)

        assert result.total_queries == 2
        assert result.citation_rate > 0
        assert result.empty_rate == 0

    @patch("evaluation.evaluate_generation.create_llm")
    @patch("evaluation.evaluate_generation.QdrantManager")
    @patch("evaluation.evaluate_generation.create_embedding_model")
    def test_evaluate_no_results(self, MockCreateEmbed, MockQdrant, MockCreateLlm, evaluator):
        MockCreateEmbed.return_value.encode.return_value = np.random.rand(2, 1024)
        MockQdrant.return_value.search.return_value = []

        result = evaluator.evaluate(k=5)

        assert result.empty_rate == 1.0
        for a in result.analyses:
            assert a.is_empty is True

    @patch("evaluation.evaluate_generation.create_llm")
    @patch("evaluation.evaluate_generation.QdrantManager")
    @patch("evaluation.evaluate_generation.create_embedding_model")
    def test_evaluate_sample(self, MockCreateEmbed, MockQdrant, MockCreateLlm, evaluator):
        MockCreateEmbed.return_value.encode.return_value = np.random.rand(1, 1024)
        MockQdrant.return_value.search.return_value = []

        result = evaluator.evaluate(k=1, sample=1)
        assert result.total_queries == 1

    @patch("evaluation.evaluate_generation.create_llm")
    @patch("evaluation.evaluate_generation.QdrantManager")
    @patch("evaluation.evaluate_generation.create_embedding_model")
    def test_save_results(self, MockCreateEmbed, MockQdrant, MockCreateLlm, evaluator, tmp_path):
        MockCreateEmbed.return_value.encode.return_value = np.random.rand(2, 1024)
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

    @patch("evaluation.evaluate_generation.config")
    @patch("evaluation.evaluate_generation.SparseEncoder")
    @patch("evaluation.evaluate_generation.create_llm")
    @patch("evaluation.evaluate_generation.QdrantManager")
    @patch("evaluation.evaluate_generation.create_embedding_model")
    def test_evaluate_hybrid_mode(self, MockCreateEmbed, MockQdrant, MockCreateLlm, MockSparse, mock_config, evaluator):
        """When SEARCH_SPARSE_ENABLED=True, evaluation should use hybrid search."""
        mock_config.SEARCH_DENSE_ENABLED = True
        mock_config.SEARCH_SPARSE_ENABLED = True

        MockCreateEmbed.return_value.encode.return_value = np.random.rand(2, 1024)
        MockSparse.return_value.encode.return_value = [{"indices": [0], "values": [1.0]}] * 2

        mock_point = MagicMock()
        mock_point.payload = {"regulation_id": "DOC-1", "text": "Artigo."}
        mock_point.score = 0.9
        MockQdrant.return_value.search.return_value = [mock_point]

        MockCreateLlm.return_value.generate_with_context.return_value = "Conforme ICA-96-1, regras."

        result = evaluator.evaluate(k=5, sample=1)
        assert result.total_queries == 1

        call_kwargs = MockQdrant.return_value.search.call_args
        assert "dense_vector" in call_kwargs.kwargs or "dense_vector" in (call_kwargs[1] if len(call_kwargs) > 1 else {})


class TestCitationExtractionAndGrounding:
    def test_extracts_bracketed_citations(self):
        text = "Conforme [ICA 100-40], complementado por [MCA 56-5]."
        cites = extract_citations(text)
        assert "ICA 100-40" in cites
        assert "MCA 56-5" in cites

    def test_extracts_bracketed_with_article(self):
        text = "Ver [ICA 100-40-art10] e [Decreto 97.464]."
        cites = extract_citations(text)
        assert "ICA 100-40-art10" in cites
        assert "Decreto 97.464" in cites

    def test_falls_back_to_unbracketed_when_no_brackets(self):
        text = "Conforme ICA-96-1-art10 e ICA-100-47, ..."
        cites = extract_citations(text)
        assert len(cites) >= 1

    def test_grounding_marks_cited_doc_as_grounded(self):
        cites = ["ICA 100-40"]
        retrieved = ["ica_100-40/2023-art1", "ica_7-58/2020-art1"]
        result = compute_grounding(cites, retrieved)
        assert result["grounded"] == 1
        assert result["hallucinated"] == 0

    def test_grounding_flags_hallucinated_citation(self):
        cites = ["ICA 999-99"]
        retrieved = ["ica_100-40/2023-art1"]
        result = compute_grounding(cites, retrieved)
        assert result["grounded"] == 0
        assert result["hallucinated"] == 1
        assert "ICA 999-99" in result["hallucinated_list"]

    def test_grounding_doc_level_matches_any_chunk(self):
        cites = ["ICA 7-58"]
        retrieved = ["ica_7-58/2020-art2-0", "ica_7-58/2020-art5-1"]
        result = compute_grounding(cites, retrieved)
        assert result["grounded"] == 1

    def test_analyze_response_populates_grounding_fields(self):
        retrieved = ["ica_100-40/2023-art1"]
        text = "Conforme [ICA 100-40], a regra é... também ver [ICA 999-99]."
        a = analyze_response("Q1", "q", text, retrieved, 0, 0)
        assert a.has_citation is True
        assert a.citations_grounded == 1
        assert a.citations_hallucinated == 1
        assert "ICA 999-99" in a.hallucinated_citations


class TestLLMJudgeParseVerdict:
    def test_parses_clean_json(self):
        v = _parse_verdict('{"score": 4, "reasoning": "good"}')
        assert v.score == pytest.approx(0.8)
        assert v.raw_score == 4
        assert v.parse_error is None

    def test_normalizes_score_to_unit_interval(self):
        for raw, expected in [(0, 0.0), (3, 0.6), (5, 1.0)]:
            v = _parse_verdict(f'{{"score": {raw}, "reasoning": "x"}}')
            assert v.score == pytest.approx(expected)

    def test_rejects_out_of_range_score(self):
        v = _parse_verdict('{"score": 9, "reasoning": "x"}')
        assert v.score is None
        assert v.parse_error is not None

    def test_handles_code_fence_wrap(self):
        v = _parse_verdict('```json\n{"score": 2, "reasoning": "ok"}\n```')
        assert v.raw_score == 2

    def test_invalid_json_returns_parse_error(self):
        v = _parse_verdict("not json at all")
        assert v.score is None
        assert "json_decode" in v.parse_error


class TestEvaluatePipelineMode:
    @patch("search.orchestrator.pipeline.RAGPipeline")
    def test_pipeline_mode_collects_grounding(self, MockPipeline, evaluator):
        mock_pipeline = MockPipeline.return_value

        def fake_query(question, **kwargs):
            return {
                "answer": "Conforme [ICA 100-40], a regra é tal.",
                "sources": [{"regulation_id": "ica_100-40/2023-art1", "score": 5.0}],
                "trace": {
                    "rewritten_queries": [{"text": "x"}],
                    "documents_after_dedup": 5,
                    "documents_accepted": 3,
                    "timings": {
                        "rewriter_ms": 100, "searcher_ms": 10,
                        "evaluator_ms": 50, "generator_ms": 500, "total_ms": 660,
                    },
                },
            }

        mock_pipeline.query.side_effect = fake_query
        result = evaluator.evaluate_pipeline(k=5)

        assert result.mode == "pipeline"
        assert result.judge_used is False
        assert result.citation_grounding_rate == pytest.approx(1.0)
        assert result.mean_citations_per_response == pytest.approx(1.0)
        for a in result.analyses:
            assert a.citations_grounded == 1
            assert a.citations_hallucinated == 0

    @patch("search.orchestrator.pipeline.RAGPipeline")
    def test_pipeline_mode_with_judge_populates_scores(self, MockPipeline, evaluator):
        mock_pipeline = MockPipeline.return_value
        mock_pipeline.query.return_value = {
            "answer": "Conforme [ICA 100-40], regra X.",
            "sources": [{"regulation_id": "ica_100-40/2023-art1"}],
            "trace": {
                "rewritten_queries": [],
                "documents_after_dedup": 1,
                "documents_accepted": 1,
                "timings": {
                    "rewriter_ms": 0, "searcher_ms": 0,
                    "evaluator_ms": 0, "generator_ms": 0, "total_ms": 0,
                },
            },
        }

        judge = MagicMock()
        judge.model_name = "test-judge-model"
        judge.judge_faithfulness.return_value = JudgeVerdict(
            score=0.8, raw_score=4, reasoning="grounded ok",
        )
        judge.judge_relevance.return_value = JudgeVerdict(
            score=1.0, raw_score=5, reasoning="on topic",
        )

        result = evaluator.evaluate_pipeline(k=1, judge=judge)

        assert result.judge_used is True
        assert result.judge_model == "test-judge-model"
        assert result.mean_faithfulness == pytest.approx(0.8)
        assert result.mean_relevance == pytest.approx(1.0)
        assert result.judge_unparseable_rate == 0.0
        for a in result.analyses:
            assert a.faithfulness_score == pytest.approx(0.8)
            assert a.relevance_score == pytest.approx(1.0)

    @patch("search.orchestrator.pipeline.RAGPipeline")
    def test_pipeline_mode_judge_unparseable_tracked(self, MockPipeline, evaluator):
        MockPipeline.return_value.query.return_value = {
            "answer": "x",
            "sources": [],
            "trace": {
                "rewritten_queries": [],
                "documents_after_dedup": 0,
                "documents_accepted": 0,
                "timings": {
                    "rewriter_ms": 0, "searcher_ms": 0,
                    "evaluator_ms": 0, "generator_ms": 0, "total_ms": 0,
                },
            },
        }

        judge = MagicMock()
        judge.model_name = "m"
        judge.judge_faithfulness.return_value = JudgeVerdict(
            score=None, raw_score=None, reasoning="", parse_error="x",
        )
        judge.judge_relevance.return_value = JudgeVerdict(
            score=None, raw_score=None, reasoning="", parse_error="x",
        )

        result = evaluator.evaluate_pipeline(k=1, judge=judge)
        assert result.mean_faithfulness is None
        assert result.judge_unparseable_rate == 1.0


class TestPrintReport:
    @patch("evaluation.evaluate_generation.create_llm")
    @patch("evaluation.evaluate_generation.QdrantManager")
    @patch("evaluation.evaluate_generation.create_embedding_model")
    def test_print_report_runs(self, MockCreateEmbed, MockQdrant, MockCreateLlm, evaluator, capsys):
        MockCreateEmbed.return_value.encode.return_value = np.random.rand(2, 1024)

        mock_point = MagicMock()
        mock_point.payload = {"regulation_id": "D1", "text": "t"}
        mock_point.score = 0.9
        MockQdrant.return_value.search.return_value = [mock_point]

        MockCreateLlm.return_value.generate_with_context.return_value = "Possivelmente algo."

        result = evaluator.evaluate(k=1)
        print_report(result)

        captured = capsys.readouterr()
        assert "GENERATION QUALITY" in captured.out
        assert "HEDGING" in captured.out
