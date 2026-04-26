"""Tests for the modular RAG pipeline orchestrator.

Adapted from the original test_rag.py to work with the new modular
architecture (Rewriter -> Searcher -> Evaluator -> Generator).
"""

from unittest.mock import MagicMock

import pytest

from search.orchestrator.pipeline import RAGPipeline
from search.cache import InMemoryCache
from search.shared.exceptions import SearchBackendError
from search.shared.schemas import (
    EvaluatedDocument,
    RewrittenQuery,
    SearchResults,
)


@pytest.fixture
def mock_rewriter():
    rw = MagicMock()
    rw.rewrite.return_value = [
        RewrittenQuery(text="teste reescrito", facet_type="general")
    ]
    return rw


@pytest.fixture
def mock_searcher():
    s = MagicMock()
    s.search.return_value = SearchResults(
        documents=[
            {"text": "Art. 1", "regulation_id": "doc-1", "score": 0.9}
        ],
        results_per_query={"teste reescrito": 1},
        total_before_dedup=1,
        total_after_dedup=1,
    )
    return s


@pytest.fixture
def mock_evaluator():
    ev = MagicMock()
    ev.max_eval_tokens = 480
    ev.evaluate.return_value = [
        EvaluatedDocument(
            document={"text": "Art. 1", "regulation_id": "doc-1", "score": 0.9},
            relevance_score=85.0,
            query_text="teste reescrito",
        )
    ]
    return ev


@pytest.fixture
def mock_generator():
    gen = MagicMock()
    gen.generate.return_value = "Resposta gerada."
    gen.llm = MagicMock()
    gen.llm.model_name = "test-model"
    gen.grounded_only = True
    return gen


@pytest.fixture
def rag(mock_rewriter, mock_searcher, mock_evaluator, mock_generator):
    return RAGPipeline(
        rewriter=mock_rewriter,
        searcher=mock_searcher,
        evaluator=mock_evaluator,
        generator=mock_generator,
    )


SAMPLE_HISTORY = [
    {"role": "user", "content": "O que é licitação?"},
    {"role": "assistant", "content": "Licitação é um procedimento..."},
]


class TestQueryResponseFormat:
    def test_returns_expected_keys(self, rag):
        result = rag.query("teste", limit=1)

        assert "answer" in result
        assert "sources" in result
        assert "total_time_ms" in result
        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)

    def test_empty_search_results_returns_default_answer(
        self, rag, mock_searcher,
    ):
        mock_searcher.search.return_value = SearchResults(
            documents=[],
            results_per_query={},
            total_before_dedup=0,
            total_after_dedup=0,
        )

        result = rag.query("nada relevante", limit=1)

        assert "Não encontrei" in result["answer"]
        assert result["sources"] == []
        assert result["llm_time_ms"] == 0


class TestSearchBackendErrorHandling:
    def test_returns_service_unavailable(self, rag, mock_searcher):
        mock_searcher.search.side_effect = SearchBackendError("down")

        result = rag.query("teste")

        assert "indisponível" in result["answer"]
        assert result["sources"] == []


class TestPipelineStages:
    def test_rewriter_is_called(self, rag, mock_rewriter):
        rag.query("minha pergunta")
        mock_rewriter.rewrite.assert_called_once()
        args = mock_rewriter.rewrite.call_args
        assert args[0][0] == "minha pergunta"

    def test_searcher_receives_rewritten_queries(self, rag, mock_searcher):
        rag.query("pergunta")
        mock_searcher.search.assert_called_once()
        queries = mock_searcher.search.call_args[0][0]
        assert len(queries) == 1
        assert queries[0].text == "teste reescrito"

    def test_evaluator_receives_documents(self, rag, mock_evaluator):
        rag.query("pergunta")
        mock_evaluator.evaluate.assert_called_once()

    def test_generator_receives_evaluated_docs(self, rag, mock_generator):
        rag.query("pergunta")
        mock_generator.generate.assert_called_once()
        args = mock_generator.generate.call_args
        evaluated = args[0][0]
        assert len(evaluated) == 1
        assert evaluated[0].relevance_score == 85.0


class TestDebugTrace:
    def test_trace_not_returned_by_default(self, rag):
        result = rag.query("teste")
        assert "trace" not in result

    def test_trace_returned_when_debug_enabled(self, rag):
        result = rag.query("teste", debug=True)
        assert "trace" in result
        trace = result["trace"]
        assert trace["original_query"] == "teste"
        assert len(trace["rewritten_queries"]) == 1
        assert "timings" in trace

    def test_trace_contains_evaluation_scores(self, rag):
        result = rag.query("teste", debug=True)
        trace = result["trace"]
        assert "evaluation_scores" in trace
        assert trace["documents_accepted"] == 1

    def test_trace_evaluation_scores_include_eval_text(self, rag):
        """Each evaluation entry must carry the exact text the
        cross-encoder scored, so the offline extractor can surface it."""
        result = rag.query("teste", debug=True)
        scores = result["trace"]["evaluation_scores"]
        assert scores, "expected at least one evaluation entry"
        for entry in scores:
            assert "eval_text" in entry
            assert "eval_max_tokens" in entry
            assert entry["eval_max_tokens"] == 480
        accepted = next(e for e in scores if e["accepted"])
        # _build_eval_text prepends metadata; for our mock doc we only
        # have ``text`` so the eval_text must at least contain it.
        assert "Art. 1" in accepted["eval_text"]


class TestStreamingMode:
    def test_query_stream_returns_answer_stream(self, rag, mock_generator):
        mock_generator.generate.return_value = iter(["chunk1", "chunk2"])

        result = rag.query("q", stream=True)

        assert "answer_stream" in result
        assert "sources" in result
        assert "search_time_ms" in result
        assert "answer" not in result

    def test_stream_no_results_returns_answer_not_stream(
        self, rag, mock_searcher,
    ):
        mock_searcher.search.return_value = SearchResults(
            documents=[], results_per_query={},
            total_before_dedup=0, total_after_dedup=0,
        )

        result = rag.query("q", stream=True)

        assert "answer" in result
        assert "answer_stream" not in result


class TestResponseCache:
    def test_cache_hit_skips_pipeline(
        self, mock_rewriter, mock_searcher, mock_evaluator, mock_generator,
    ):
        cache = InMemoryCache()
        rag = RAGPipeline(
            rewriter=mock_rewriter,
            searcher=mock_searcher,
            evaluator=mock_evaluator,
            generator=mock_generator,
            response_cache=cache,
        )

        r1 = rag.query("mesma pergunta", limit=5)
        r2 = rag.query("mesma pergunta", limit=5)

        assert r1["answer"] == r2["answer"]
        assert mock_rewriter.rewrite.call_count == 1
        assert cache.stats()["hits"] == 1

    def test_cache_skipped_with_debug(
        self, mock_rewriter, mock_searcher, mock_evaluator, mock_generator,
    ):
        cache = InMemoryCache()
        rag = RAGPipeline(
            rewriter=mock_rewriter,
            searcher=mock_searcher,
            evaluator=mock_evaluator,
            generator=mock_generator,
            response_cache=cache,
        )

        rag.query("q", debug=True)
        rag.query("q", debug=True)

        assert mock_rewriter.rewrite.call_count == 2
        assert cache.stats()["hits"] == 0


class TestGroundedOnlyPassthrough:
    def test_grounded_only_passed_to_generator(self, rag, mock_generator):
        rag.query("teste", grounded_only=False)
        _, kwargs = mock_generator.generate.call_args
        assert kwargs["grounded_only"] is False


class TestRewriterDisabled:
    def test_skips_rewriter_when_disabled(
        self, mock_rewriter, mock_searcher, mock_evaluator, mock_generator,
    ):
        rag = RAGPipeline(
            rewriter=mock_rewriter,
            searcher=mock_searcher,
            evaluator=mock_evaluator,
            generator=mock_generator,
            rewriter_enabled=False,
        )

        result = rag.query("query original")

        mock_rewriter.rewrite.assert_not_called()
        queries = mock_searcher.search.call_args[0][0]
        assert len(queries) == 1
        assert queries[0].text == "query original"
        assert queries[0].facet_type == "passthrough"
        assert "answer" in result

    def test_trace_shows_passthrough(
        self, mock_rewriter, mock_searcher, mock_evaluator, mock_generator,
    ):
        rag = RAGPipeline(
            rewriter=mock_rewriter,
            searcher=mock_searcher,
            evaluator=mock_evaluator,
            generator=mock_generator,
            rewriter_enabled=False,
        )

        result = rag.query("q", debug=True)
        trace = result["trace"]
        assert trace["rewritten_queries"][0]["facet_type"] == "passthrough"


class TestEvaluatorDisabled:
    def test_skips_evaluator_when_disabled(
        self, mock_rewriter, mock_searcher, mock_evaluator, mock_generator,
    ):
        rag = RAGPipeline(
            rewriter=mock_rewriter,
            searcher=mock_searcher,
            evaluator=mock_evaluator,
            generator=mock_generator,
            evaluator_enabled=False,
        )

        result = rag.query("pergunta")

        mock_evaluator.evaluate.assert_not_called()
        mock_evaluator.evaluate_multi_query.assert_not_called()
        assert "answer" in result

    def test_passes_all_docs_to_generator(
        self, mock_rewriter, mock_searcher, mock_evaluator, mock_generator,
    ):
        mock_searcher.search.return_value = SearchResults(
            documents=[
                {"text": "doc 1", "regulation_id": "d1", "score": 0.9},
                {"text": "doc 2", "regulation_id": "d2", "score": 0.7},
            ],
            results_per_query={"q": 2},
            total_before_dedup=2,
            total_after_dedup=2,
        )

        rag = RAGPipeline(
            rewriter=mock_rewriter,
            searcher=mock_searcher,
            evaluator=mock_evaluator,
            generator=mock_generator,
            evaluator_enabled=False,
        )

        rag.query("pergunta")
        args = mock_generator.generate.call_args[0]
        evaluated = args[0]
        assert len(evaluated) == 2

    def test_uses_vector_score_when_disabled(
        self, mock_rewriter, mock_searcher, mock_evaluator, mock_generator,
    ):
        mock_searcher.search.return_value = SearchResults(
            documents=[
                {"text": "doc 1", "regulation_id": "d1", "score": 0.85},
            ],
            results_per_query={"q": 1},
            total_before_dedup=1,
            total_after_dedup=1,
        )

        rag = RAGPipeline(
            rewriter=mock_rewriter,
            searcher=mock_searcher,
            evaluator=mock_evaluator,
            generator=mock_generator,
            evaluator_enabled=False,
        )

        rag.query("pergunta")
        args = mock_generator.generate.call_args[0]
        evaluated = args[0]
        assert evaluated[0].relevance_score == pytest.approx(85.0)


class TestBothDisabled:
    def test_pipeline_works_with_both_disabled(
        self, mock_rewriter, mock_searcher, mock_evaluator, mock_generator,
    ):
        rag = RAGPipeline(
            rewriter=mock_rewriter,
            searcher=mock_searcher,
            evaluator=mock_evaluator,
            generator=mock_generator,
            rewriter_enabled=False,
            evaluator_enabled=False,
        )

        result = rag.query("pergunta direta")

        mock_rewriter.rewrite.assert_not_called()
        mock_evaluator.evaluate.assert_not_called()
        assert "answer" in result

        queries = mock_searcher.search.call_args[0][0]
        assert queries[0].text == "pergunta direta"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
