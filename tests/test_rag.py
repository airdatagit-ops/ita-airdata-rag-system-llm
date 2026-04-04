"""Tests for RAG pipeline."""

from unittest.mock import MagicMock, patch, call

import pytest

from search.rag import RAGPipeline
from search.cache import InMemoryCache
from search.exceptions import SearchBackendError


@pytest.fixture
def mock_search():
    return MagicMock()


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def rag(mock_search, mock_llm):
    return RAGPipeline(search=mock_search, llm=mock_llm)


SAMPLE_RESULTS = [
    {"text": "Art. 1", "regulation_id": "doc-1", "score": 0.9}
]

SAMPLE_HISTORY = [
    {"role": "user", "content": "O que é licitação?"},
    {"role": "assistant", "content": "Licitação é um procedimento..."},
]


class TestQueryResponseFormat:
    def test_returns_expected_keys(self, rag, mock_search, mock_llm):
        mock_search.search.return_value = SAMPLE_RESULTS
        mock_llm.generate.return_value = "Resposta gerada."

        result = rag.query("teste", limit=1)

        assert "answer" in result
        assert "sources" in result
        assert "total_time_ms" in result
        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)

    def test_empty_results_returns_default_answer(self, rag, mock_search):
        mock_search.search.return_value = []

        result = rag.query("nada relevante", limit=1)

        assert "Não encontrei" in result["answer"]
        assert result["sources"] == []
        assert result["llm_time_ms"] == 0


class TestTemporalQuery:
    def test_delegates_to_search_temporal(self, rag, mock_search, mock_llm):
        mock_search.search_temporal.return_value = [
            {"text": "Art. 5", "regulation_id": "doc-5", "score": 0.8}
        ]
        mock_llm.generate.return_value = "Temporal."

        result = rag.query("teste", date="2023-05-15", limit=1)

        mock_search.search_temporal.assert_called_once_with("teste", "2023-05-15", limit=1)
        assert result["answer"] == "Temporal."


class TestSearchBackendErrorHandling:
    def test_returns_service_unavailable(self, rag, mock_search):
        mock_search.search.side_effect = SearchBackendError("down")

        result = rag.query("teste")

        assert "indisponível" in result["answer"]
        assert result["sources"] == []


class TestPromptSelection:
    """Verify the pipeline picks the right prompt based on history."""

    @patch("search.rag.build_rag_prompt")
    def test_query_without_history_uses_rag_prompt(
        self, mock_build_rag, rag, mock_search, mock_llm,
    ):
        mock_search.search.return_value = SAMPLE_RESULTS
        mock_build_rag.return_value = "rag-prompt"
        mock_llm.generate.return_value = "R"

        rag.query("pergunta")

        mock_build_rag.assert_called_once()
        assert "pergunta" in str(mock_build_rag.call_args)

    @patch("search.rag.build_chat_prompt")
    @patch("search.rag.build_search_query")
    def test_query_with_history_uses_chat_prompt(
        self, mock_bsq, mock_bcp, rag, mock_search, mock_llm,
    ):
        mock_bsq.return_value = "enriched query"
        mock_search.search.return_value = SAMPLE_RESULTS
        mock_bcp.return_value = "chat-prompt"
        mock_llm.generate.return_value = "R"

        rag.query("nova pergunta", history=SAMPLE_HISTORY)

        mock_bcp.assert_called_once()
        mock_bsq.assert_called_once_with("nova pergunta", SAMPLE_HISTORY)

    @patch("search.rag.build_search_query")
    def test_search_query_enriched_with_history(
        self, mock_bsq, rag, mock_search, mock_llm,
    ):
        mock_bsq.return_value = "enriched"
        mock_search.search.return_value = SAMPLE_RESULTS
        mock_llm.generate.return_value = "R"

        rag.query("atual", history=SAMPLE_HISTORY)

        mock_bsq.assert_called_once_with("atual", SAMPLE_HISTORY)
        mock_search.search.assert_called_once_with("enriched", limit=5)


class TestStreamingMode:
    def test_query_stream_returns_answer_stream(self, rag, mock_search, mock_llm):
        mock_search.search.return_value = SAMPLE_RESULTS
        mock_llm.generate.return_value = iter(["chunk1", "chunk2"])

        result = rag.query("q", stream=True)

        assert "answer_stream" in result
        assert "sources" in result
        assert "search_time_ms" in result
        assert "answer" not in result

    def test_stream_passes_params_to_llm(self, rag, mock_search, mock_llm):
        mock_search.search.return_value = SAMPLE_RESULTS
        mock_llm.generate.return_value = iter([])

        rag.query("q", stream=True, temperature=0.5, max_tokens=100)

        _, kwargs = mock_llm.generate.call_args
        assert kwargs["stream"] is True
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 100

    def test_stream_no_results_returns_answer_not_stream(self, rag, mock_search):
        mock_search.search.return_value = []

        result = rag.query("q", stream=True)

        assert "answer" in result
        assert "answer_stream" not in result


class TestResponseCache:
    def test_cache_hit_skips_search_and_llm(self, mock_search, mock_llm):
        cache = InMemoryCache()
        rag = RAGPipeline(search=mock_search, llm=mock_llm, response_cache=cache)

        mock_search.search.return_value = SAMPLE_RESULTS
        mock_llm.generate.return_value = "Resposta."

        r1 = rag.query("mesma pergunta", limit=5)
        r2 = rag.query("mesma pergunta", limit=5)

        assert r1["answer"] == r2["answer"]
        mock_search.search.assert_called_once()
        mock_llm.generate.assert_called_once()
        assert cache.stats()["hits"] == 1

    def test_different_params_miss(self, mock_search, mock_llm):
        cache = InMemoryCache()
        rag = RAGPipeline(search=mock_search, llm=mock_llm, response_cache=cache)

        mock_search.search.return_value = SAMPLE_RESULTS
        mock_llm.generate.return_value = "R"

        rag.query("q", limit=3)
        rag.query("q", limit=5)

        assert mock_search.search.call_count == 2

    def test_no_cache_always_executes(self, mock_search, mock_llm):
        rag = RAGPipeline(search=mock_search, llm=mock_llm)

        mock_search.search.return_value = SAMPLE_RESULTS
        mock_llm.generate.return_value = "R"

        rag.query("q")
        rag.query("q")

        assert mock_search.search.call_count == 2

    def test_cache_skipped_with_history(self, mock_search, mock_llm):
        cache = InMemoryCache()
        rag = RAGPipeline(search=mock_search, llm=mock_llm, response_cache=cache)

        mock_search.search.return_value = SAMPLE_RESULTS
        mock_llm.generate.return_value = "R"

        rag.query("q", history=SAMPLE_HISTORY)
        rag.query("q", history=SAMPLE_HISTORY)

        assert mock_search.search.call_count == 2
        assert mock_llm.generate.call_count == 2
        assert cache.stats()["hits"] == 0

    def test_cache_skipped_with_stream(self, mock_search, mock_llm):
        cache = InMemoryCache()
        rag = RAGPipeline(search=mock_search, llm=mock_llm, response_cache=cache)

        mock_search.search.return_value = SAMPLE_RESULTS
        mock_llm.generate.return_value = iter([])

        rag.query("q", stream=True)
        rag.query("q", stream=True)

        assert mock_search.search.call_count == 2
        assert cache.stats()["hits"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
