"""Tests for RAG pipeline."""

from unittest.mock import MagicMock, patch

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


class TestQueryResponseFormat:
    def test_returns_expected_keys(self, rag, mock_search, mock_llm):
        mock_search.search.return_value = [
            {"text": "Art. 1", "regulation_id": "doc-1", "score": 0.9}
        ]
        mock_llm.generate_with_context.return_value = "Resposta gerada."

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
        mock_llm.generate_with_context.return_value = "Temporal."

        result = rag.query("teste", date="2023-05-15", limit=1)

        mock_search.search_temporal.assert_called_once_with("teste", "2023-05-15", limit=1)
        assert result["answer"] == "Temporal."


class TestSearchBackendErrorHandling:
    def test_returns_service_unavailable(self, rag, mock_search):
        mock_search.search.side_effect = SearchBackendError("down")

        result = rag.query("teste")

        assert "indisponível" in result["answer"]
        assert result["sources"] == []


class TestResponseCache:
    def test_cache_hit_skips_search_and_llm(self, mock_search, mock_llm):
        cache = InMemoryCache()
        rag = RAGPipeline(search=mock_search, llm=mock_llm, response_cache=cache)

        mock_search.search.return_value = SAMPLE_RESULTS
        mock_llm.generate_with_context.return_value = "Resposta."

        r1 = rag.query("mesma pergunta", limit=5)
        r2 = rag.query("mesma pergunta", limit=5)

        assert r1["answer"] == r2["answer"]
        mock_search.search.assert_called_once()
        mock_llm.generate_with_context.assert_called_once()
        assert cache.stats()["hits"] == 1

    def test_different_params_miss(self, mock_search, mock_llm):
        cache = InMemoryCache()
        rag = RAGPipeline(search=mock_search, llm=mock_llm, response_cache=cache)

        mock_search.search.return_value = SAMPLE_RESULTS
        mock_llm.generate_with_context.return_value = "R"

        rag.query("q", limit=3)
        rag.query("q", limit=5)

        assert mock_search.search.call_count == 2

    def test_no_cache_always_executes(self, mock_search, mock_llm):
        rag = RAGPipeline(search=mock_search, llm=mock_llm)

        mock_search.search.return_value = SAMPLE_RESULTS
        mock_llm.generate_with_context.return_value = "R"

        rag.query("q")
        rag.query("q")

        assert mock_search.search.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
