"""Tests for the DocumentSearcher module."""

from unittest.mock import MagicMock, patch

import pytest

from search.searcher.searcher import DocumentSearcher
from search.shared.exceptions import SearchBackendError
from search.shared.schemas import (
    RewrittenQuery,
    SearchFilter,
    SearchSort,
    FilterOperator,
    SortOrder,
)


@pytest.fixture
def mock_vector_search():
    vs = MagicMock()
    vs.search.return_value = [
        {"text": "Art. 1", "regulation_id": "doc-1", "score": 0.9},
        {"text": "Art. 2", "regulation_id": "doc-2", "score": 0.8},
    ]
    return vs


@pytest.fixture
def searcher(mock_vector_search):
    return DocumentSearcher(vector_search=mock_vector_search, default_limit=5)


class TestSearchBasic:
    def test_single_query_search(self, searcher, mock_vector_search):
        queries = [RewrittenQuery(text="test query")]
        result = searcher.search(queries)

        assert result.total_after_dedup == 2
        assert len(result.documents) == 2
        mock_vector_search.search.assert_called_once()

    def test_results_per_query_tracked(self, searcher):
        queries = [RewrittenQuery(text="my query")]
        result = searcher.search(queries)

        assert "my query" in result.results_per_query
        assert result.results_per_query["my query"] == 2


class TestParallelSearch:
    def test_multiple_queries_search(self, searcher, mock_vector_search):
        queries = [
            RewrittenQuery(text="query 1"),
            RewrittenQuery(text="query 2"),
        ]
        result = searcher.search(queries)

        assert mock_vector_search.search.call_count == 2
        assert result.total_before_dedup == 4

    def test_partial_failure_returns_results(self, searcher, mock_vector_search):
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [{"text": "ok", "regulation_id": "doc-1", "score": 0.9}]
            raise Exception("Search error")

        mock_vector_search.search.side_effect = side_effect

        queries = [
            RewrittenQuery(text="good query"),
            RewrittenQuery(text="bad query"),
        ]
        result = searcher.search(queries)

        assert len(result.documents) >= 1


class TestDeduplication:
    def test_deduplicates_by_regulation_id_and_text(self, searcher, mock_vector_search):
        mock_vector_search.search.return_value = [
            {"text": "Art. 1 text here", "regulation_id": "doc-1", "score": 0.9},
            {"text": "Art. 1 text here", "regulation_id": "doc-1", "score": 0.7},
        ]

        queries = [RewrittenQuery(text="q1"), RewrittenQuery(text="q2")]
        result = searcher.search(queries)

        doc_ids_with_text = [
            f"{d['regulation_id']}:{d['text'][:100]}" for d in result.documents
        ]
        assert len(set(doc_ids_with_text)) == len(result.documents)

    def test_keeps_highest_score(self, searcher, mock_vector_search):
        mock_vector_search.search.return_value = [
            {"text": "Same text content", "regulation_id": "doc-1", "score": 0.5},
            {"text": "Same text content", "regulation_id": "doc-1", "score": 0.95},
        ]

        queries = [RewrittenQuery(text="q")]
        result = searcher.search(queries)

        assert result.documents[0]["score"] == 0.95


class TestFilterConversion:
    def test_queries_with_filters_pass_to_vector_search(
        self, searcher, mock_vector_search,
    ):
        queries = [
            RewrittenQuery(
                text="test",
                filters=[
                    SearchFilter(field="metadata.type", operator=FilterOperator.EQ, value="ICA"),
                ],
            ),
        ]
        searcher.search(queries)

        call_kwargs = mock_vector_search.search.call_args
        assert call_kwargs is not None


class TestSearchBackendFailure:
    def test_raises_on_backend_error(self, searcher, mock_vector_search):
        mock_vector_search.search.side_effect = SearchBackendError("Qdrant down")

        queries = [RewrittenQuery(text="test")]

        with pytest.raises(SearchBackendError):
            searcher.search(queries)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
