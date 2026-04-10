"""Tests for the QueryRewriter module."""

import json
from unittest.mock import MagicMock

import pytest

from search.rewriter.rewriter import QueryRewriter
from search.shared.schemas import RewrittenQuery


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.model_name = "test-model"
    return llm


@pytest.fixture
def rewriter(mock_llm):
    return QueryRewriter(llm=mock_llm, max_queries=3, timeout=5)


VALID_LLM_RESPONSE = json.dumps([
    {
        "text": "normas sobre certificação de pilotos",
        "filters": [{"field": "metadata.type", "operator": "eq", "value": "ICA"}],
        "sorts": [],
        "facet_type": "document_type",
    },
    {
        "text": "requisitos de habilitação para pilotos",
        "filters": [],
        "sorts": [],
        "facet_type": "general",
    },
])


class TestRewriteBasic:
    def test_returns_rewritten_queries(self, rewriter, mock_llm):
        mock_llm.generate.return_value = VALID_LLM_RESPONSE
        result = rewriter.rewrite("pilotos")

        assert len(result) == 2
        assert isinstance(result[0], RewrittenQuery)
        assert result[0].text == "normas sobre certificação de pilotos"
        assert len(result[0].filters) == 1
        assert result[0].facet_type == "document_type"

    def test_max_queries_limit(self, rewriter, mock_llm):
        many_queries = json.dumps([
            {"text": f"query {i}", "filters": [], "sorts": [], "facet_type": "general"}
            for i in range(10)
        ])
        mock_llm.generate.return_value = many_queries
        result = rewriter.rewrite("test", max_queries=2)

        assert len(result) <= 2


class TestRewriteFallback:
    def test_fallback_on_empty_response(self, rewriter, mock_llm):
        mock_llm.generate.return_value = ""
        result = rewriter.rewrite("query original")

        assert len(result) == 1
        assert result[0].text == "query original"
        assert result[0].facet_type == "original"

    def test_fallback_on_invalid_json(self, rewriter, mock_llm):
        mock_llm.generate.return_value = "not json at all"
        result = rewriter.rewrite("query original")

        assert len(result) == 1
        assert result[0].text == "query original"

    def test_fallback_on_llm_exception(self, rewriter, mock_llm):
        mock_llm.generate.side_effect = Exception("LLM error")
        result = rewriter.rewrite("query original")

        assert len(result) == 1
        assert result[0].text == "query original"


class TestRewriteValidation:
    def test_truncates_long_queries(self, mock_llm):
        rw = QueryRewriter(llm=mock_llm, max_query_length=20, timeout=5)

        long_query = json.dumps([
            {"text": "a" * 100, "filters": [], "sorts": [], "facet_type": "general"}
        ])
        mock_llm.generate.return_value = long_query
        result = rw.rewrite("test")

        assert len(result[0].text) == 20

    def test_strips_empty_queries(self, rewriter, mock_llm):
        response = json.dumps([
            {"text": "", "filters": [], "sorts": [], "facet_type": "general"},
            {"text": "valid query", "filters": [], "sorts": [], "facet_type": "general"},
        ])
        mock_llm.generate.return_value = response
        result = rewriter.rewrite("test")

        assert len(result) == 1
        assert result[0].text == "valid query"


class TestRewriteFilterParsing:
    def test_parses_filters_correctly(self, rewriter, mock_llm):
        response = json.dumps([{
            "text": "regulamentos DECEA",
            "filters": [
                {"field": "metadata.authority", "operator": "eq", "value": "DECEA"},
                {"field": "effective_date", "operator": "gte", "value": "2023-01-01"},
            ],
            "sorts": [{"field": "effective_date", "order": "desc"}],
            "facet_type": "authority",
        }])
        mock_llm.generate.return_value = response
        result = rewriter.rewrite("test")

        assert len(result[0].filters) == 2
        assert result[0].filters[0].field == "metadata.authority"
        assert result[0].filters[0].value == "DECEA"
        assert len(result[0].sorts) == 1
        assert result[0].sorts[0].order.value == "desc"

    def test_handles_markdown_wrapped_json(self, rewriter, mock_llm):
        wrapped = '```json\n' + VALID_LLM_RESPONSE + '\n```'
        mock_llm.generate.return_value = wrapped
        result = rewriter.rewrite("test")

        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
