"""Tests for shared pipeline schemas."""

import pytest
from pydantic import ValidationError

from search.shared.schemas import (
    EvaluatedDocument,
    FilterOperator,
    PipelineTrace,
    RewrittenQuery,
    SearchFilter,
    SearchResults,
    SearchSort,
    SortOrder,
    StageTimings,
)


class TestSearchFilter:
    def test_basic_filter(self):
        f = SearchFilter(field="status", operator=FilterOperator.EQ, value="active")
        assert f.field == "status"
        assert f.operator == FilterOperator.EQ
        assert f.value == "active"

    def test_range_filter(self):
        f = SearchFilter(
            field="effective_date",
            operator=FilterOperator.RANGE,
            value={"gte": "2023-01-01", "lte": "2024-01-01"},
        )
        assert isinstance(f.value, dict)

    def test_in_filter(self):
        f = SearchFilter(
            field="metadata.category",
            operator=FilterOperator.IN,
            value=["ICA", "DCA"],
        )
        assert isinstance(f.value, list)


class TestRewrittenQuery:
    def test_basic_query(self):
        q = RewrittenQuery(text="test query")
        assert q.text == "test query"
        assert q.filters == []
        assert q.sorts == []
        assert q.facet_type == ""

    def test_query_with_filters_and_sorts(self):
        q = RewrittenQuery(
            text="test",
            filters=[SearchFilter(field="status", value="active")],
            sorts=[SearchSort(field="effective_date", order=SortOrder.DESC)],
            facet_type="temporal",
        )
        assert len(q.filters) == 1
        assert len(q.sorts) == 1

    def test_max_length_validation(self):
        with pytest.raises(ValidationError):
            RewrittenQuery(text="x" * 1001)


class TestEvaluatedDocument:
    def test_score_bounds(self):
        ed = EvaluatedDocument(
            document={"text": "t", "regulation_id": "d1"},
            relevance_score=50.0,
            query_text="q",
        )
        assert ed.relevance_score == 50.0

    def test_score_out_of_bounds(self):
        with pytest.raises(ValidationError):
            EvaluatedDocument(
                document={}, relevance_score=150.0, query_text="q",
            )


class TestPipelineTrace:
    def test_defaults(self):
        trace = PipelineTrace(original_query="test")
        assert trace.original_query == "test"
        assert trace.rewritten_queries == []
        assert trace.errors == []
        assert trace.timings.total_ms == 0

    def test_to_dict(self):
        trace = PipelineTrace(original_query="test")
        d = trace.to_dict()
        assert isinstance(d, dict)
        assert d["original_query"] == "test"
        assert "timings" in d

    def test_serializable(self):
        import json
        trace = PipelineTrace(
            original_query="test",
            rewritten_queries=[RewrittenQuery(text="rewritten")],
            timings=StageTimings(rewriter_ms=10, total_ms=100),
        )
        json_str = json.dumps(trace.to_dict())
        assert "rewritten" in json_str


class TestSearchResults:
    def test_defaults(self):
        sr = SearchResults()
        assert sr.documents == []
        assert sr.total_before_dedup == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
