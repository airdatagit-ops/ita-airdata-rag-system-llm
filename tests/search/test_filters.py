"""Tests for the Qdrant filter conversion module."""

import pytest
from qdrant_client.models import Filter, FieldCondition, MatchValue

from search.searcher.filters import build_qdrant_filter, sort_documents
from search.shared.schemas import (
    FilterOperator,
    SearchFilter,
    SearchSort,
    SortOrder,
)


class TestBuildQdrantFilter:
    def test_eq_filter(self):
        filters = [SearchFilter(field="status", operator=FilterOperator.EQ, value="active")]
        result = build_qdrant_filter(filters)

        assert result is not None
        assert len(result.must) == 1

    def test_datetime_filter(self):
        filters = [
            SearchFilter(field="effective_date", operator=FilterOperator.GTE, value="2023-01-01"),
        ]
        result = build_qdrant_filter(filters)

        assert result is not None
        assert len(result.must) == 1

    def test_unsupported_field_ignored(self):
        filters = [
            SearchFilter(field="unknown_field", operator=FilterOperator.EQ, value="x"),
        ]
        result = build_qdrant_filter(filters)

        assert result is None

    def test_with_base_filter(self):
        base = Filter(must=[
            FieldCondition(key="status", match=MatchValue(value="active")),
        ])
        filters = [
            SearchFilter(field="metadata.category", operator=FilterOperator.EQ, value="ICA"),
        ]
        result = build_qdrant_filter(filters, base_filter=base)

        assert result is not None
        assert len(result.must) == 2

    def test_empty_filters_no_base(self):
        result = build_qdrant_filter([])
        assert result is None

    def test_range_filter(self):
        filters = [
            SearchFilter(
                field="effective_date",
                operator=FilterOperator.RANGE,
                value={"gte": "2023-01-01", "lte": "2024-01-01"},
            ),
        ]
        result = build_qdrant_filter(filters)
        assert result is not None


class TestSortDocuments:
    def test_sort_desc(self):
        docs = [
            {"regulation_id": "a", "score": 0.5},
            {"regulation_id": "b", "score": 0.9},
            {"regulation_id": "c", "score": 0.7},
        ]
        sorts = [SearchSort(field="score", order=SortOrder.DESC)]
        result = sort_documents(docs, sorts)

        scores = [d["score"] for d in result]
        assert scores == sorted(scores, reverse=True)

    def test_sort_asc(self):
        docs = [
            {"regulation_id": "a", "score": 0.9},
            {"regulation_id": "b", "score": 0.5},
        ]
        sorts = [SearchSort(field="score", order=SortOrder.ASC)]
        result = sort_documents(docs, sorts)

        assert result[0]["score"] == 0.5

    def test_empty_sorts(self):
        docs = [{"regulation_id": "a"}]
        result = sort_documents(docs, [])
        assert result == docs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
