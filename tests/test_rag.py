"""Tests for RAG pipeline."""

import pytest
from search.rag import RAGPipeline


@pytest.fixture
def rag():
    """Create RAG pipeline."""
    return RAGPipeline()


def test_query_response_format(rag):
    """Test RAG query response format."""
    result = rag.query("teste", limit=1)

    assert "answer" in result
    assert "sources" in result
    assert "total_time_ms" in result
    assert isinstance(result["answer"], str)
    assert isinstance(result["sources"], list)


def test_temporal_query(rag):
    """Test temporal query."""
    result = rag.query("teste", date="2023-05-15", limit=1)

    assert "answer" in result
    # Should filter by date


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
