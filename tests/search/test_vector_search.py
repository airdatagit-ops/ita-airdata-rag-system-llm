"""Tests for VectorSearch with dependency injection and parallel encoding."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from search.vector_search import VectorSearch
from search.cache import InMemoryCache
from search.exceptions import SearchBackendError


@pytest.fixture
def mock_dense():
    model = MagicMock()
    model.encode.return_value = np.random.rand(1024)
    return model


@pytest.fixture
def mock_sparse():
    model = MagicMock()
    model.encode_single.return_value = MagicMock()
    return model


@pytest.fixture
def mock_db():
    return MagicMock()


class TestDependencyInjection:
    def test_uses_injected_instances(self, mock_dense, mock_sparse, mock_db):
        vs = VectorSearch(dense_model=mock_dense, sparse_model=mock_sparse, db=mock_db)

        assert vs.dense_model is mock_dense
        assert vs.sparse_model is mock_sparse
        assert vs.db is mock_db

    def test_none_disables_component(self, mock_db):
        vs = VectorSearch(dense_model=None, sparse_model=None, db=mock_db)

        assert vs.dense_model is None
        assert vs.sparse_model is None

    @patch("search.vector_search.config")
    def test_sentinel_creates_default_when_enabled(self, mock_config, mock_db):
        mock_config.SEARCH_DENSE_ENABLED = False
        mock_config.SEARCH_SPARSE_ENABLED = False

        vs = VectorSearch(db=mock_db)

        assert vs.dense_model is None
        assert vs.sparse_model is None


class TestEncodeQuery:
    def test_dense_only(self, mock_dense, mock_db):
        vs = VectorSearch(dense_model=mock_dense, sparse_model=None, db=mock_db)

        dense, sparse = vs._encode_query("test query")

        mock_dense.encode.assert_called_once_with("test query")
        assert dense is not None
        assert sparse is None

    def test_sparse_only(self, mock_sparse, mock_db):
        vs = VectorSearch(dense_model=None, sparse_model=mock_sparse, db=mock_db)

        dense, sparse = vs._encode_query("test query")

        mock_sparse.encode_single.assert_called_once_with("test query")
        assert dense is None
        assert sparse is not None

    def test_parallel_encoding(self, mock_dense, mock_sparse, mock_db):
        vs = VectorSearch(dense_model=mock_dense, sparse_model=mock_sparse, db=mock_db)

        dense, sparse = vs._encode_query("test query")

        mock_dense.encode.assert_called_once()
        mock_sparse.encode_single.assert_called_once()
        assert dense is not None
        assert sparse is not None


class TestEmbeddingCache:
    def test_cache_miss_then_hit(self, mock_dense, mock_db):
        cache = InMemoryCache()
        vs = VectorSearch(
            dense_model=mock_dense, sparse_model=None, db=mock_db,
            embedding_cache=cache,
        )

        vs._encode_query("same query")
        vs._encode_query("same query")

        assert mock_dense.encode.call_count == 1
        assert cache.stats()["hits"] == 1
        assert cache.stats()["misses"] == 1

    def test_different_queries_miss(self, mock_dense, mock_db):
        cache = InMemoryCache()
        vs = VectorSearch(
            dense_model=mock_dense, sparse_model=None, db=mock_db,
            embedding_cache=cache,
        )

        vs._encode_query("query a")
        vs._encode_query("query b")

        assert mock_dense.encode.call_count == 2
        assert cache.stats()["misses"] == 2

    def test_no_cache_always_encodes(self, mock_dense, mock_db):
        vs = VectorSearch(
            dense_model=mock_dense, sparse_model=None, db=mock_db,
        )

        vs._encode_query("q")
        vs._encode_query("q")

        assert mock_dense.encode.call_count == 2


class TestSearch:
    def test_delegates_to_db_search(self, mock_dense, mock_db):
        mock_point = MagicMock()
        mock_point.payload = {"regulation_id": "doc-1", "text": "t", "metadata": {}}
        mock_point.score = 0.9
        mock_db.search.return_value = [mock_point]

        vs = VectorSearch(dense_model=mock_dense, sparse_model=None, db=mock_db)
        results = vs.search("query", limit=3)

        mock_db.search.assert_called_once()
        assert len(results) == 1
        assert results[0]["regulation_id"] == "doc-1"

    def test_propagates_search_backend_error(self, mock_dense, mock_db):
        mock_db.search.side_effect = SearchBackendError("fail")

        vs = VectorSearch(dense_model=mock_dense, sparse_model=None, db=mock_db)

        with pytest.raises(SearchBackendError):
            vs.search("query")


class TestSearchTemporal:
    def test_delegates_to_db_search_temporal(self, mock_dense, mock_db):
        mock_point = MagicMock()
        mock_point.payload = {
            "regulation_id": "doc-1",
            "version": "v1",
            "text": "t",
            "effective_date": "2023-01-01",
            "expiry_date": None,
            "metadata": {},
        }
        mock_point.score = 0.85
        mock_db.search_temporal.return_value = [mock_point]

        vs = VectorSearch(dense_model=mock_dense, sparse_model=None, db=mock_db)
        results = vs.search_temporal("query", date="2023-06-01", limit=5)

        mock_db.search_temporal.assert_called_once()
        assert len(results) == 1
        assert results[0]["effective_date"] == "2023-01-01"
