"""Tests for QdrantManager upload and indexing control."""

from unittest.mock import MagicMock, patch

import pytest

from database.qdrant_manager import QdrantManager
from search.exceptions import SearchBackendError


def _make_collection_info(dense=True, sparse=True):
    """Build a mock collection info with named vector configs."""
    info = MagicMock()
    vectors = {}
    if dense:
        vectors["dense"] = MagicMock()
    sparse_vectors = {}
    if sparse:
        sparse_vectors["sparse"] = MagicMock()

    info.config.params.vectors = vectors
    info.config.params.sparse_vectors = sparse_vectors
    return info


def _make_manager(dense=True, sparse=True):
    """Create a QdrantManager with a mocked client and collection vectors."""
    with patch("database.qdrant_manager.QdrantClient") as MockClient:
        mock_client = MagicMock()
        mock_client.get_collection.return_value = _make_collection_info(dense, sparse)
        MockClient.return_value = mock_client
        mgr = QdrantManager(host="localhost", port=6333, collection_name="test")
        return mgr


@pytest.fixture
def manager():
    return _make_manager(dense=True, sparse=True)


class TestDisableIndexing:

    def test_sets_threshold_to_zero(self, manager):
        manager.disable_indexing()

        manager.client.update_collection.assert_called_once()
        kwargs = manager.client.update_collection.call_args[1]
        assert kwargs["collection_name"] == "test"
        assert kwargs["optimizer_config"].indexing_threshold == 0

    def test_enable_restores_default_threshold(self, manager):
        manager.enable_indexing()

        kwargs = manager.client.update_collection.call_args[1]
        assert kwargs["optimizer_config"].indexing_threshold == 20_000

    def test_enable_accepts_custom_threshold(self, manager):
        manager.enable_indexing(threshold=50_000)

        kwargs = manager.client.update_collection.call_args[1]
        assert kwargs["optimizer_config"].indexing_threshold == 50_000


class TestWaitForIndexing:

    def test_returns_immediately_when_green(self, manager):
        info = MagicMock()
        info.status.name = "GREEN"
        manager.client.get_collection.return_value = info

        manager.wait_for_indexing(timeout_sec=5)

        manager.client.get_collection.assert_called()

    @patch("time.sleep")
    def test_polls_until_green(self, mock_sleep, manager):
        yellow = MagicMock()
        yellow.status.name = "YELLOW"
        green = MagicMock()
        green.status.name = "GREEN"

        manager.client.get_collection.reset_mock()
        manager.client.get_collection.side_effect = [yellow, yellow, green]

        manager.wait_for_indexing(timeout_sec=300)

        assert manager.client.get_collection.call_count == 3
        assert mock_sleep.call_count == 2


class TestUpsertPoints:

    def test_uses_config_defaults(self, manager):
        points = [{"id": "1", "vector": [0.1, 0.2], "payload": {}}]

        with patch("database.qdrant_manager.config") as mock_config:
            mock_config.INGESTION_BATCH_SIZE = 200
            mock_config.NUM_WORKERS = 8
            manager.upsert_points(points)

        kwargs = manager.client.upload_points.call_args[1]
        assert kwargs["batch_size"] == 200
        assert kwargs["parallel"] == 8

    def test_wait_defaults_to_false(self, manager):
        points = [{"id": "1", "vector": [0.1, 0.2], "payload": {}}]
        manager.upsert_points(points)

        kwargs = manager.client.upload_points.call_args[1]
        assert kwargs["wait"] is False

    def test_wait_true_when_requested(self, manager):
        points = [{"id": "1", "vector": [0.1, 0.2], "payload": {}}]
        manager.upsert_points(points, wait=True)

        kwargs = manager.client.upload_points.call_args[1]
        assert kwargs["wait"] is True

    def test_allows_override_batch_and_parallel(self, manager):
        points = [{"id": "1", "vector": [0.1, 0.2], "payload": {}}]
        manager.upsert_points(points, batch_size=32, parallel=1)

        kwargs = manager.client.upload_points.call_args[1]
        assert kwargs["batch_size"] == 32
        assert kwargs["parallel"] == 1

    def test_returns_true_on_success(self, manager):
        points = [{"id": "1", "vector": [0.1], "payload": {}}]
        assert manager.upsert_points(points) is True

    def test_returns_false_on_error(self, manager):
        manager.client.upload_points.side_effect = RuntimeError("fail")
        points = [{"id": "1", "vector": [0.1], "payload": {}}]
        assert manager.upsert_points(points) is False


class TestSearchErrorHandling:

    def test_raises_search_backend_error_on_failure(self, manager):
        manager.client.query_points.side_effect = RuntimeError("connection lost")

        with pytest.raises(SearchBackendError, match="Qdrant search failed"):
            manager.search(dense_vector=[0.1, 0.2])


class TestCollectionIntrospection:

    def test_detects_dense_and_sparse(self):
        mgr = _make_manager(dense=True, sparse=True)
        assert mgr.has_dense is True
        assert mgr.has_sparse is True

    def test_detects_sparse_only(self):
        mgr = _make_manager(dense=False, sparse=True)
        assert mgr.has_dense is False
        assert mgr.has_sparse is True

    def test_detects_dense_only(self):
        mgr = _make_manager(dense=True, sparse=False)
        assert mgr.has_dense is True
        assert mgr.has_sparse is False

    def test_handles_missing_collection(self):
        with patch("database.qdrant_manager.QdrantClient") as MockClient:
            mock_client = MagicMock()
            mock_client.get_collection.side_effect = Exception("not found")
            MockClient.return_value = mock_client
            mgr = QdrantManager(host="localhost", port=6333, collection_name="nope")

        assert mgr.has_dense is False
        assert mgr.has_sparse is False


class TestSearchVectorFallback:
    """Verify search gracefully falls back when vectors don't match the collection."""

    def test_dense_ignored_on_sparse_only_collection(self):
        mgr = _make_manager(dense=False, sparse=True)
        mock_result = MagicMock()
        mock_result.points = []
        mgr.client.query_points.return_value = mock_result

        mgr.search(dense_vector=[0.1], sparse_vector=MagicMock())

        call_kwargs = mgr.client.query_points.call_args
        assert call_kwargs[1].get("using") == "sparse"

    def test_sparse_ignored_on_dense_only_collection(self):
        mgr = _make_manager(dense=True, sparse=False)
        mock_result = MagicMock()
        mock_result.points = []
        mgr.client.query_points.return_value = mock_result

        mgr.search(dense_vector=[0.1], sparse_vector=MagicMock())

        call_kwargs = mgr.client.query_points.call_args
        assert call_kwargs[1].get("using") == "dense"

    def test_error_when_no_vectors_match(self):
        mgr = _make_manager(dense=False, sparse=False)

        with pytest.raises(SearchBackendError, match="No usable vectors"):
            mgr.search(dense_vector=[0.1])


class TestHybridPrefetchMultiplier:
    """Verify SEARCH_PREFETCH_MULTIPLIER drives the per-branch prefetch size."""

    def _hybrid_call_kwargs(self, multiplier: int, limit: int):
        from config import config as global_config

        original = global_config.SEARCH_PREFETCH_MULTIPLIER
        global_config.SEARCH_PREFETCH_MULTIPLIER = multiplier
        try:
            mgr = _make_manager(dense=True, sparse=True)
            mgr.client.query_points.return_value = MagicMock(points=[])
            mgr.search(
                dense_vector=[0.1, 0.2],
                sparse_vector=MagicMock(),
                limit=limit,
            )
            return mgr.client.query_points.call_args[1]
        finally:
            global_config.SEARCH_PREFETCH_MULTIPLIER = original

    def test_default_multiplier_three(self):
        kwargs = self._hybrid_call_kwargs(multiplier=3, limit=5)
        prefetch = kwargs["prefetch"]
        assert all(p.limit == 15 for p in prefetch), "expected 5*3=15 per branch"

    def test_custom_multiplier_propagates(self):
        kwargs = self._hybrid_call_kwargs(multiplier=8, limit=5)
        prefetch = kwargs["prefetch"]
        assert all(p.limit == 40 for p in prefetch), "expected 5*8=40 per branch"

    def test_final_limit_unchanged_by_multiplier(self):
        """The fused top-K must remain ``limit``, regardless of prefetch."""
        kwargs = self._hybrid_call_kwargs(multiplier=12, limit=5)
        assert kwargs["limit"] == 5
