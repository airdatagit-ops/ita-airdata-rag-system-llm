"""Tests for QdrantManager upload and indexing control."""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from database.qdrant_manager import QdrantManager


@pytest.fixture
def manager():
    with patch("database.qdrant_manager.QdrantClient") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mgr = QdrantManager(host="localhost", port=6333, collection_name="test")
        yield mgr


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

        manager.client.get_collection.assert_called_once_with("test")

    @patch("time.sleep")
    def test_polls_until_green(self, mock_sleep, manager):
        yellow = MagicMock()
        yellow.status.name = "YELLOW"
        green = MagicMock()
        green.status.name = "GREEN"

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
