"""Tests for the Parquet-backed EmbeddingStore."""

import pytest
import pyarrow as pa

from pipeline.embedding_store import EmbeddingStore


@pytest.fixture
def emb_store(tmp_path):
    return EmbeddingStore(base_dir=str(tmp_path / "embeddings"))


def _make_dense_table(n: int = 3, doc_id: str = "doc1", dim: int = 4) -> pa.Table:
    return pa.table({
        "chunk_id": [f"c{i}" for i in range(n)],
        "doc_id": [doc_id] * n,
        "chunk_index": list(range(n)),
        "text": [f"chunk text {i}" for i in range(n)],
        "dense_vector": [[float(j) for j in range(dim)] for _ in range(n)],
        "content_hash": ["hash1"] * n,
        "metadata": ['{"key": "val"}'] * n,
    })


def _make_sparse_table(n: int = 3, doc_id: str = "doc1") -> pa.Table:
    return pa.table({
        "chunk_id": [f"c{i}" for i in range(n)],
        "doc_id": [doc_id] * n,
        "chunk_index": list(range(n)),
        "text": [f"chunk text {i}" for i in range(n)],
        "sparse_indices": [[0, 1, 2]] * n,
        "sparse_values": [[0.1, 0.2, 0.3]] * n,
        "content_hash": ["hash1"] * n,
        "metadata": ['{"key": "val"}'] * n,
    })


class TestSaveAndLoadDense:

    def test_roundtrip(self, emb_store):
        table = _make_dense_table()
        emb_store.save_dense(table, "lexml")

        loaded = emb_store.load_dense("lexml")
        assert loaded is not None
        assert loaded.num_rows == 3
        assert loaded.column("chunk_id").to_pylist() == ["c0", "c1", "c2"]

    def test_load_all_sources(self, emb_store):
        emb_store.save_dense(_make_dense_table(2, "d1"), "lexml")
        emb_store.save_dense(_make_dense_table(3, "d2"), "decea")

        all_table = emb_store.load_dense()
        assert all_table is not None
        assert all_table.num_rows == 5

    def test_load_nonexistent_returns_none(self, emb_store):
        assert emb_store.load_dense("nonexistent") is None


class TestSaveAndLoadSparse:

    def test_roundtrip(self, emb_store):
        table = _make_sparse_table()
        emb_store.save_sparse(table, "lexml")

        loaded = emb_store.load_sparse("lexml")
        assert loaded is not None
        assert loaded.num_rows == 3


class TestRemoveByDocIds:

    def test_removes_correct_rows(self, emb_store):
        t1 = _make_dense_table(2, "keep")
        t2 = _make_dense_table(3, "remove")
        merged = pa.concat_tables([t1, t2])
        emb_store.save_dense(merged, "lexml")

        emb_store.remove_by_doc_ids({"remove"}, "lexml", kind="dense")

        loaded = emb_store.load_dense("lexml")
        assert loaded.num_rows == 2
        assert set(loaded.column("doc_id").to_pylist()) == {"keep"}

    def test_noop_on_missing_file(self, emb_store):
        emb_store.remove_by_doc_ids({"x"}, "nonexistent", kind="dense")


class TestAppendToSource:

    def test_appends_to_existing(self, emb_store):
        t1 = _make_dense_table(2, "doc1")
        emb_store.save_dense(t1, "lexml")

        t2 = _make_dense_table(3, "doc2")
        emb_store.append_to_source(t2, "lexml", kind="dense")

        loaded = emb_store.load_dense("lexml")
        assert loaded.num_rows == 5

    def test_creates_if_not_exists(self, emb_store):
        t = _make_dense_table(2, "doc1")
        emb_store.append_to_source(t, "new_source", kind="dense")

        loaded = emb_store.load_dense("new_source")
        assert loaded is not None
        assert loaded.num_rows == 2


class TestGetEmbeddedDocIds:

    def test_returns_unique_doc_ids(self, emb_store):
        t = _make_dense_table(5, "mydoc")
        emb_store.save_dense(t, "lexml")

        ids = emb_store.get_embedded_doc_ids("dense")
        assert ids == {"mydoc"}

    def test_empty_when_no_data(self, emb_store):
        assert emb_store.get_embedded_doc_ids("dense") == set()


class TestStats:

    def test_stats_structure(self, emb_store):
        emb_store.save_dense(_make_dense_table(), "lexml")
        emb_store.save_sparse(_make_sparse_table(), "lexml")

        s = emb_store.stats()
        assert "dense" in s
        assert "sparse" in s
        assert s["dense"]["lexml"]["rows"] == 3
        assert s["sparse"]["lexml"]["rows"] == 3
        assert s["dense"]["lexml"]["size_mb"] >= 0
