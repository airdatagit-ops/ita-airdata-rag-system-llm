"""Tests for the SQLite-backed DocumentStore."""

import json
import pytest
from pathlib import Path

from pipeline.document_store import DocumentStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_store.db"
    return DocumentStore(db_path=str(db_path))


class TestComputeContentHash:

    def test_deterministic(self):
        assert DocumentStore.compute_content_hash("abc") == DocumentStore.compute_content_hash("abc")

    def test_normalizes_whitespace(self):
        h1 = DocumentStore.compute_content_hash("  hello   world  ")
        h2 = DocumentStore.compute_content_hash("hello world")
        assert h1 == h2

    def test_case_insensitive(self):
        h1 = DocumentStore.compute_content_hash("Hello World")
        h2 = DocumentStore.compute_content_hash("hello world")
        assert h1 == h2

    def test_different_content_differs(self):
        assert DocumentStore.compute_content_hash("a") != DocumentStore.compute_content_hash("b")


class TestUpsertDocument:

    def test_insert_new(self, store):
        action = store.upsert_document("doc1", "lexml", "some content")
        assert action == "inserted"

    def test_unchanged_on_same_content(self, store):
        store.upsert_document("doc1", "lexml", "content A")
        action = store.upsert_document("doc1", "lexml", "content A")
        assert action == "unchanged"

    def test_updated_on_different_content(self, store):
        store.upsert_document("doc1", "lexml", "content A")
        action = store.upsert_document("doc1", "lexml", "content B")
        assert action == "updated"

    def test_stores_metadata(self, store):
        meta = {"authority": "federal", "subjects": "aviation"}
        store.upsert_document(
            "doc1", "lexml", "content",
            metadata=meta, title="Test Title", urn="urn:lex:br:test",
        )
        doc = store.get_document("doc1")
        assert doc["title"] == "Test Title"
        assert doc["urn"] == "urn:lex:br:test"
        assert doc["metadata"]["authority"] == "federal"

    def test_update_preserves_scraped_at(self, store):
        store.upsert_document("doc1", "lexml", "v1")
        doc_v1 = store.get_document("doc1")
        store.upsert_document("doc1", "lexml", "v2")
        doc_v2 = store.get_document("doc1")
        assert doc_v2["scraped_at"] == doc_v1["scraped_at"]
        assert doc_v2["updated_at"] != doc_v1["scraped_at"]


class TestGetDocument:

    def test_returns_none_for_missing(self, store):
        assert store.get_document("nonexistent") is None

    def test_returns_correct_document(self, store):
        store.upsert_document("doc1", "decea", "hello world", title="Title 1")
        doc = store.get_document("doc1")
        assert doc["doc_id"] == "doc1"
        assert doc["source"] == "decea"
        assert doc["content"] == "hello world"
        assert doc["title"] == "Title 1"


class TestGetDocumentsBySource:

    def test_filters_by_source(self, store):
        store.upsert_document("d1", "lexml", "content 1")
        store.upsert_document("d2", "decea", "content 2")
        store.upsert_document("d3", "lexml", "content 3")

        lexml_docs = store.get_documents_by_source("lexml")
        decea_docs = store.get_documents_by_source("decea")

        assert len(lexml_docs) == 2
        assert len(decea_docs) == 1
        assert decea_docs[0]["doc_id"] == "d2"


class TestGetAllDocIds:

    def test_returns_all_ids(self, store):
        store.upsert_document("a", "lexml", "c1")
        store.upsert_document("b", "decea", "c2")
        ids = store.get_all_doc_ids()
        assert ids == {"a", "b"}


class TestCount:

    def test_total_count(self, store):
        store.upsert_document("a", "lexml", "c1")
        store.upsert_document("b", "decea", "c2")
        assert store.count() == 2

    def test_count_by_source(self, store):
        store.upsert_document("a", "lexml", "c1")
        store.upsert_document("b", "decea", "c2")
        store.upsert_document("c", "lexml", "c3")
        assert store.count("lexml") == 2
        assert store.count("decea") == 1


class TestEmbeddingTracking:

    def test_initially_empty(self, store):
        assert store.get_embedded_hashes() == {}

    def test_log_and_retrieve(self, store):
        store.upsert_document("doc1", "lexml", "content")
        doc = store.get_document("doc1")
        store.log_embedding("doc1", doc["content_hash"], "dense", "model-v1", 5)

        hashes = store.get_embedded_hashes()
        assert "doc1" in hashes
        assert hashes["doc1"] == doc["content_hash"]

    def test_docs_needing_embedding_new(self, store):
        store.upsert_document("doc1", "lexml", "content")
        docs = store.get_docs_needing_embedding()
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "doc1"

    def test_docs_needing_embedding_after_log(self, store):
        store.upsert_document("doc1", "lexml", "content")
        doc = store.get_document("doc1")
        store.log_embedding("doc1", doc["content_hash"], "dense", "model-v1", 3)
        docs = store.get_docs_needing_embedding()
        assert len(docs) == 0

    def test_docs_needing_embedding_after_content_change(self, store):
        store.upsert_document("doc1", "lexml", "content v1")
        doc = store.get_document("doc1")
        store.log_embedding("doc1", doc["content_hash"], "dense", "model-v1", 3)

        store.upsert_document("doc1", "lexml", "content v2")
        docs = store.get_docs_needing_embedding()
        assert len(docs) == 1

    def test_remove_embedding_log(self, store):
        store.upsert_document("doc1", "lexml", "c1")
        store.log_embedding("doc1", "hash1", "dense", "m", 1)
        assert len(store.get_embedded_hashes()) == 1

        store.remove_embedding_log(["doc1"])
        assert len(store.get_embedded_hashes()) == 0


class TestTemporalFields:

    def test_insert_with_temporal_fields(self, store):
        action = store.upsert_document(
            "doc1", "lexml", "content about aviation law",
            effective_date="2023-06-15",
            expiry_date=None,
            status="active",
        )
        assert action == "inserted"
        doc = store.get_document("doc1")
        assert doc["effective_date"] == "2023-06-15"
        assert doc["expiry_date"] is None
        assert doc["status"] == "active"

    def test_insert_revoked_document(self, store):
        store.upsert_document(
            "doc1", "lexml", "revoked law content",
            effective_date="2020-01-01",
            expiry_date="2023-12-31",
            status="revoked",
        )
        doc = store.get_document("doc1")
        assert doc["status"] == "revoked"
        assert doc["expiry_date"] == "2023-12-31"

    def test_default_status_is_active(self, store):
        store.upsert_document("doc1", "lexml", "some content")
        doc = store.get_document("doc1")
        assert doc["status"] == "active"

    def test_update_preserves_temporal_fields(self, store):
        store.upsert_document(
            "doc1", "lexml", "content v1",
            effective_date="2023-01-01",
            status="active",
        )
        store.upsert_document(
            "doc1", "lexml", "content v2",
            effective_date="2024-01-01",
            status="revoked",
            expiry_date="2024-06-01",
        )
        doc = store.get_document("doc1")
        assert doc["content"] == "content v2"
        assert doc["effective_date"] == "2024-01-01"
        assert doc["status"] == "revoked"
        assert doc["expiry_date"] == "2024-06-01"

    def test_temporal_fields_nullable(self, store):
        store.upsert_document("doc1", "lexml", "no dates found")
        doc = store.get_document("doc1")
        assert doc["effective_date"] is None
        assert doc["expiry_date"] is None


class TestCanonicalDedup:

    def test_exists_canonical_false(self, store):
        assert store.exists_canonical("ica_100-12") is False

    def test_exists_canonical_true(self, store):
        store.upsert_document(
            "doc1", "sislaer", "content",
            canonical_id="ica_100-12",
        )
        assert store.exists_canonical("ica_100-12") is True

    def test_exists_canonical_none(self, store):
        assert store.exists_canonical(None) is False
        assert store.exists_canonical("") is False

    def test_new_columns_stored(self, store):
        store.upsert_document(
            "doc1", "sislaer", "content",
            number="100-12", authority="DECEA", canonical_id="ica_100-12",
        )
        doc = store.get_document("doc1")
        assert doc["number"] == "100-12"
        assert doc["authority"] == "DECEA"
        assert doc["canonical_id"] == "ica_100-12"


class TestDocumentRelations:

    def test_upsert_and_get_relation(self, store):
        store.upsert_document("doc1", "sislaer", "c1")
        store.upsert_relation("doc1", "2000", "amends")
        rels = store.get_relations("doc1")
        assert len(rels) == 1
        assert rels[0]["target_ref"] == "2000"
        assert rels[0]["relation_type"] == "amends"

    def test_upsert_relation_idempotent(self, store):
        store.upsert_document("doc1", "sislaer", "c1")
        store.upsert_relation("doc1", "2000", "amends")
        store.upsert_relation("doc1", "2000", "amends")
        rels = store.get_relations("doc1")
        assert len(rels) == 1

    def test_multiple_relations(self, store):
        store.upsert_document("doc1", "sislaer", "c1")
        store.upsert_relation("doc1", "2000", "amends")
        store.upsert_relation("doc1", "3000", "correlates")
        rels = store.get_relations("doc1")
        assert len(rels) == 2

    def test_resolve_relations(self, store):
        store.upsert_document(
            "ica_100-1/2025", "sislaer", "c1",
            source_ref="sislaer:100",
        )
        store.upsert_document(
            "ica_200-1/2025", "sislaer", "c2",
            source_ref="sislaer:200",
        )
        store.upsert_relation("ica_100-1/2025", "200", "amends")

        resolved = store.resolve_relations()
        assert resolved == 1

        rels = store.get_relations("ica_100-1/2025")
        assert rels[0]["target_doc_id"] == "ica_200-1/2025"

    def test_resolve_relations_unresolvable(self, store):
        store.upsert_document(
            "ica_100-1/2025", "sislaer", "c1",
            source_ref="sislaer:100",
        )
        store.upsert_relation("ica_100-1/2025", "999", "correlates")

        resolved = store.resolve_relations()
        assert resolved == 0

        rels = store.get_relations("ica_100-1/2025")
        assert rels[0]["target_doc_id"] is None

    def test_get_relations_as_target(self, store):
        store.upsert_document("sislaer_100", "sislaer", "c1")
        store.upsert_document("sislaer_200", "sislaer", "c2")
        store.upsert_relation("sislaer_100", "200", "amends", target_doc_id="sislaer_200")

        rels = store.get_relations("sislaer_200")
        assert len(rels) == 1
        assert rels[0]["source_doc_id"] == "sislaer_100"

    def test_upsert_relation_preserves_resolved_target(self, store):
        store.upsert_document("doc_a", "sislaer", "c1")
        store.upsert_document("doc_b", "sislaer", "c2")
        store.upsert_relation("doc_a", "500", "amends", target_doc_id="doc_b")

        store.upsert_relation("doc_a", "500", "amends")

        rels = store.get_relations("doc_a")
        assert len(rels) == 1
        assert rels[0]["target_doc_id"] == "doc_b"

    def test_upsert_relation_updates_target(self, store):
        store.upsert_document("doc_a", "sislaer", "c1")
        store.upsert_document("doc_b", "sislaer", "c2")
        store.upsert_document("doc_c", "sislaer", "c3")
        store.upsert_relation("doc_a", "500", "amends", target_doc_id="doc_b")

        store.upsert_relation("doc_a", "500", "amends", target_doc_id="doc_c")

        rels = store.get_relations("doc_a")
        assert len(rels) == 1
        assert rels[0]["target_doc_id"] == "doc_c"

    def test_delete_by_source_cleans_relations(self, store):
        store.upsert_document("doc1", "sislaer", "c1")
        store.upsert_document("doc2", "lexml", "c2")
        store.upsert_relation("doc1", "999", "amends")

        store.delete_by_source("sislaer")

        rels = store.get_relations("doc1")
        assert len(rels) == 0

    def test_delete_by_source_nullifies_target(self, store):
        store.upsert_document("doc_src", "lexml", "c1")
        store.upsert_document("doc_tgt", "sislaer", "c2")
        store.upsert_relation("doc_src", "ref", "correlates", target_doc_id="doc_tgt")

        store.delete_by_source("sislaer")

        rels = store.get_relations("doc_src")
        assert len(rels) == 1
        assert rels[0]["target_doc_id"] is None


class TestStats:

    def test_stats_structure(self, store):
        store.upsert_document("d1", "lexml", "c1")
        store.upsert_document("d2", "decea", "c2")
        store.log_embedding("d1", "h", "dense", "m", 2)

        s = store.stats()
        assert s["total_processed"] == 2
        assert s["by_source"]["lexml"] == 1
        assert s["by_source"]["decea"] == 1
        assert s["embedded_documents"] == 1
