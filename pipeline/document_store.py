"""
SQLite-backed document store for the 3-phase ingestion pipeline.

Provides content-hash-based change detection so that scrapers only
persist documents whose content has actually changed.

Usage:
    from pipeline.document_store import DocumentStore

    store = DocumentStore()
    action = store.upsert_document("doc-1", "lexml", "full text …", {...})
    # action is 'inserted', 'updated', or 'unchanged'
"""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

from loguru import logger

from config import config

Action = Literal["inserted", "updated", "unchanged"]

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    urn          TEXT,
    url          TEXT,
    title        TEXT,
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    doc_type     TEXT,
    metadata     TEXT,
    scraped_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_source       ON documents(source);
CREATE INDEX IF NOT EXISTS idx_doc_type     ON documents(doc_type);

CREATE TABLE IF NOT EXISTS embedding_log (
    doc_id         TEXT PRIMARY KEY,
    content_hash   TEXT NOT NULL,
    embedding_mode TEXT NOT NULL,
    model_name     TEXT NOT NULL,
    num_chunks     INTEGER,
    embedded_at    TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentStore:
    """Thin wrapper around an SQLite database that stores scraped documents."""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or config.STORE_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.db_path), timeout=30)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._init_db()
        logger.info(f"DocumentStore ready ({self.db_path})")

    # ── lifecycle ───────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA_SQL)

    def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None

    @contextmanager
    def _conn(self):
        conn = self._connection
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ── content hashing ─────────────────────────────────────────

    @staticmethod
    def compute_content_hash(content: str) -> str:
        normalized = " ".join(content.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # ── CRUD ────────────────────────────────────────────────────

    def upsert_document(
        self,
        doc_id: str,
        source: str,
        content: str,
        metadata: Optional[Dict] = None,
        *,
        urn: str = None,
        url: str = None,
        title: str = None,
        doc_type: str = None,
    ) -> Action:
        content_hash = self.compute_content_hash(content)
        meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        now = _now_iso()

        with self._conn() as conn:
            row = conn.execute(
                "SELECT content_hash FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()

            if row is None:
                conn.execute(
                    """INSERT INTO documents
                       (doc_id, source, urn, url, title, content,
                        content_hash, doc_type, metadata, scraped_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (doc_id, source, urn, url, title, content,
                     content_hash, doc_type, meta_json, now, now),
                )
                return "inserted"

            if row["content_hash"] == content_hash:
                return "unchanged"

            conn.execute(
                """UPDATE documents
                   SET content = ?, content_hash = ?, title = ?,
                       url = ?, urn = ?, doc_type = ?,
                       metadata = ?, updated_at = ?
                   WHERE doc_id = ?""",
                (content, content_hash, title, url, urn,
                 doc_type, meta_json, now, doc_id),
            )
            return "updated"

    def get_document(self, doc_id: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def get_documents_by_source(self, source: str) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE source = ?", (source,)
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_all_documents(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM documents").fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_all_doc_ids(self) -> Set[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT doc_id FROM documents").fetchall()
            return {r["doc_id"] for r in rows}

    def exists(self, doc_id: str) -> bool:
        """Check if a document exists without loading the full row."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE doc_id = ? LIMIT 1", (doc_id,)
            ).fetchone()
            return row is not None

    def delete_by_source(self, source: str) -> int:
        """Delete all documents (and their embedding logs) for a given source.

        Returns the number of deleted documents.
        """
        with self._conn() as conn:
            doc_ids = [
                r["doc_id"]
                for r in conn.execute(
                    "SELECT doc_id FROM documents WHERE source = ?", (source,)
                ).fetchall()
            ]
            if doc_ids:
                placeholders = ",".join("?" * len(doc_ids))
                conn.execute(
                    f"DELETE FROM embedding_log WHERE doc_id IN ({placeholders})",
                    doc_ids,
                )
            n = conn.execute(
                "DELETE FROM documents WHERE source = ?", (source,)
            ).rowcount
            logger.info(f"Deleted {n} documents from source '{source}'")
            return n

    def count(self, source: str = None) -> int:
        with self._conn() as conn:
            if source:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM documents WHERE source = ?",
                    (source,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
            return row["n"]

    # ── embedding tracking ──────────────────────────────────────

    def get_embedded_hashes(self) -> Dict[str, str]:
        """Return {doc_id: content_hash} for all docs that have been embedded."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT doc_id, content_hash FROM embedding_log"
            ).fetchall()
            return {r["doc_id"]: r["content_hash"] for r in rows}

    def get_docs_needing_embedding(self, mode: str = None) -> List[Dict]:
        """Return documents that are new or whose content changed since last embedding."""
        with self._conn() as conn:
            query = """
                SELECT d.*
                FROM documents d
                LEFT JOIN embedding_log e ON d.doc_id = e.doc_id
                WHERE e.doc_id IS NULL
                   OR e.content_hash != d.content_hash
            """
            params: list = []
            if mode:
                query += " OR e.embedding_mode != ?"
                params.append(mode)
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def log_embedding(
        self,
        doc_id: str,
        content_hash: str,
        embedding_mode: str,
        model_name: str,
        num_chunks: int,
    ) -> None:
        now = _now_iso()
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO embedding_log
                   (doc_id, content_hash, embedding_mode, model_name, num_chunks, embedded_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (doc_id, content_hash, embedding_mode, model_name, num_chunks, now),
            )

    def remove_embedding_log(self, doc_ids: List[str]) -> None:
        if not doc_ids:
            return
        with self._conn() as conn:
            placeholders = ",".join("?" * len(doc_ids))
            conn.execute(
                f"DELETE FROM embedding_log WHERE doc_id IN ({placeholders})",
                doc_ids,
            )

    # ── stats ───────────────────────────────────────────────────

    def stats(self) -> Dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            by_source = conn.execute(
                "SELECT source, COUNT(*) AS n FROM documents GROUP BY source"
            ).fetchall()
            embedded = conn.execute(
                "SELECT COUNT(*) AS n FROM embedding_log"
            ).fetchone()["n"]
            return {
                "total_documents": total,
                "by_source": {r["source"]: r["n"] for r in by_source},
                "embedded_documents": embedded,
            }

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict:
        d = dict(row)
        if d.get("metadata"):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d
