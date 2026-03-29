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
from typing import Dict, List, Literal, Optional, Set

from loguru import logger

from config import config

Action = Literal["inserted", "updated", "unchanged"]

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id         TEXT PRIMARY KEY,
    source         TEXT NOT NULL,
    urn            TEXT,
    url            TEXT,
    title          TEXT,
    content        TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    doc_type       TEXT,
    metadata       TEXT,
    effective_date TEXT,
    expiry_date    TEXT,
    status         TEXT DEFAULT 'active',
    number         TEXT,
    authority      TEXT,
    canonical_id   TEXT,
    version_year   TEXT,
    is_latest      INTEGER DEFAULT 1,
    scraped_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_source       ON documents(source);
CREATE INDEX IF NOT EXISTS idx_doc_type     ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_status       ON documents(status);
CREATE INDEX IF NOT EXISTS idx_number       ON documents(number);
CREATE INDEX IF NOT EXISTS idx_authority    ON documents(authority);
CREATE INDEX IF NOT EXISTS idx_canonical    ON documents(canonical_id);

CREATE TABLE IF NOT EXISTS document_relations (
    source_doc_id  TEXT NOT NULL,
    target_doc_id  TEXT,
    target_ref     TEXT NOT NULL,
    relation_type  TEXT NOT NULL CHECK(relation_type IN (
        'amends', 'amended_by', 'correlates', 'revokes', 'revoked_by'
    )),
    UNIQUE(source_doc_id, target_ref, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON document_relations(source_doc_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON document_relations(target_doc_id);

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
            self._migrate(conn)

    @staticmethod
    def _migrate(conn) -> None:
        """Add columns introduced after the initial schema."""
        existing = {r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()}
        for col, typedef in [
            ("version_year", "TEXT"),
            ("is_latest", "INTEGER DEFAULT 1"),
        ]:
            if col not in existing:
                conn.execute(f"ALTER TABLE documents ADD COLUMN {col} {typedef}")
                logger.info(f"Migrated: added column '{col}' to documents")

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
        effective_date: str = None,
        expiry_date: str = None,
        status: str = "active",
        number: str = None,
        authority: str = None,
        canonical_id: str = None,
        version_year: str = None,
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
                        content_hash, doc_type, metadata,
                        effective_date, expiry_date, status,
                        number, authority, canonical_id,
                        version_year, scraped_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (doc_id, source, urn, url, title, content,
                     content_hash, doc_type, meta_json,
                     effective_date, expiry_date, status,
                     number, authority, canonical_id,
                     version_year, now, now),
                )
                return "inserted"

            if row["content_hash"] == content_hash:
                return "unchanged"

            conn.execute(
                """UPDATE documents
                   SET content = ?, content_hash = ?, title = ?,
                       url = ?, urn = ?, doc_type = ?,
                       metadata = ?, effective_date = ?,
                       expiry_date = ?, status = ?,
                       number = ?, authority = ?, canonical_id = ?,
                       version_year = ?, updated_at = ?
                   WHERE doc_id = ?""",
                (content, content_hash, title, url, urn,
                 doc_type, meta_json, effective_date,
                 expiry_date, status,
                 number, authority, canonical_id,
                 version_year, now, doc_id),
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

    # ── canonical dedup ────────────────────────────────────────

    def exists_canonical(self, canonical_id: str) -> bool:
        """Check if any document with this canonical_id already exists."""
        if not canonical_id:
            return False
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE canonical_id = ? LIMIT 1",
                (canonical_id,),
            ).fetchone()
            return row is not None

    # ── document relations ───────────────────────────────────

    def upsert_relation(
        self,
        source_doc_id: str,
        target_ref: str,
        relation_type: str,
        target_doc_id: str = None,
    ) -> None:
        """Insert or update a relationship between documents."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO document_relations
                   (source_doc_id, target_doc_id, target_ref, relation_type)
                   VALUES (?, ?, ?, ?)""",
                (source_doc_id, target_doc_id, target_ref, relation_type),
            )

    def get_relations(self, doc_id: str) -> List[Dict]:
        """Get all relations where doc_id is source or target."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM document_relations
                   WHERE source_doc_id = ? OR target_doc_id = ?""",
                (doc_id, doc_id),
            ).fetchall()
            return [dict(r) for r in rows]

    def resolve_relations(self) -> int:
        """Resolve target_ref -> target_doc_id for SISLAER relations.

        Matches target_ref (a codigoRegistro) against documents whose
        metadata contains a matching ``source_ref``. Returns the number
        of resolved relations.
        """
        with self._conn() as conn:
            # Try matching via metadata JSON source_ref (new canonical IDs)
            result = conn.execute(
                """UPDATE document_relations
                   SET target_doc_id = (
                       SELECT doc_id FROM documents
                       WHERE json_extract(metadata, '$.source_ref') =
                             'sislaer:' || document_relations.target_ref
                       LIMIT 1
                   )
                   WHERE target_doc_id IS NULL
                     AND EXISTS (
                       SELECT 1 FROM documents
                       WHERE json_extract(metadata, '$.source_ref') =
                             'sislaer:' || document_relations.target_ref
                     )""",
            )
            resolved = result.rowcount
            if resolved:
                logger.info(f"Resolved {resolved} document relations")
            return resolved

    # ── version management ─────────────────────────────────────

    def compute_latest_versions(self) -> int:
        """Mark the latest active version per canonical_id group.

        Uses ``version_year DESC`` to pick the most recent active version.
        Documents without a ``canonical_id`` are always treated as latest.
        Returns the number of documents marked as latest.
        """
        with self._conn() as conn:
            conn.execute("UPDATE documents SET is_latest = 0")
            conn.execute("""
                UPDATE documents SET is_latest = 1
                WHERE doc_id IN (
                    SELECT doc_id FROM (
                        SELECT doc_id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY canonical_id
                                   ORDER BY version_year DESC NULLS LAST
                               ) AS rn
                        FROM documents
                        WHERE canonical_id IS NOT NULL
                    ) WHERE rn = 1
                )
            """)
            conn.execute(
                "UPDATE documents SET is_latest = 1 WHERE canonical_id IS NULL"
            )
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE is_latest = 1"
            ).fetchone()
            n = row["n"]
            logger.info(f"Computed latest versions: {n} documents marked as is_latest")
            return n

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
