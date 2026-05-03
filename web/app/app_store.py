"""Operational SQLite store for the web app.

`AppStore` is the single persistence point for runtime/operational data
that the web service produces (as opposed to the immutable document
catalog in ``data/store.db``).  Today it owns the feedback event log;
the same connection is intended to host future domains (users,
feature flags, app settings, chat persistence, audit log) without
requiring a refactor.

Design choices
--------------
* **WAL journal mode** so the FastAPI write path does not block
  Datasette readers.
* **Plain ``.sql`` migrations** under ``web/app/migrations``, applied
  idempotently through a ``schema_migrations`` bookkeeping table.
  Alembic would be overkill for a single-file SQLite app; numbered
  scripts keep the schema deterministic across dev and prod.
* **Namespaced domain helpers** (``store.feedback.insert_event(...)``)
  so new domains land as sibling namespaces instead of growing a
  god-object.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from loguru import logger

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

FEEDBACK_KINDS = ("thumbs", "star", "comment", "clear")
FEEDBACK_STAR_CATEGORIES = (
    "factual_accuracy",
    "completeness",
    "clarity",
    "citation_quality",
    "relevance",
)
FEEDBACK_THUMBS = ("up", "down")
FEEDBACK_REASON_CODES = (
    "hallucination",
    "incomplete",
    "off_topic",
    "wrong_citation",
    "unclear",
    "other",
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


class FeedbackNamespace:
    """Feedback-domain helpers bound to an :class:`AppStore` connection."""

    def __init__(self, store: "AppStore") -> None:
        self._store = store

    def insert_event(
        self,
        *,
        session_id: str,
        message_id: str,
        kind: str,
        thumbs: Optional[str] = None,
        star_category: Optional[str] = None,
        star_value: Optional[int] = None,
        reason_code: Optional[str] = None,
        comment: Optional[str] = None,
        model_used: Optional[str] = None,
        used_rag: Optional[bool] = None,
        user_query: Optional[str] = None,
        assistant_text: Optional[str] = None,
        sources_json: Optional[str] = None,
        client_ip_hash: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> int:
        """Insert a single feedback event. Returns the new ``event_id``."""

        if kind not in FEEDBACK_KINDS:
            raise ValueError(f"invalid feedback kind: {kind!r}")

        row = (
            session_id,
            message_id,
            kind,
            thumbs,
            star_category,
            star_value,
            reason_code,
            comment,
            model_used,
            int(used_rag) if used_rag is not None else None,
            user_query,
            assistant_text,
            sources_json,
            client_ip_hash,
            created_at or _utcnow_iso(),
        )
        with self._store._write_lock:
            conn = self._store._conn
            cur = conn.execute(
                """
                INSERT INTO feedback_events(
                    session_id, message_id, kind,
                    thumbs, star_category, star_value, reason_code,
                    comment, model_used, used_rag,
                    user_query, assistant_text, sources_json, client_ip_hash,
                    created_at
                ) VALUES (?,?,?, ?,?,?,?, ?,?,?, ?,?,?,?, ?)
                """,
                row,
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_current(self, session_id: str, message_id: str) -> Optional[dict]:
        """Return the consolidated state for ``(session_id, message_id)``.

        Reads the ``feedback_current`` view.  Returns ``None`` when no
        event has been recorded yet.
        """
        conn = self._store._conn
        row = conn.execute(
            """
            SELECT * FROM feedback_current
            WHERE session_id = ? AND message_id = ?
            """,
            (session_id, message_id),
        ).fetchone()
        return dict(row) if row is not None else None


class AppStore:
    """Single entry-point for the web-app operational SQLite database."""

    def __init__(self, db_path: str | Path, *, migrations_dir: Path | None = None) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._migrations_dir = migrations_dir or MIGRATIONS_DIR
        self._write_lock = threading.Lock()

        self._conn = sqlite3.connect(
            str(self._path),
            timeout=5.0,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._configure_pragmas()
        self._apply_migrations()

        self.feedback = FeedbackNamespace(self)

    @property
    def path(self) -> Path:
        return self._path

    def _configure_pragmas(self) -> None:
        cur = self._conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

    def _apply_migrations(self) -> None:
        # ``executescript`` commits any pending transaction before running,
        # which conflicts with an explicit BEGIN/COMMIT wrapper.  Because
        # every migration script uses ``IF NOT EXISTS`` for DDL and
        # ``INSERT OR IGNORE`` for the tracking row, re-running after a
        # crash is safe — the schema converges to the same state.
        conn = self._conn
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name       TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            """
        )
        applied = {
            r["name"]
            for r in conn.execute("SELECT name FROM schema_migrations").fetchall()
        }
        for script in self._discover_migrations():
            if script.name in applied:
                continue
            logger.info(f"Applying migration {script.name}")
            sql = script.read_text(encoding="utf-8")
            with self._write_lock:
                conn.executescript(sql)
                conn.execute(
                    "INSERT OR IGNORE INTO schema_migrations(name, applied_at) VALUES (?, ?)",
                    (script.name, _utcnow_iso()),
                )

    def _discover_migrations(self) -> List[Path]:
        if not self._migrations_dir.exists():
            return []
        return sorted(self._migrations_dir.glob("*.sql"))

    def applied_migrations(self) -> List[str]:
        return [
            r["name"]
            for r in self._conn.execute(
                "SELECT name FROM schema_migrations ORDER BY name"
            ).fetchall()
        ]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # pragma: no cover — defensive on shutdown
            pass

    def __enter__(self) -> "AppStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "AppStore",
    "FeedbackNamespace",
    "FEEDBACK_KINDS",
    "FEEDBACK_STAR_CATEGORIES",
    "FEEDBACK_THUMBS",
    "FEEDBACK_REASON_CODES",
]
