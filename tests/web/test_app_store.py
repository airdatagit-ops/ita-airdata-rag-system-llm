"""Non-trivial behaviour tests for ``web/app/app_store.py``.

Covered:
* Migration runner is idempotent (second construction does not re-apply
  or duplicate the tracking row).
* WAL journal mode is active.
* ``feedback_current`` reflects the latest event per (session, message,
  category) — including clears.
* ``CHECK`` constraints enforce the taxonomy (``kind``, ``star_value``,
  ``reason_code``, ``thumbs``).
* JSON backfill is idempotent thanks to the dedup unique index.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def test_migration_runner_is_idempotent(tmp_path, app_store_module):
    db = tmp_path / "app.db"
    first = app_store_module.AppStore(db)
    first_applied = first.applied_migrations()
    first.close()

    second = app_store_module.AppStore(db)
    try:
        assert second.applied_migrations() == first_applied
        # Only one row per migration name — the tracking table stays tidy.
        rows = second._conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()
        assert rows[0] == len(first_applied)
    finally:
        second.close()


def test_wal_mode_is_active(app_store):
    mode = app_store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_feedback_current_latest_wins_per_category_including_clear(app_store):
    fb = app_store.feedback
    fb.insert_event(session_id="s1", message_id="m1", kind="thumbs", thumbs="up")
    fb.insert_event(
        session_id="s1",
        message_id="m1",
        kind="star",
        star_category="clarity",
        star_value=4,
    )
    # Override same-category rating; view must pick the latest event by event_id
    # (the two rows can share the same ISO-second timestamp).
    fb.insert_event(
        session_id="s1",
        message_id="m1",
        kind="star",
        star_category="clarity",
        star_value=2,
    )
    fb.insert_event(
        session_id="s1",
        message_id="m1",
        kind="star",
        star_category="relevance",
        star_value=5,
    )
    fb.insert_event(session_id="s1", message_id="m1", kind="comment", comment="nice")

    current = fb.get_current("s1", "m1")
    assert current["last_thumbs"] == "up"
    assert current["rating_clarity"] == 2
    assert current["rating_relevance"] == 5
    assert current["last_comment"] == "nice"

    # A ``clear`` event for the same category must wipe it from the view
    # even when it lands on the same ISO-second timestamp as the star.
    fb.insert_event(
        session_id="s1", message_id="m1", kind="clear", star_category="clarity"
    )
    assert fb.get_current("s1", "m1")["rating_clarity"] is None
    # Other categories are unaffected.
    assert fb.get_current("s1", "m1")["rating_relevance"] == 5


def test_thumbs_down_with_reason_code_is_recoverable_from_view(app_store):
    fb = app_store.feedback
    fb.insert_event(
        session_id="s",
        message_id="m",
        kind="thumbs",
        thumbs="down",
        reason_code="hallucination",
    )
    current = fb.get_current("s", "m")
    assert current["last_thumbs"] == "down"
    assert current["last_thumbs_reason"] == "hallucination"


@pytest.mark.parametrize(
    "kwargs",
    [
        # star_value out of range
        dict(
            session_id="s",
            message_id="m",
            kind="star",
            star_category="clarity",
            star_value=9,
        ),
        # invalid thumbs
        dict(session_id="s", message_id="m", kind="thumbs", thumbs="maybe"),
        # invalid reason_code
        dict(
            session_id="s",
            message_id="m",
            kind="thumbs",
            thumbs="down",
            reason_code="bogus",
        ),
    ],
)
def test_check_constraints_reject_invalid_values(app_store, kwargs):
    with pytest.raises(sqlite3.IntegrityError):
        app_store.feedback.insert_event(**kwargs)


def test_invalid_kind_is_rejected_before_sql(app_store):
    with pytest.raises(ValueError):
        app_store.feedback.insert_event(
            session_id="s", message_id="m", kind="not-a-kind"
        )


def test_backfill_is_idempotent(tmp_path, app_store_module, backfill_module):
    # Prepare a synthetic chat_history JSON with a rating.
    history_dir = tmp_path / "chat_history"
    history_dir.mkdir()
    session_id = "sess-xyz"
    history = {
        "session_id": session_id,
        "title": "t",
        "model": "model-a",
        "created_at": "2026-05-01T10:00:00Z",
        "messages": [
            {"role": "user", "content": "q", "timestamp": "2026-05-01T10:00:00Z"},
            {
                "role": "assistant",
                "content": "a",
                "timestamp": "2026-05-01T10:00:01Z",
                "message_id": "msg-1",
                "model": "model-a",
                "use_rag": True,
                "sources": [{"regulation_id": "RBAC 61", "score": 0.9}],
                "ratings": {"clarity": 4, "relevance": 5},
            },
        ],
    }
    (history_dir / f"{session_id}.json").write_text(
        json.dumps(history), encoding="utf-8"
    )

    db = tmp_path / "app.db"

    stats1 = backfill_module.backfill(history_dir, db)
    assert stats1["files"] == 1
    assert stats1["imported"] == 2  # two star categories
    assert stats1["skipped_duplicates"] == 0

    stats2 = backfill_module.backfill(history_dir, db)
    assert stats2["files"] == 1
    assert stats2["imported"] == 0  # nothing new
    assert stats2["skipped_duplicates"] == 2  # both rows blocked by unique index

    # And the view reflects the right latest-wins state.
    store = app_store_module.AppStore(db)
    try:
        current = store.feedback.get_current(session_id, "msg-1")
        assert current["rating_clarity"] == 4
        assert current["rating_relevance"] == 5
    finally:
        store.close()
