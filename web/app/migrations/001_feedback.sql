-- 001_feedback.sql
-- Append-only feedback event log plus a consolidated view for analytics.
-- Owned by the AppStore feedback namespace (web/app/app_store.py).

CREATE TABLE IF NOT EXISTS feedback_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    message_id      TEXT NOT NULL,
    kind            TEXT NOT NULL CHECK(kind IN ('thumbs','star','comment','clear')),
    thumbs          TEXT CHECK(thumbs IN ('up','down')),
    star_category   TEXT CHECK(star_category IN (
        'factual_accuracy','completeness','clarity','citation_quality','relevance'
    )),
    star_value      INTEGER CHECK(star_value BETWEEN 0 AND 5),
    reason_code     TEXT CHECK(reason_code IN (
        'hallucination','incomplete','off_topic','wrong_citation','unclear','other'
    )),
    comment         TEXT,
    model_used      TEXT,
    used_rag        INTEGER,
    user_query      TEXT,
    assistant_text  TEXT,
    sources_json    TEXT,
    client_ip_hash  TEXT,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fb_session ON feedback_events(session_id);
CREATE INDEX IF NOT EXISTS idx_fb_message ON feedback_events(message_id);
CREATE INDEX IF NOT EXISTS idx_fb_created ON feedback_events(created_at);

-- Dedup guard used by the JSON backfill script: a given (session, message,
-- category, value, created_at) tuple is only imported once per rating row
-- that already existed in the legacy JSON history.
CREATE UNIQUE INDEX IF NOT EXISTS uq_fb_backfill_star
    ON feedback_events(session_id, message_id, kind, star_category, star_value, created_at)
    WHERE kind = 'star';

-- Consolidated "current state" per (session, message).  Latest-wins per
-- domain; resolved by ``event_id`` (strictly monotonic) so that events
-- sharing a ``created_at`` timestamp still get a deterministic order.
DROP VIEW IF EXISTS feedback_current;
CREATE VIEW feedback_current AS
WITH last_event AS (
    SELECT session_id, message_id,
           MAX(event_id)   AS last_event_id,
           MAX(created_at) AS last_event_at
    FROM feedback_events
    GROUP BY session_id, message_id
),
last_thumbs AS (
    SELECT fe.session_id, fe.message_id, fe.thumbs, fe.reason_code
    FROM feedback_events fe
    JOIN (
        SELECT session_id, message_id, MAX(event_id) AS max_eid
        FROM feedback_events
        WHERE kind = 'thumbs'
        GROUP BY session_id, message_id
    ) mx ON mx.session_id = fe.session_id
        AND mx.message_id = fe.message_id
    WHERE fe.kind = 'thumbs' AND fe.event_id = mx.max_eid
),
last_comment AS (
    SELECT fe.session_id, fe.message_id, fe.comment
    FROM feedback_events fe
    JOIN (
        SELECT session_id, message_id, MAX(event_id) AS max_eid
        FROM feedback_events
        WHERE kind = 'comment' AND comment IS NOT NULL AND comment <> ''
        GROUP BY session_id, message_id
    ) mx ON mx.session_id = fe.session_id
        AND mx.message_id = fe.message_id
    WHERE fe.kind = 'comment' AND fe.event_id = mx.max_eid
),
star_by_cat AS (
    SELECT fe.session_id, fe.message_id, fe.star_category, fe.kind, fe.star_value
    FROM feedback_events fe
    JOIN (
        SELECT session_id, message_id, star_category, MAX(event_id) AS max_eid
        FROM feedback_events
        WHERE kind IN ('star','clear') AND star_category IS NOT NULL
        GROUP BY session_id, message_id, star_category
    ) mx ON mx.session_id    = fe.session_id
        AND mx.message_id    = fe.message_id
        AND mx.star_category = fe.star_category
    WHERE fe.event_id = mx.max_eid
)
SELECT
    le.session_id,
    le.message_id,
    lt.thumbs       AS last_thumbs,
    lt.reason_code  AS last_thumbs_reason,
    MAX(CASE WHEN s.star_category='factual_accuracy' AND s.kind='star' THEN s.star_value END) AS rating_factual_accuracy,
    MAX(CASE WHEN s.star_category='completeness'     AND s.kind='star' THEN s.star_value END) AS rating_completeness,
    MAX(CASE WHEN s.star_category='clarity'          AND s.kind='star' THEN s.star_value END) AS rating_clarity,
    MAX(CASE WHEN s.star_category='citation_quality' AND s.kind='star' THEN s.star_value END) AS rating_citation_quality,
    MAX(CASE WHEN s.star_category='relevance'        AND s.kind='star' THEN s.star_value END) AS rating_relevance,
    lc.comment      AS last_comment,
    le.last_event_at
FROM last_event le
LEFT JOIN last_thumbs  lt ON lt.session_id = le.session_id AND lt.message_id = le.message_id
LEFT JOIN last_comment lc ON lc.session_id = le.session_id AND lc.message_id = le.message_id
LEFT JOIN star_by_cat  s  ON  s.session_id = le.session_id AND  s.message_id = le.message_id
GROUP BY le.session_id, le.message_id, lt.thumbs, lt.reason_code, lc.comment, le.last_event_at;
