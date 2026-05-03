# Migration: explicit chat feedback → SQLite `data/app.db`

**Date:** 2026-05-03
**Status:** Implemented on `feat/chat-feedback-sqlite`
**Owner:** web / feedback loop

---

## 1. Summary

Move the explicit user feedback captured in the chat UI from per-session
JSON files to a dedicated SQLite operational database (`data/app.db`),
queryable through the existing Datasette instance alongside the
immutable document catalog (`data/store.db`).

The feedback UX is also overhauled: fast-path thumbs up/down, optional
free-text comment, machine-readable reason code for thumbs-down, and the
existing 5-star grid is preserved as a collapsible "detail" section.

This migration does **not** feed any data back into the RAG pipeline
yet. The goal is to start collecting clean, queryable signals so that
future improvements (ranking/generation tuning, cohorting, abuse
detection) have a stable source of truth to consume.

## 2. Why

### 2.1 Weaknesses of the previous flow

- 5-star-only UI → low completion rate; most users never pick all five
  categories.
- No textual comment, no reason taxonomy → qualitative triage is
  impossible.
- No "undo" for a mistakenly clicked star.
- JSON-only storage (`web/chat_history/*.json`) → aggregations require
  scanning every file and can't be served by Datasette.
- No timestamp on the feedback event itself (only the message
  timestamp), no history of changes.

### 2.2 Goals

- Capture a **fast** primary signal (thumbs) plus **rich** optional
  signals (reason, comment, stars).
- Make feedback **queryable** by SQL in the existing Datasette UI
  (authentication already wired up).
- Keep the door open for future operational data (users, feature flags,
  app settings, chat persistence, audit log) without a second migration.

## 3. Architecture

### 3.1 Data layers

| Layer | File | Mode | Purpose |
|---|---|---|---|
| Catalog | `data/store.db` | Immutable (Datasette `--immutable`) | Document catalog produced by the ingestion pipeline. Rebuilt and atomically swapped. |
| Operational | `data/app.db` | WAL, mutable | All runtime data the web app produces. Today: `feedback_events` + view. |
| Vector | Qdrant | — | Dense/sparse vectors for retrieval. |

### 3.2 Why a single `app.db` (not `feedback.db`)

Alternatives considered:

- **DB-per-domain** (`feedback.db`, `users.db`, …): clean boundaries,
  but cross-domain joins need `ATTACH`, three backups, three files in
  Datasette, fragmented migrations.
- **New table inside `store.db`** + remove `--immutable`: acts as a
  single SQLite, but couples user feedback to the ingestion pipeline
  (scripts/collect/embed/index also write to store.db) — risk of locks
  and confusing concerns.
- **Postgres**: overkill for the current scale; SQLite/WAL + Datasette
  already cover the requirements.

**Chosen:** a single `data/app.db` serving *all* future operational
domains. Table-prefix namespaces (`feedback_*`, `users_*`, `flags_*`,
`chat_*`) keep concerns separate, cross-domain joins are trivial, one
backup, one migration stream.

### 3.3 Datasette wiring

```bash
datasette serve --immutable data/store.db data/app.db --metadata metadata.yml ...
```

`--immutable` applies only to the file immediately following it; the
next positional path (`data/app.db`) is opened in write mode. Both
`deploy/ragexplore.service` and `make explore` were updated.

### 3.4 AppStore

`web/app/app_store.py` exposes `AppStore`:

- Opens the SQLite with `journal_mode=WAL`, `synchronous=NORMAL`,
  `foreign_keys=ON`, `busy_timeout=5000`.
- Runs numbered SQL migrations from `web/app/migrations/*.sql` on
  construction, tracked by a `schema_migrations` table
  (idempotent: re-applying is a no-op).
- Exposes **domain namespaces**: today `store.feedback.insert_event(...)`
  / `store.feedback.get_current(...)`. Future domains land as sibling
  namespaces (`store.users`, `store.flags`, …) without touching
  existing code.

### 3.5 Schema (001_feedback.sql)

- **`feedback_events`** — append-only audit log. One row per user
  action. Columns cover `kind` ∈ {`thumbs`, `star`, `comment`, `clear`}
  with `CHECK` constraints, plus denormalised snapshots of the
  ratee message (`model_used`, `used_rag`, `user_query`,
  `assistant_text`, `sources_json`, `client_ip_hash`).
- **`feedback_current`** — view with the latest state per
  `(session_id, message_id)`. Latest-wins is resolved by
  `MAX(event_id)` (strictly monotonic), not `MAX(created_at)` — this
  avoids ambiguity when several events share an ISO-second timestamp.
- **`uq_fb_backfill_star`** — partial unique index used by the
  backfill script to dedup re-imports of the legacy JSON ratings.

### 3.6 Write path

```
UI (chat.html)  --POST /api/chat/feedback-->  web/main.py
                                              ├── updates chat_history/{sid}.json
                                              └── app_store.feedback.insert_event(...)
                                                  └── data/app.db  (WAL)
Datasette (8001/explore) reads app.db + store.db concurrently.
```

The JSON dual-write is deliberate and cheap: session replay in the UI
keeps reading from `chat_history/*.json` unchanged. When/if chat itself
migrates to SQLite, we drop the JSON write with no UI changes.

## 4. API contract

### 4.1 `POST /api/chat/feedback` (new)

```json
{
  "session_id": "string",
  "message_id": "string",
  "kind": "thumbs" | "star" | "comment" | "clear",
  "thumbs": "up" | "down",                                // kind=thumbs
  "category": "factual_accuracy" | "completeness" | "clarity" | "citation_quality" | "relevance",
  "rating": 0-5,                                          // kind=star (0 = clear)
  "reason_code": "hallucination" | "incomplete" | "off_topic" | "wrong_citation" | "unclear" | "other",
  "comment": "string"                                     // kind=comment
}
```

Validation errors return `400`; unrecognised payload fields return `422`.

### 4.2 `POST /api/chat/rate` (deprecated alias)

Maps 1:1 to `kind='star'`. Kept for one release; clients should migrate.

## 5. UX changes

- **Primary row**: "Essa resposta foi útil?" + 👍 / 👎 / 💬 Comentar /
  ⭐ Detalhes. One click is enough.
- **Thumbs-down** reveals a `<select>` of reasons (same taxonomy as
  `feedback_events.reason_code`).
- **Comentar** opens a textarea for free text.
- **Detalhes** expands the legacy 5-star grid. Clicking an already-filled
  star clears that category (sends `rating=0`).
- Status region uses `role="status"` + `aria-live="polite"`; stars are
  focusable (`tabindex="0"`).

## 6. Migrations and backfill

- First boot auto-runs `001_feedback.sql`. No manual step.
- `make app-init` creates `data/app.db` with migrations applied (no-op
  on subsequent runs).
- `make feedback-backfill` (or
  `python scripts/backfill_feedback_from_json.py`) imports existing
  ratings from `web/chat_history/*.json` as `kind='star'` events.
  Idempotent via `uq_fb_backfill_star`.

## 7. Reserved space for future domains

The same `app.db` absorbs the roadmap below without a second migration
or table rename. Each domain enters as a new numbered migration
(`002_users.sql`, `003_flags.sql`, …).

```sql
-- users
CREATE TABLE users (
  user_id TEXT PRIMARY KEY, email TEXT UNIQUE, display_name TEXT,
  role TEXT DEFAULT 'user', created_at TEXT NOT NULL, last_seen_at TEXT
);

-- dynamic config with scope
CREATE TABLE app_settings (
  scope TEXT NOT NULL DEFAULT 'global',
  key TEXT NOT NULL, value TEXT NOT NULL,
  value_type TEXT NOT NULL DEFAULT 'json',
  updated_at TEXT NOT NULL,
  PRIMARY KEY (scope, key)
);

-- feature flags with scoped rules (global/env/user/cohort/session)
CREATE TABLE feature_flags (
  flag_key TEXT PRIMARY KEY, description TEXT,
  default_value TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE feature_flag_rules (
  rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
  flag_key TEXT NOT NULL REFERENCES feature_flags(flag_key),
  scope_type TEXT NOT NULL CHECK(scope_type IN ('global','env','user','cohort','session')),
  scope_id TEXT, value TEXT NOT NULL,
  priority INTEGER NOT NULL DEFAULT 0, enabled INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);

-- chat persistence (replace JSON session store)
CREATE TABLE chat_sessions (session_id TEXT PRIMARY KEY, user_id TEXT, title TEXT, model TEXT, created_at TEXT, last_updated TEXT);
CREATE TABLE chat_messages (message_id TEXT PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, model TEXT, use_rag INTEGER, sources_json TEXT, processing_time_ms INTEGER, created_at TEXT);

-- generic audit
CREATE TABLE audit_log (event_id INTEGER PRIMARY KEY AUTOINCREMENT, actor TEXT, action TEXT, target TEXT, payload TEXT, created_at TEXT);
```

`feedback_events.user_id` will be added as part of the users migration
(column add + index), keeping the log append-only.

## 8. Privacy

No raw IP is stored. A short hash (`sha256`, first 16 hex chars) of the
client IP is persisted in `client_ip_hash` as a weak abuse/dedup signal.
Comments are user-generated content and MUST be redacted before any
export outside the admin plane.

## 9. Operator runbook

- Create the DB: `make app-init`
- Inspect: `make explore` then open `/explore/app/feedback_current`
- Backfill legacy: `make feedback-backfill`
- Tests: `pytest tests/web -q`

## 10. Rollback

Remove the following and the system reverts to JSON-only:

- `--immutable data/store.db data/app.db` → `--immutable data/store.db`
  in `deploy/ragexplore.service` and `Makefile` target `explore`.
- Restore the legacy `/api/chat/rate` handler that only writes JSON
  (the dual-write is additive; the JSON is still the source for session
  replay).
- The SQLite file can stay on disk; no upstream code depends on it yet.
