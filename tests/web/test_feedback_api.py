"""End-to-end test for the feedback HTTP endpoints.

Exercises ``web/main.py`` via FastAPI's ``TestClient`` with a per-test
temporary ``app.db`` and ``chat_history/`` directory.  We inject a stub
``app.config`` module so that importing ``web/main.py`` does not try to
load Pydantic settings from the project's root ``.env`` (which contains
keys the web Settings class intentionally does not allow).
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = PROJECT_ROOT / "web"


def _build_stub_settings(tmp_path: Path) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        ENVIRONMENT="test",
        HOST="127.0.0.1",
        PORT=8082,
        RELOAD=False,
        ROOT_PATH="",
        API_BASE_URL="http://127.0.0.1:8083",
        API_KEY="test-key",
        APP_NAME="Aviation RAG Web (test)",
        APP_VERSION="test",
        APP_DB_PATH=str(tmp_path / "app.db"),
    )


@pytest.fixture()
def web_client(tmp_path, monkeypatch):
    # Stub the ``app`` package so that ``from app.config import settings``
    # resolves to our test settings instead of loading the production
    # Pydantic BaseSettings (which fails on the repo's root .env).
    stub_settings = _build_stub_settings(tmp_path)
    app_pkg = types.ModuleType("app")
    app_config = types.ModuleType("app.config")
    app_config.settings = stub_settings
    app_pkg.config = app_config

    # Load app_store directly so ``web/main.py`` can ``from app.app_store import AppStore``.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "app.app_store", WEB_DIR / "app" / "app_store.py"
    )
    app_store_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_store_mod)
    app_pkg.app_store = app_store_mod

    saved = {
        name: sys.modules.get(name) for name in ("app", "app.config", "app.app_store")
    }
    sys.modules["app"] = app_pkg
    sys.modules["app.config"] = app_config
    sys.modules["app.app_store"] = app_store_mod

    # Isolate chat_history under tmp_path via cwd switch (web/main.py uses
    # the relative path "chat_history").
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chat_history").mkdir()
    # Static dir / templates dir are referenced at import time; point them
    # at the real web/ paths so FastAPI mounts succeed.
    monkeypatch.setenv("TEMPLATES_DIR", str(WEB_DIR / "templates"))

    # Load web/main.py after stubs are in place.
    main_spec = importlib.util.spec_from_file_location(
        "web_main_under_test", WEB_DIR / "main.py"
    )
    main_mod = importlib.util.module_from_spec(main_spec)
    # main.py uses ``StaticFiles(directory="static")`` and
    # ``Jinja2Templates(directory="templates")`` with paths relative to cwd.
    # Create symlinks so those relative lookups succeed under tmp_path.
    (tmp_path / "static").symlink_to(WEB_DIR / "static")
    (tmp_path / "templates").symlink_to(WEB_DIR / "templates")
    main_spec.loader.exec_module(main_mod)

    client = TestClient(main_mod.app)
    # TestClient with ``with`` triggers startup events which instantiates AppStore.
    with client:
        yield client, main_mod, tmp_path

    # Restore module table
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def _write_session(tmp_path: Path, session_id: str, message_id: str) -> None:
    history = {
        "session_id": session_id,
        "title": "t",
        "model": "model-a",
        "created_at": "2026-05-01T10:00:00Z",
        "messages": [
            {"role": "user", "content": "Qual RBAC sobre medicamentos?", "timestamp": "2026-05-01T10:00:00Z"},
            {
                "role": "assistant",
                "content": "Resposta do assistente.",
                "timestamp": "2026-05-01T10:00:01Z",
                "message_id": message_id,
                "model": "model-a",
                "use_rag": True,
                "sources": [{"regulation_id": "RBAC 67", "score": 0.9}],
                "ratings": {},
            },
        ],
    }
    path = tmp_path / "chat_history" / f"{session_id}.json"
    path.write_text(json.dumps(history), encoding="utf-8")


def test_thumbs_down_writes_event_and_updates_json(web_client):
    client, main_mod, tmp_path = web_client
    sid, mid = "s-01", "msg-01"
    _write_session(tmp_path, sid, mid)

    resp = client.post(
        "/api/chat/feedback",
        json={
            "session_id": sid,
            "message_id": mid,
            "kind": "thumbs",
            "thumbs": "down",
            "reason_code": "hallucination",
        },
    )
    assert resp.status_code == 200, resp.text

    # JSON mirror updated
    data = json.loads((tmp_path / "chat_history" / f"{sid}.json").read_text())
    assistant = next(m for m in data["messages"] if m.get("message_id") == mid)
    assert assistant["feedback"] == {"thumbs": "down", "reason_code": "hallucination"}

    # Event persisted
    current = main_mod.app_store.feedback.get_current(sid, mid)
    assert current["last_thumbs"] == "down"
    assert current["last_thumbs_reason"] == "hallucination"


def test_star_rating_zero_clears_category(web_client):
    client, main_mod, tmp_path = web_client
    sid, mid = "s-02", "msg-02"
    _write_session(tmp_path, sid, mid)

    # Set a rating
    r1 = client.post(
        "/api/chat/feedback",
        json={
            "session_id": sid,
            "message_id": mid,
            "kind": "star",
            "category": "clarity",
            "rating": 4,
        },
    )
    assert r1.status_code == 200

    # Then clear with rating=0
    r2 = client.post(
        "/api/chat/feedback",
        json={
            "session_id": sid,
            "message_id": mid,
            "kind": "star",
            "category": "clarity",
            "rating": 0,
        },
    )
    assert r2.status_code == 200

    data = json.loads((tmp_path / "chat_history" / f"{sid}.json").read_text())
    assistant = next(m for m in data["messages"] if m.get("message_id") == mid)
    assert "clarity" not in assistant.get("ratings", {})

    current = main_mod.app_store.feedback.get_current(sid, mid)
    assert current["rating_clarity"] is None


def test_legacy_rate_alias_maps_to_star_and_denormalises(web_client):
    client, main_mod, tmp_path = web_client
    sid, mid = "s-03", "msg-03"
    _write_session(tmp_path, sid, mid)

    resp = client.post(
        "/api/chat/rate",
        json={
            "session_id": sid,
            "message_id": mid,
            "category": "relevance",
            "rating": 5,
        },
    )
    assert resp.status_code == 200

    rows = main_mod.app_store._conn.execute(
        "SELECT kind, star_category, star_value, model_used, user_query, assistant_text "
        "FROM feedback_events WHERE session_id=? AND message_id=?",
        (sid, mid),
    ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert (row["kind"], row["star_category"], row["star_value"]) == (
        "star",
        "relevance",
        5,
    )
    # Denormalised snapshots should be populated from the JSON history.
    assert row["model_used"] == "model-a"
    assert row["user_query"] == "Qual RBAC sobre medicamentos?"
    assert "Resposta do assistente" in (row["assistant_text"] or "")


@pytest.mark.parametrize(
    "payload,expected_status",
    [
        # rating out of range for kind=star
        (
            {
                "session_id": "s",
                "message_id": "m",
                "kind": "star",
                "category": "clarity",
                "rating": 7,
            },
            422,
        ),
        # invalid thumbs value
        (
            {
                "session_id": "s",
                "message_id": "m",
                "kind": "thumbs",
                "thumbs": "maybe",
            },
            400,
        ),
        # invalid kind
        (
            {"session_id": "s", "message_id": "m", "kind": "bogus"},
            400,
        ),
    ],
)
def test_invalid_payload_is_rejected(web_client, payload, expected_status):
    client, _, _ = web_client
    resp = client.post("/api/chat/feedback", json=payload)
    assert resp.status_code == expected_status, resp.text
