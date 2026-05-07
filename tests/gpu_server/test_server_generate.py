"""Tests for ``gpu_server.server`` ``/v1/generate`` endpoint.

The GPU server normally lives on a remote host and pre-loads heavy
embedder/cross-encoder models via the FastAPI lifespan hook. We stub those
loaders out so the test process can exercise just the LLM proxy path —
which is what regressed when ``gemma4:*`` (a *thinking model*) was added
to the prod model fleet and the proxy silently dropped ``message.thinking``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


GPU_SERVER_DIR = Path(__file__).resolve().parents[2] / "gpu_server"


@pytest.fixture
def server_module(monkeypatch):
    """Import ``gpu_server.server`` with the heavy model loaders stubbed."""
    if str(GPU_SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(GPU_SERVER_DIR))

    # Wipe any cached import so each test runs against a fresh module.
    for cached in ("server",):
        sys.modules.pop(cached, None)

    server = importlib.import_module("server")

    monkeypatch.setattr(server, "_load_embedding_model", lambda: None)
    monkeypatch.setattr(server, "_load_cross_encoder", lambda: None)
    return server


@pytest.fixture
def client(server_module):
    from fastapi.testclient import TestClient

    with TestClient(server_module.app) as c:
        yield c


def _stub_ollama(server_module, response_payload):
    """Replace ``_get_ollama_client`` with a stub that returns *response_payload*."""
    fake_client = MagicMock()
    fake_client.chat = MagicMock(return_value=response_payload)
    server_module._get_ollama_client = lambda: fake_client
    return fake_client


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

class TestGenerateBackwardCompat:
    """Pre-existing clients must keep seeing the same ``content``/``model``/
    ``elapsed_ms`` fields and the new fields must be optional."""

    def test_legacy_response_shape_still_works(self, server_module, client):
        _stub_ollama(server_module, {
            "message": {"role": "assistant", "content": "hello"},
        })

        resp = client.post("/v1/generate", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "hi"}],
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["content"] == "hello"
        assert body["model"] == "llama3.1:8b"
        assert "elapsed_ms" in body
        # New optional fields default to None when upstream doesn't return them.
        assert body["thinking"] is None
        assert body["done_reason"] is None

    def test_does_not_forward_think_or_keep_alive_when_unset(
        self, server_module, client,
    ):
        fake_client = _stub_ollama(server_module, {
            "message": {"content": "ok"},
        })

        client.post("/v1/generate", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "hi"}],
        })

        kwargs = fake_client.chat.call_args.kwargs
        assert "think" not in kwargs
        assert "keep_alive" not in kwargs
        assert kwargs["stream"] is False


# ---------------------------------------------------------------------------
# New behaviour: ``think``, ``keep_alive``, diagnostic fields, warnings
# ---------------------------------------------------------------------------

class TestGenerateThinkPassthrough:
    def test_think_false_is_forwarded(self, server_module, client):
        fake_client = _stub_ollama(server_module, {
            "message": {"content": "SELECT 1;"},
            "done_reason": "stop",
        })

        client.post("/v1/generate", json={
            "model": "gemma4:26b",
            "messages": [{"role": "user", "content": "select 1"}],
            "think": False,
        })

        assert fake_client.chat.call_args.kwargs["think"] is False

    def test_think_string_effort_is_forwarded(self, server_module, client):
        fake_client = _stub_ollama(server_module, {"message": {"content": "x"}})

        client.post("/v1/generate", json={
            "model": "gemma4:26b",
            "messages": [{"role": "user", "content": "hi"}],
            "think": "low",
        })

        assert fake_client.chat.call_args.kwargs["think"] == "low"

    def test_keep_alive_is_forwarded(self, server_module, client):
        fake_client = _stub_ollama(server_module, {"message": {"content": "x"}})

        client.post("/v1/generate", json={
            "model": "gemma4:26b",
            "messages": [{"role": "user", "content": "hi"}],
            "keep_alive": "30m",
        })

        assert fake_client.chat.call_args.kwargs["keep_alive"] == "30m"


class TestGenerateDiagnostics:
    def test_returns_thinking_and_done_reason_when_present(
        self, server_module, client,
    ):
        _stub_ollama(server_module, {
            "message": {"content": "answer", "thinking": "deliberating"},
            "done_reason": "stop",
            "eval_count": 100,
            "prompt_eval_count": 25,
        })

        body = client.post("/v1/generate", json={
            "model": "gemma4:26b",
            "messages": [{"role": "user", "content": "q"}],
        }).json()

        assert body["content"] == "answer"
        assert body["thinking"] == "deliberating"
        assert body["done_reason"] == "stop"
        assert body["eval_count"] == 100
        assert body["prompt_eval_count"] == 25

    def test_warns_when_thinking_eats_full_budget(
        self, server_module, client, caplog,
    ):
        """The exact failure mode from the user's curl: ``num_predict=512``,
        ``done_reason=length``, empty content. We need a loud warning so this
        never goes unnoticed in prod again."""
        import logging
        from loguru import logger as loguru_logger

        _stub_ollama(server_module, {
            "message": {
                "content": "",
                "thinking": "...big internal chain of thought..." * 50,
            },
            "done_reason": "length",
            "eval_count": 512,
        })

        handler_id = loguru_logger.add(
            lambda m: logging.getLogger("loguru").warning(m), level="WARNING",
        )
        try:
            with caplog.at_level(logging.WARNING, logger="loguru"):
                body = client.post("/v1/generate", json={
                    "model": "gemma4:26b",
                    "messages": [{"role": "user", "content": "q"}],
                    "options": {"num_predict": 512},
                }).json()
        finally:
            loguru_logger.remove(handler_id)

        # Server still answers 200 with the empty content (so we don't break
        # callers that handle empty replies), but it MUST shout about it.
        assert body["content"] == ""
        assert body["done_reason"] == "length"
        assert body["thinking"].startswith("...big")
        assert any(
            "done_reason=length" in rec.message for rec in caplog.records
        ), f"expected truncation warning, got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------

class TestGenerateErrorHandling:
    def test_ollama_error_becomes_502(self, server_module, client):
        fake_client = MagicMock()
        fake_client.chat.side_effect = RuntimeError("connection refused")
        server_module._get_ollama_client = lambda: fake_client

        resp = client.post("/v1/generate", json={
            "model": "gemma4:26b",
            "messages": [{"role": "user", "content": "q"}],
        })

        assert resp.status_code == 502
        assert "connection refused" in resp.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
