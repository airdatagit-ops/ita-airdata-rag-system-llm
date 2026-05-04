"""Tests for GPU client: factory functions, remote model classes, and lazy loading."""

from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

from models.gpu_client import (
    RemoteDocumentEvaluator,
    RemoteEmbeddingModel,
    RemoteLlamaModel,
    _OllamaListProxy,
    create_embedding_model,
    create_evaluator,
    create_llm,
)


# ---------------------------------------------------------------------------
# Factory functions — routing based on INFERENCE_MODE
# ---------------------------------------------------------------------------

class TestCreateEmbeddingModel:
    @patch("models.gpu_client.config")
    def test_remote_mode_returns_remote(self, mock_cfg):
        mock_cfg.INFERENCE_MODE = "remote"
        mock_cfg.EMBEDDING_MODEL = "test-model"
        mock_cfg.EMBEDDING_DIMENSION = 768
        mock_cfg.GPU_SERVER_URL = "http://gpu:8090"
        mock_cfg.GPU_SERVER_API_KEY = ""
        mock_cfg.GPU_SERVER_TIMEOUT = 120

        model = create_embedding_model()

        assert isinstance(model, RemoteEmbeddingModel)
        assert model.device == "remote"

    @patch("models.gpu_client.config")
    def test_cpu_mode_passes_device_cpu(self, mock_cfg):
        mock_cfg.INFERENCE_MODE = "cpu"
        with patch("models.embeddings.EmbeddingModel") as MockEmbed:
            MockEmbed.return_value = MagicMock()
            create_embedding_model()
            MockEmbed.assert_called_once_with(device="cpu")

    @patch("models.gpu_client.config")
    def test_local_mode_imports_local_class(self, mock_cfg):
        mock_cfg.INFERENCE_MODE = "local"
        with patch("models.embeddings.EmbeddingModel") as MockEmbed:
            MockEmbed.return_value = MagicMock()
            create_embedding_model()
            MockEmbed.assert_called_once_with()


class TestCreateEvaluator:
    @patch("models.gpu_client.config")
    def test_remote_mode_returns_remote(self, mock_cfg):
        mock_cfg.INFERENCE_MODE = "remote"
        mock_cfg.CROSS_ENCODER_MODEL = "test-ce"
        mock_cfg.EVALUATOR_THRESHOLD = 25
        mock_cfg.EVALUATOR_BATCH_SIZE = 32
        mock_cfg.EVALUATOR_MAX_TOKENS = 480
        mock_cfg.GPU_SERVER_URL = "http://gpu:8090"
        mock_cfg.GPU_SERVER_API_KEY = ""
        mock_cfg.GPU_SERVER_TIMEOUT = 120

        evaluator = create_evaluator()

        assert isinstance(evaluator, RemoteDocumentEvaluator)

    @patch("models.gpu_client.config")
    def test_cpu_mode_passes_device_cpu(self, mock_cfg):
        mock_cfg.INFERENCE_MODE = "cpu"
        with patch("search.evaluator.evaluator.DocumentEvaluator") as MockEval:
            MockEval.return_value = MagicMock()
            create_evaluator()
            MockEval.assert_called_once_with(device="cpu")

    @patch("models.gpu_client.config")
    def test_local_mode_imports_local_class(self, mock_cfg):
        mock_cfg.INFERENCE_MODE = "local"
        with patch("search.evaluator.evaluator.DocumentEvaluator") as MockEval:
            MockEval.return_value = MagicMock()
            create_evaluator()
            MockEval.assert_called_once_with()


class TestCreateLlm:
    @patch("models.gpu_client.config")
    def test_remote_mode_returns_remote(self, mock_cfg):
        mock_cfg.INFERENCE_MODE = "remote"
        mock_cfg.OLLAMA_MODEL = "llama3.2:3b"
        mock_cfg.LLM_TEMPERATURE = 0.7
        mock_cfg.LLM_TOP_P = 0.9
        mock_cfg.LLM_MAX_TOKENS = 1024
        mock_cfg.LLM_THINK = None
        mock_cfg.GPU_SERVER_URL = "http://gpu:8090"
        mock_cfg.GPU_SERVER_API_KEY = ""
        mock_cfg.GPU_SERVER_TIMEOUT = 120

        llm = create_llm()

        assert isinstance(llm, RemoteLlamaModel)
        assert llm.model_name == "llama3.2:3b"

    @patch("models.gpu_client.config")
    def test_remote_mode_with_custom_model(self, mock_cfg):
        mock_cfg.INFERENCE_MODE = "remote"
        mock_cfg.OLLAMA_MODEL = "default"
        mock_cfg.LLM_TEMPERATURE = 0.7
        mock_cfg.LLM_TOP_P = 0.9
        mock_cfg.LLM_MAX_TOKENS = 1024
        mock_cfg.LLM_THINK = None
        mock_cfg.GPU_SERVER_URL = "http://gpu:8090"
        mock_cfg.GPU_SERVER_API_KEY = ""
        mock_cfg.GPU_SERVER_TIMEOUT = 120

        llm = create_llm(model_name="llama3.1:70b")

        assert llm.model_name == "llama3.1:70b"

    @patch("models.gpu_client.config")
    def test_local_mode_imports_local_class(self, mock_cfg):
        mock_cfg.INFERENCE_MODE = "local"
        with patch("models.llm.LlamaModel") as MockLlm:
            MockLlm.return_value = MagicMock()
            create_llm()
            MockLlm.assert_called_once_with()


# ---------------------------------------------------------------------------
# RemoteEmbeddingModel — HTTP calls
# ---------------------------------------------------------------------------

class TestRemoteEmbeddingModel:
    @patch("models.gpu_client.config")
    def _make_model(self, mock_cfg):
        mock_cfg.EMBEDDING_MODEL = "test-embed"
        mock_cfg.EMBEDDING_DIMENSION = 1024
        mock_cfg.EMBEDDING_BATCH_SIZE = 32
        mock_cfg.GPU_SERVER_URL = "http://gpu:8090"
        mock_cfg.GPU_SERVER_API_KEY = ""
        mock_cfg.GPU_SERVER_TIMEOUT = 120
        return RemoteEmbeddingModel()

    @patch("models.gpu_client._base_url", return_value="http://gpu:8090")
    @patch("models.gpu_client._headers", return_value={"Content-Type": "application/json"})
    @patch("models.gpu_client.config")
    def test_encode_single_text(self, mock_cfg, _h, _u):
        mock_cfg.EMBEDDING_MODEL = "test-embed"
        mock_cfg.EMBEDDING_DIMENSION = 1024
        mock_cfg.EMBEDDING_BATCH_SIZE = 32
        mock_cfg.GPU_SERVER_URL = "http://gpu:8090"
        mock_cfg.GPU_SERVER_API_KEY = ""
        mock_cfg.GPU_SERVER_TIMEOUT = 120
        model = RemoteEmbeddingModel()

        fake_response = MagicMock()
        fake_response.json.return_value = {
            "embeddings": [list(range(1024))],
            "dimension": 1024,
            "elapsed_ms": 10,
        }
        fake_response.raise_for_status = MagicMock()
        model._client = MagicMock()
        model._client.post.return_value = fake_response

        result = model.encode("test text")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)
        model._client.post.assert_called_once()

    @patch("models.gpu_client._base_url", return_value="http://gpu:8090")
    @patch("models.gpu_client._headers", return_value={"Content-Type": "application/json"})
    @patch("models.gpu_client.config")
    def test_encode_batch_texts(self, mock_cfg, _h, _u):
        mock_cfg.EMBEDDING_MODEL = "test-embed"
        mock_cfg.EMBEDDING_DIMENSION = 1024
        mock_cfg.EMBEDDING_BATCH_SIZE = 32
        mock_cfg.GPU_SERVER_URL = "http://gpu:8090"
        mock_cfg.GPU_SERVER_API_KEY = ""
        mock_cfg.GPU_SERVER_TIMEOUT = 120
        model = RemoteEmbeddingModel()

        fake_response = MagicMock()
        fake_response.json.return_value = {
            "embeddings": [list(range(1024))] * 3,
            "dimension": 1024,
            "elapsed_ms": 20,
        }
        fake_response.raise_for_status = MagicMock()
        model._client = MagicMock()
        model._client.post.return_value = fake_response

        result = model.encode(["a", "b", "c"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1024)

    @patch("models.gpu_client._base_url", return_value="http://gpu:8090")
    @patch("models.gpu_client._headers", return_value={"Content-Type": "application/json"})
    @patch("models.gpu_client.config")
    def test_encode_large_batch_uses_client_batching(self, mock_cfg, _h, _u):
        """When texts exceed CLIENT_BATCH_SIZE, encode splits into multiple HTTP requests."""
        mock_cfg.EMBEDDING_MODEL = "test-embed"
        mock_cfg.EMBEDDING_DIMENSION = 4
        mock_cfg.EMBEDDING_BATCH_SIZE = 32
        mock_cfg.GPU_SERVER_URL = "http://gpu:8090"
        mock_cfg.GPU_SERVER_API_KEY = ""
        mock_cfg.GPU_SERVER_TIMEOUT = 120
        model = RemoteEmbeddingModel()
        model.CLIENT_BATCH_SIZE = 3  # small value to trigger batching

        def _fake_post(url, headers, json):
            n = len(json["texts"])
            resp = MagicMock()
            resp.json.return_value = {
                "embeddings": [[float(i)] * 4 for i in range(n)],
                "dimension": 4,
                "elapsed_ms": 5,
            }
            resp.raise_for_status = MagicMock()
            return resp

        model._client = MagicMock()
        model._client.post.side_effect = _fake_post

        result = model.encode([f"text_{i}" for i in range(7)])

        assert isinstance(result, np.ndarray)
        assert result.shape == (7, 4)
        assert model._client.post.call_count == 3  # ceil(7/3) = 3 HTTP requests

    @patch("models.gpu_client._base_url", return_value="http://gpu:8090")
    @patch("models.gpu_client._headers", return_value={"Content-Type": "application/json"})
    @patch("models.gpu_client.config")
    def test_get_similarity(self, mock_cfg, _h, _u):
        mock_cfg.EMBEDDING_MODEL = "test-embed"
        mock_cfg.EMBEDDING_DIMENSION = 1024
        mock_cfg.EMBEDDING_BATCH_SIZE = 32
        mock_cfg.GPU_SERVER_URL = "http://gpu:8090"
        mock_cfg.GPU_SERVER_API_KEY = ""
        mock_cfg.GPU_SERVER_TIMEOUT = 120
        model = RemoteEmbeddingModel()

        v = np.random.rand(1024).astype(np.float32)
        v /= np.linalg.norm(v)

        sim = model.get_similarity(v, v)

        assert sim == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# RemoteDocumentEvaluator — HTTP calls
# ---------------------------------------------------------------------------

class TestRemoteDocumentEvaluator:
    @patch("models.gpu_client._base_url", return_value="http://gpu:8090")
    @patch("models.gpu_client._headers", return_value={"Content-Type": "application/json"})
    @patch("models.gpu_client.config")
    def test_evaluate_filters_by_threshold(self, mock_cfg, _h, _u):
        mock_cfg.CROSS_ENCODER_MODEL = "test-ce"
        mock_cfg.EVALUATOR_THRESHOLD = 50
        mock_cfg.EVALUATOR_BATCH_SIZE = 32
        mock_cfg.EVALUATOR_MAX_TOKENS = 480
        mock_cfg.GPU_SERVER_URL = "http://gpu:8090"
        mock_cfg.GPU_SERVER_API_KEY = ""
        mock_cfg.GPU_SERVER_TIMEOUT = 120

        evaluator = RemoteDocumentEvaluator()

        fake_resp = MagicMock()
        fake_resp.json.return_value = {
            "scores": [{"index": 0, "score": 2.0}, {"index": 1, "score": -5.0}],
            "elapsed_ms": 50,
        }
        fake_resp.raise_for_status = MagicMock()
        evaluator._client = MagicMock()
        evaluator._client.post.return_value = fake_resp

        docs = [
            {"text": "relevant doc", "regulation_id": "d1"},
            {"text": "irrelevant doc", "regulation_id": "d2"},
        ]
        result = evaluator.evaluate(docs, "query")

        assert len(result) == 1
        assert result[0].document["regulation_id"] == "d1"
        assert result[0].relevance_score > 50

    @patch("models.gpu_client._base_url", return_value="http://gpu:8090")
    @patch("models.gpu_client._headers", return_value={"Content-Type": "application/json"})
    @patch("models.gpu_client.config")
    def test_evaluate_empty_docs(self, mock_cfg, _h, _u):
        mock_cfg.CROSS_ENCODER_MODEL = "test-ce"
        mock_cfg.EVALUATOR_THRESHOLD = 50
        mock_cfg.EVALUATOR_BATCH_SIZE = 32
        mock_cfg.EVALUATOR_MAX_TOKENS = 480
        mock_cfg.GPU_SERVER_URL = "http://gpu:8090"
        mock_cfg.GPU_SERVER_API_KEY = ""
        mock_cfg.GPU_SERVER_TIMEOUT = 120

        evaluator = RemoteDocumentEvaluator()
        result = evaluator.evaluate([], "query")
        assert result == []


# ---------------------------------------------------------------------------
# RemoteLlamaModel — HTTP calls
# ---------------------------------------------------------------------------

class TestRemoteLlamaModel:
    @staticmethod
    def _setup_cfg(mock_cfg, **overrides):
        defaults = {
            "OLLAMA_MODEL": "test-llm",
            "LLM_TEMPERATURE": 0.7,
            "LLM_TOP_P": 0.9,
            "LLM_MAX_TOKENS": 1024,
            "LLM_THINK": None,
            "GPU_SERVER_URL": "http://gpu:8090",
            "GPU_SERVER_API_KEY": "",
            "GPU_SERVER_TIMEOUT": 120,
        }
        defaults.update(overrides)
        for k, v in defaults.items():
            setattr(mock_cfg, k, v)

    @staticmethod
    def _stub_post(llm, payload):
        fake_resp = MagicMock()
        fake_resp.json.return_value = payload
        fake_resp.raise_for_status = MagicMock()
        llm._client = MagicMock()
        llm._client.post.return_value = fake_resp
        return fake_resp

    @patch("models.gpu_client._base_url", return_value="http://gpu:8090")
    @patch("models.gpu_client._headers", return_value={"Content-Type": "application/json"})
    @patch("models.gpu_client.config")
    def test_generate_returns_content(self, mock_cfg, _h, _u):
        self._setup_cfg(mock_cfg)

        llm = RemoteLlamaModel()
        self._stub_post(llm, {
            "content": "Generated answer.",
            "model": "test-llm",
            "elapsed_ms": 500,
        })

        result = llm.generate("prompt text", system_prompt="system")

        assert result == "Generated answer."
        llm._client.post.assert_called_once()
        call_json = llm._client.post.call_args[1]["json"]
        assert call_json["model"] == "test-llm"
        assert len(call_json["messages"]) == 2
        # ``think`` must NOT be sent unless explicitly requested — keeps wire
        # compat with proxy builds that don't accept it yet.
        assert "think" not in call_json

    @patch("models.gpu_client._base_url", return_value="http://gpu:8090")
    @patch("models.gpu_client._headers", return_value={"Content-Type": "application/json"})
    @patch("models.gpu_client.config")
    def test_chat_sends_messages_directly(self, mock_cfg, _h, _u):
        self._setup_cfg(mock_cfg)

        llm = RemoteLlamaModel()
        self._stub_post(llm, {"content": "Chat reply.", "elapsed_ms": 100})

        msgs = [{"role": "user", "content": "Hello"}]
        result = llm.chat(msgs)

        assert result == "Chat reply."

    # ------------------------------------------------------------------
    # ``think`` parameter — passthrough + opt-out
    # ------------------------------------------------------------------

    @patch("models.gpu_client._base_url", return_value="http://gpu:8090")
    @patch("models.gpu_client._headers", return_value={"Content-Type": "application/json"})
    @patch("models.gpu_client.config")
    def test_generate_forwards_think_from_constructor(self, mock_cfg, _h, _u):
        self._setup_cfg(mock_cfg)

        llm = RemoteLlamaModel(think=False)
        assert llm.think is False
        self._stub_post(llm, {"content": "ok", "elapsed_ms": 10})

        llm.generate("p")

        call_json = llm._client.post.call_args[1]["json"]
        assert call_json["think"] is False

    @patch("models.gpu_client._base_url", return_value="http://gpu:8090")
    @patch("models.gpu_client._headers", return_value={"Content-Type": "application/json"})
    @patch("models.gpu_client.config")
    def test_generate_per_call_think_overrides_default(self, mock_cfg, _h, _u):
        self._setup_cfg(mock_cfg)

        llm = RemoteLlamaModel(think=True)
        self._stub_post(llm, {"content": "ok", "elapsed_ms": 10})

        llm.generate("p", think=False)

        call_json = llm._client.post.call_args[1]["json"]
        assert call_json["think"] is False

    @patch("models.gpu_client._base_url", return_value="http://gpu:8090")
    @patch("models.gpu_client._headers", return_value={"Content-Type": "application/json"})
    @patch("models.gpu_client.config")
    def test_chat_forwards_think(self, mock_cfg, _h, _u):
        self._setup_cfg(mock_cfg)

        llm = RemoteLlamaModel()
        self._stub_post(llm, {"content": "ok", "elapsed_ms": 10})

        llm.chat([{"role": "user", "content": "hi"}], think="medium")

        call_json = llm._client.post.call_args[1]["json"]
        assert call_json["think"] == "medium"

    @patch("models.gpu_client._base_url", return_value="http://gpu:8090")
    @patch("models.gpu_client._headers", return_value={"Content-Type": "application/json"})
    @patch("models.gpu_client.config")
    def test_think_from_config_is_normalised(self, mock_cfg, _h, _u):
        # Production wires this via .env: ``LLM_THINK=false``.
        self._setup_cfg(mock_cfg, LLM_THINK="false")

        llm = RemoteLlamaModel()

        assert llm.think is False

    @patch("models.gpu_client._base_url", return_value="http://gpu:8090")
    @patch("models.gpu_client._headers", return_value={"Content-Type": "application/json"})
    @patch("models.gpu_client.config")
    def test_warns_when_done_reason_length_with_empty_content(
        self, mock_cfg, _h, _u, caplog,
    ):
        """Reproduce the silent-empty-response failure mode for thinking
        models so the client gets a loud warning instead of an empty string."""
        import logging
        from loguru import logger as loguru_logger

        self._setup_cfg(mock_cfg)
        llm = RemoteLlamaModel()
        self._stub_post(llm, {
            "content": "",
            "model": "test-llm",
            "elapsed_ms": 5000,
            "done_reason": "length",
            "eval_count": 512,
            "thinking": "internal reasoning that ate the budget" * 20,
        })

        # Loguru → stdlib logging bridge so caplog can capture it.
        handler_id = loguru_logger.add(
            lambda m: logging.getLogger("loguru").warning(m), level="WARNING",
        )
        try:
            with caplog.at_level(logging.WARNING, logger="loguru"):
                result = llm.generate("p")
        finally:
            loguru_logger.remove(handler_id)

        assert result == ""
        assert any(
            "done_reason=length" in rec.message for rec in caplog.records
        ), f"expected truncation warning, got: {[r.message for r in caplog.records]}"

    @patch("models.gpu_client._base_url", return_value="http://gpu:8090")
    @patch("models.gpu_client._headers", return_value={"Content-Type": "application/json"})
    @patch("models.gpu_client.config")
    def test_extra_response_fields_do_not_break_legacy_return(
        self, mock_cfg, _h, _u,
    ):
        """A new-shape response (with ``thinking``/``done_reason``) must keep
        returning the same plain string the existing callers expect."""
        self._setup_cfg(mock_cfg)
        llm = RemoteLlamaModel()
        self._stub_post(llm, {
            "content": "the answer",
            "model": "test-llm",
            "elapsed_ms": 100,
            "thinking": "step 1, step 2",
            "done_reason": "stop",
            "eval_count": 42,
            "prompt_eval_count": 7,
        })

        assert llm.generate("p") == "the answer"


# ---------------------------------------------------------------------------
# _normalise_think — env/config string parsing
# ---------------------------------------------------------------------------

class TestNormaliseThink:
    @pytest.mark.parametrize("raw,expected", [
        (None, None),
        ("", None),
        ("None", None),
        (True, True),
        (False, False),
        ("true", True),
        ("FALSE", False),
        ("1", True),
        ("0", False),
        ("low", "low"),
        ("MEDIUM", "medium"),
        ("high", "high"),
        ("garbage", None),
    ])
    def test_normalisation(self, raw, expected):
        from models.gpu_client import _normalise_think
        assert _normalise_think(raw) == expected


# ---------------------------------------------------------------------------
# _OllamaListProxy
# ---------------------------------------------------------------------------

class TestOllamaListProxy:
    def test_list_returns_models(self):
        mock_client = MagicMock()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"models": [{"name": "m1"}, {"name": "m2"}]}
        fake_resp.raise_for_status = MagicMock()
        mock_client.get.return_value = fake_resp

        proxy = _OllamaListProxy(mock_client)
        result = proxy.list()

        assert len(result["models"]) == 2

    def test_list_returns_empty_on_error(self):
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("down")

        proxy = _OllamaListProxy(mock_client)
        result = proxy.list()

        assert result == {"models": []}


# ---------------------------------------------------------------------------
# Lazy loading in models/__init__.py
# ---------------------------------------------------------------------------

class TestModelsLazyLoading:
    def test_create_functions_importable(self):
        from models import create_embedding_model, create_evaluator, create_llm

        assert callable(create_embedding_model)
        assert callable(create_evaluator)
        assert callable(create_llm)

    def test_getattr_embedding_model(self):
        import models
        with patch("models.embeddings.EmbeddingModel", create=True) as MockClass:
            MockClass.__name__ = "EmbeddingModel"
            cls = models.__getattr__("EmbeddingModel")
            assert cls is not None

    def test_getattr_llama_model(self):
        import models
        with patch("models.llm.LlamaModel", create=True) as MockClass:
            MockClass.__name__ = "LlamaModel"
            cls = models.__getattr__("LlamaModel")
            assert cls is not None

    def test_getattr_unknown_raises(self):
        import models
        with pytest.raises(AttributeError, match="NoSuchThing"):
            models.__getattr__("NoSuchThing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
