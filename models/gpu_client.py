"""Remote GPU inference client.

Provides drop-in replacements for ``EmbeddingModel`` and ``DocumentEvaluator``
that delegate computation to the GPU inference server over HTTP.  Also provides
``RemoteLlamaModel`` that proxies Ollama calls through the GPU server.

All classes expose the same public API as their local counterparts so that
callers (``api/server.py``, pipeline scripts, etc.) need zero changes.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import httpx
from loguru import logger

from config import config

_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=float(config.GPU_SERVER_TIMEOUT),
    write=60.0,
    pool=10.0,
)


def _headers() -> Dict[str, str]:
    h: Dict[str, str] = {"Content-Type": "application/json"}
    if config.GPU_SERVER_API_KEY:
        h["X-API-Key"] = config.GPU_SERVER_API_KEY
    return h


def _base_url() -> str:
    return config.GPU_SERVER_URL.rstrip("/")



# ---------------------------------------------------------------------------
# RemoteEmbeddingModel — drop-in for models.embeddings.EmbeddingModel
# ---------------------------------------------------------------------------

class RemoteEmbeddingModel:
    """Calls the GPU server's ``/v1/embeddings`` endpoint."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cache_dir: str | None = None,
    ):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.device = "remote"
        self.dimension = config.EMBEDDING_DIMENSION
        self._client = httpx.Client(timeout=_TIMEOUT)
        logger.info(
            f"RemoteEmbeddingModel → {_base_url()}/v1/embeddings "
            f"(model={self.model_name})"
        )

    # Max texts per HTTP request to avoid payload/timeout issues during ingestion
    CLIENT_BATCH_SIZE = 512

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int | None = None,
        show_progress: bool = False,
        normalize: bool = True,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            was_single = True
        else:
            was_single = False

        server_batch = batch_size or config.EMBEDDING_BATCH_SIZE
        n = len(texts)

        if n <= self.CLIENT_BATCH_SIZE:
            result = self._encode_chunk(texts, normalize, server_batch)
            return result[0] if was_single else result

        all_embs: List[np.ndarray] = []
        t0 = time.time()

        chunks = range(0, n, self.CLIENT_BATCH_SIZE)
        if show_progress:
            from tqdm import tqdm
            chunks = tqdm(
                chunks,
                total=(n + self.CLIENT_BATCH_SIZE - 1) // self.CLIENT_BATCH_SIZE,
                desc="Remote encode",
                unit="batch",
            )

        for start in chunks:
            chunk_texts = texts[start : start + self.CLIENT_BATCH_SIZE]
            embs = self._encode_chunk(chunk_texts, normalize, server_batch)
            all_embs.append(embs)

        result = np.vstack(all_embs)
        elapsed = time.time() - t0
        logger.info(
            f"Remote encode: {n} texts in {elapsed:.1f}s "
            f"({n / elapsed:.0f} texts/s, "
            f"{(n + self.CLIENT_BATCH_SIZE - 1) // self.CLIENT_BATCH_SIZE} HTTP requests)"
        )
        return result

    def _encode_chunk(
        self,
        texts: List[str],
        normalize: bool,
        server_batch: int,
    ) -> np.ndarray:
        """Send a single batch of texts to the GPU server."""
        resp = self._client.post(
            f"{_base_url()}/v1/embeddings",
            headers=_headers(),
            json={
                "texts": texts,
                "normalize": normalize,
                "batch_size": server_batch,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self.dimension = data["dimension"]
        return np.array(data["embeddings"], dtype=np.float32)

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        return self.encode(texts, batch_size=batch_size, show_progress=show_progress)

    def get_similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray],
    ) -> float:
        if isinstance(text1, str):
            emb1 = self.encode(text1)
        else:
            emb1 = text1
        if isinstance(text2, str):
            emb2 = self.encode(text2)
        else:
            emb2 = text2
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def __repr__(self) -> str:
        return f"RemoteEmbeddingModel(url={_base_url()}, model={self.model_name})"


# ---------------------------------------------------------------------------
# RemoteDocumentEvaluator — drop-in for search.evaluator.evaluator.DocumentEvaluator
# ---------------------------------------------------------------------------

class RemoteDocumentEvaluator:
    """Calls the GPU server's ``/v1/rerank`` endpoint."""

    def __init__(
        self,
        model_name: str | None = None,
        threshold: int | None = None,
        batch_size: int | None = None,
        max_eval_tokens: int | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name or config.CROSS_ENCODER_MODEL
        self.threshold = threshold if threshold is not None else config.EVALUATOR_THRESHOLD
        self.batch_size = batch_size or config.EVALUATOR_BATCH_SIZE
        self.max_eval_tokens = max_eval_tokens or config.EVALUATOR_MAX_TOKENS
        self._client = httpx.Client(timeout=_TIMEOUT)
        logger.info(
            f"RemoteDocumentEvaluator → {_base_url()}/v1/rerank "
            f"(model={self.model_name}, threshold={self.threshold})"
        )

    @property
    def model(self):
        """Compatibility shim — remote evaluator has no local model."""
        return self

    def predict(self, *args, **kwargs):
        raise NotImplementedError("Remote evaluator — use evaluate() instead")

    def evaluate(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        *,
        threshold: int | None = None,
    ):
        from search.shared.schemas import EvaluatedDocument

        if not documents:
            return []

        effective_threshold = threshold if threshold is not None else self.threshold

        t0 = time.time()
        resp = self._client.post(
            f"{_base_url()}/v1/rerank",
            headers=_headers(),
            json={
                "query": query,
                "documents": documents,
                "max_tokens": self.max_eval_tokens,
                "batch_size": self.batch_size,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        raw_scores = np.array([s["score"] for s in data["scores"]], dtype=np.float64)
        normalised = 1.0 / (1.0 + np.exp(-raw_scores)) * 100.0

        evaluated = []
        for item, score in zip(documents, normalised):
            if score >= effective_threshold:
                evaluated.append(
                    EvaluatedDocument(
                        document=item,
                        relevance_score=round(float(score), 2),
                        query_text=query,
                    )
                )

        evaluated.sort(key=lambda e: e.relevance_score, reverse=True)

        elapsed = time.time() - t0
        logger.info(
            f"Remote evaluator: {len(evaluated)}/{len(documents)} docs above "
            f"threshold {effective_threshold} ({elapsed:.2f}s, server {data['elapsed_ms']}ms)"
        )
        return evaluated

    def evaluate_multi_query(
        self,
        documents: List[Dict[str, Any]],
        queries: List[str],
        *,
        threshold: int | None = None,
    ):
        from search.shared.schemas import EvaluatedDocument

        if not documents or not queries:
            return []

        effective_threshold = threshold if threshold is not None else self.threshold
        best_scores: Dict[int, tuple[float, str]] = {}

        for query in queries:
            try:
                resp = self._client.post(
                    f"{_base_url()}/v1/rerank",
                    headers=_headers(),
                    json={
                        "query": query,
                        "documents": documents,
                        "max_tokens": self.max_eval_tokens,
                        "batch_size": self.batch_size,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                raw = np.array([s["score"] for s in data["scores"]], dtype=np.float64)
                normalised = 1.0 / (1.0 + np.exp(-raw)) * 100.0
                for idx, score in enumerate(normalised):
                    current_best, _ = best_scores.get(idx, (0.0, ""))
                    if score > current_best:
                        best_scores[idx] = (float(score), query)
            except Exception as exc:
                logger.warning(f"Remote rerank failed for query '{query[:50]}': {exc}")

        evaluated = []
        for idx, (score, best_query) in best_scores.items():
            if score >= effective_threshold:
                evaluated.append(
                    EvaluatedDocument(
                        document=documents[idx],
                        relevance_score=round(score, 2),
                        query_text=best_query,
                    )
                )

        evaluated.sort(key=lambda e: e.relevance_score, reverse=True)
        logger.info(
            f"Remote evaluator (multi-query): {len(evaluated)}/{len(documents)} docs "
            f"above threshold {effective_threshold}"
        )
        return evaluated

    def __repr__(self) -> str:
        return f"RemoteDocumentEvaluator(url={_base_url()}, model={self.model_name})"


# ---------------------------------------------------------------------------
# RemoteLlamaModel — drop-in for models.llm.LlamaModel
# ---------------------------------------------------------------------------

class RemoteLlamaModel:
    """Proxies LLM calls through the GPU server's ``/v1/generate`` endpoint.

    Maintains the same public API as ``LlamaModel`` so it can be used as a
    drop-in replacement.
    """

    def __init__(
        self,
        model_name: str | None = None,
        host: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ):
        self.model_name = model_name or config.OLLAMA_MODEL
        self.host = host or _base_url()
        self.temperature = temperature or config.LLM_TEMPERATURE
        self.top_p = top_p or config.LLM_TOP_P
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS
        self.default_options = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.max_tokens,
        }
        self._client = httpx.Client(timeout=_TIMEOUT)
        self._stream_client = httpx.Client(timeout=httpx.Timeout(None))

        # Expose a lightweight Ollama client for model listing in server.py
        self.client = _OllamaListProxy(self._client)

        logger.info(
            f"RemoteLlamaModel → {_base_url()}/v1/generate "
            f"(model={self.model_name})"
        )

    def _build_options(self, temperature=None, top_p=None, max_tokens=None, extra=None):
        options = self.default_options.copy()
        if extra:
            options.update(extra)
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        return options

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        format: dict | None = None,
        extra_options: dict | None = None,
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        options = self._build_options(temperature, top_p, max_tokens, extra_options)

        if stream:
            return self._remote_stream(messages, options)

        t0 = time.time()
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "options": options,
        }
        if format is not None:
            payload["format"] = format

        resp = self._client.post(
            f"{_base_url()}/v1/generate",
            headers=_headers(),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        logger.debug(
            f"Remote generate: {time.time()-t0:.2f}s (server {data['elapsed_ms']}ms)"
        )
        return data["content"]

    def _remote_stream(self, messages, options):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "options": options,
            "stream": True,
        }
        with self._stream_client.stream(
            "POST",
            f"{_base_url()}/v1/generate/stream",
            headers=_headers(),
            json=payload,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload_str = line[6:]
                if payload_str == "[DONE]":
                    return
                try:
                    data = json.loads(payload_str)
                    if "content" in data:
                        yield data["content"]
                    elif "error" in data:
                        raise RuntimeError(f"GPU server stream error: {data['error']}")
                except json.JSONDecodeError:
                    continue

    def generate_with_context(
        self,
        query: str,
        context_documents: List[Dict],
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:
        from search.prompts import SYSTEM_PROMPT, build_context_string, build_rag_prompt

        context_str = build_context_string(context_documents)
        prompt = build_rag_prompt(query, context_str)
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt or SYSTEM_PROMPT,
            **kwargs,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ):
        options = self._build_options(temperature, top_p, max_tokens)

        if stream:
            return self._remote_stream(messages, options)

        resp = self._client.post(
            f"{_base_url()}/v1/generate",
            headers=_headers(),
            json={
                "model": self.model_name,
                "messages": messages,
                "options": options,
            },
        )
        resp.raise_for_status()
        return resp.json()["content"]

    def _test_connection(self) -> bool:
        try:
            resp = self._client.get(f"{_base_url()}/health")
            resp.raise_for_status()
            logger.success(f"GPU server connection OK: {resp.json()}")
            return True
        except Exception as e:
            logger.error(f"Cannot reach GPU server at {_base_url()}: {e}")
            raise ConnectionError(f"GPU server unreachable: {e}")

    def __repr__(self) -> str:
        return f"RemoteLlamaModel(url={_base_url()}, model={self.model_name})"


class _OllamaListProxy:
    """Minimal proxy that supports ``client.list()`` by hitting the GPU
    server's ``/v1/models`` endpoint (which proxies to Ollama)."""

    def __init__(self, http_client: httpx.Client):
        self._http = http_client

    def list(self):
        try:
            resp = self._http.get(
                f"{_base_url()}/v1/models", headers=_headers(),
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Could not list Ollama models via GPU server: {e}")
            return {"models": []}


# ---------------------------------------------------------------------------
# Factory — pick the right backend based on INFERENCE_MODE
# ---------------------------------------------------------------------------

def create_embedding_model(**kwargs) -> Union["EmbeddingModel", RemoteEmbeddingModel]:
    """Create an embedding model using the configured inference mode."""
    mode = config.INFERENCE_MODE.lower()

    if mode == "remote":
        return RemoteEmbeddingModel(**kwargs)

    from models.embeddings import EmbeddingModel

    if mode == "cpu":
        kwargs.setdefault("device", "cpu")

    return EmbeddingModel(**kwargs)


def create_evaluator(**kwargs) -> Union["DocumentEvaluator", RemoteDocumentEvaluator]:
    """Create a document evaluator using the configured inference mode."""
    mode = config.INFERENCE_MODE.lower()

    if mode == "remote":
        return RemoteDocumentEvaluator(**kwargs)

    from search.evaluator.evaluator import DocumentEvaluator

    if mode == "cpu":
        kwargs.setdefault("device", "cpu")

    return DocumentEvaluator(**kwargs)


def create_llm(**kwargs) -> Union["LlamaModel", RemoteLlamaModel]:
    """Create an LLM model using the configured inference mode."""
    mode = config.INFERENCE_MODE.lower()

    if mode == "remote":
        return RemoteLlamaModel(**kwargs)

    from models.llm import LlamaModel
    return LlamaModel(**kwargs)
