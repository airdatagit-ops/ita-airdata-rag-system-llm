"""Document evaluator module (cross-encoder reranker).

Scores each retrieved document against the query using a cross-encoder
model and filters out documents below a configurable relevance threshold.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from config import config
from search.shared.exceptions import EvaluatorError
from search.shared.schemas import EvaluatedDocument


class DocumentEvaluator:
    """Reranks documents using a cross-encoder model.

    Cross-encoders are *discriminative* models — they receive a
    ``(query, document)`` pair and output a relevance score directly.
    No prompts are needed.

    Parameters
    ----------
    model_name:
        HuggingFace cross-encoder model identifier.
    threshold:
        Minimum normalised score (0-100) to keep a document.
    batch_size:
        Number of ``(query, doc)`` pairs per ``.predict()`` call.
    device:
        ``"cuda"``, ``"cpu"`` or ``None`` for auto-detect.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        threshold: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or config.CROSS_ENCODER_MODEL
        self.threshold = threshold if threshold is not None else config.EVALUATOR_THRESHOLD
        self.batch_size = batch_size or config.EVALUATOR_BATCH_SIZE
        self._model = None
        self._device = device
        logger.info(
            f"DocumentEvaluator initialized (model={self.model_name}, "
            f"threshold={self.threshold}, batch_size={self.batch_size}) "
            f"— model will be loaded on first use"
        )

    @property
    def model(self):
        """Lazy-load the cross-encoder to avoid startup overhead when
        the evaluator is not needed (e.g. vector-only search)."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise EvaluatorError(
                "sentence-transformers is required for the Evaluator. "
                "Install with: pip install sentence-transformers"
            ) from exc

        logger.info(f"Loading cross-encoder model: {self.model_name}")
        try:
            kwargs = {}
            if self._device:
                kwargs["device"] = self._device
            self._model = CrossEncoder(self.model_name, **kwargs)
            logger.success(f"Cross-encoder loaded: {self.model_name}")
        except Exception as exc:
            raise EvaluatorError(f"Failed to load cross-encoder: {exc}") from exc

    def evaluate(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        *,
        threshold: Optional[int] = None,
    ) -> List[EvaluatedDocument]:
        """Score and filter *documents* against *query*.

        Returns only documents whose normalised score (0-100) is at or
        above *threshold*, sorted by score descending.
        """
        if not documents:
            return []

        effective_threshold = threshold if threshold is not None else self.threshold

        try:
            scores = self._score_batch(documents, query)
        except Exception as exc:
            raise EvaluatorError(f"Cross-encoder scoring failed: {exc}") from exc

        normalised = self._normalise_scores(scores)

        evaluated: List[EvaluatedDocument] = []
        for doc, score in zip(documents, normalised):
            if score >= effective_threshold:
                evaluated.append(
                    EvaluatedDocument(
                        document=doc,
                        relevance_score=round(score, 2),
                        query_text=query,
                    )
                )

        evaluated.sort(key=lambda e: e.relevance_score, reverse=True)

        logger.info(
            f"Evaluator: {len(evaluated)}/{len(documents)} docs above "
            f"threshold {effective_threshold} for query: {query[:50]}..."
        )
        return evaluated

    def evaluate_multi_query(
        self,
        documents: List[Dict[str, Any]],
        queries: List[str],
        *,
        threshold: Optional[int] = None,
    ) -> List[EvaluatedDocument]:
        """Evaluate documents against multiple queries.

        Each document is scored against every query; the **best** score
        per document is used for threshold filtering.
        """
        if not documents or not queries:
            return []

        effective_threshold = threshold if threshold is not None else self.threshold

        best_scores: Dict[int, tuple[float, str]] = {}

        for query in queries:
            try:
                scores = self._score_batch(documents, query)
                normalised = self._normalise_scores(scores)
            except Exception as exc:
                logger.warning(f"Scoring failed for query '{query[:50]}': {exc}")
                continue

            for idx, score in enumerate(normalised):
                current_best, _ = best_scores.get(idx, (0.0, ""))
                if score > current_best:
                    best_scores[idx] = (score, query)

        evaluated: List[EvaluatedDocument] = []
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
            f"Evaluator (multi-query): {len(evaluated)}/{len(documents)} docs "
            f"above threshold {effective_threshold}"
        )
        return evaluated

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score_batch(
        self, documents: List[Dict[str, Any]], query: str,
    ) -> np.ndarray:
        """Run cross-encoder prediction on (query, doc_text) pairs."""
        pairs = [
            [query, doc.get("text", "")]
            for doc in documents
        ]
        return self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False,
        )

    @staticmethod
    def _normalise_scores(raw_scores: np.ndarray) -> np.ndarray:
        """Map raw logits to 0-100 scale using sigmoid normalisation."""
        sigmoid = 1.0 / (1.0 + np.exp(-raw_scores))
        return sigmoid * 100.0
