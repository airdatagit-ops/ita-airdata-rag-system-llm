"""Tests for the DocumentEvaluator module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from search.evaluator.evaluator import DocumentEvaluator
from search.shared.schemas import EvaluatedDocument


SAMPLE_DOCS = [
    {"text": "Art. 1 sobre certificação", "regulation_id": "doc-1", "score": 0.9},
    {"text": "Art. 2 sobre licitações", "regulation_id": "doc-2", "score": 0.8},
    {"text": "Art. 3 irrelevante", "regulation_id": "doc-3", "score": 0.5},
]


@pytest.fixture
def mock_cross_encoder():
    model = MagicMock()
    model.predict.return_value = np.array([2.0, 0.5, -2.0])
    return model


@pytest.fixture
def evaluator(mock_cross_encoder):
    ev = DocumentEvaluator(threshold=50, batch_size=32)
    ev._model = mock_cross_encoder
    return ev


class TestEvaluateBasic:
    def test_returns_evaluated_documents(self, evaluator):
        result = evaluator.evaluate(SAMPLE_DOCS, "certificação de pilotos")

        assert all(isinstance(ed, EvaluatedDocument) for ed in result)
        assert all(0 <= ed.relevance_score <= 100 for ed in result)

    def test_filters_by_threshold(self, evaluator):
        result = evaluator.evaluate(SAMPLE_DOCS, "certificação", threshold=50)

        assert len(result) < len(SAMPLE_DOCS)
        assert all(ed.relevance_score >= 50 for ed in result)

    def test_sorted_by_score_descending(self, evaluator):
        result = evaluator.evaluate(SAMPLE_DOCS, "test", threshold=0)

        scores = [ed.relevance_score for ed in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_documents_returns_empty(self, evaluator):
        result = evaluator.evaluate([], "test")
        assert result == []


class TestScoreNormalization:
    def test_sigmoid_normalization(self):
        raw = np.array([0.0, 2.0, -2.0, 10.0])
        normalised = DocumentEvaluator._normalise_scores(raw)

        assert all(0 <= s <= 100 for s in normalised)
        assert normalised[0] == pytest.approx(50.0, abs=0.1)
        assert normalised[1] > 80
        assert normalised[2] < 20
        assert normalised[3] > 99


class TestMultiQueryEvaluation:
    def test_multi_query_uses_best_score(self, evaluator, mock_cross_encoder):
        call_count = 0

        def predict_side_effect(pairs, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return np.array([1.0, -1.0, 0.0])
            return np.array([-1.0, 2.0, 0.0])

        mock_cross_encoder.predict.side_effect = predict_side_effect

        result = evaluator.evaluate_multi_query(
            SAMPLE_DOCS, ["query 1", "query 2"], threshold=0,
        )

        assert len(result) == 3
        doc_1_score = next(
            ed.relevance_score for ed in result
            if ed.document["regulation_id"] == "doc-1"
        )
        doc_2_score = next(
            ed.relevance_score for ed in result
            if ed.document["regulation_id"] == "doc-2"
        )
        assert doc_1_score > 50
        assert doc_2_score > 80


class TestBuildEvalText:
    def test_plain_text_only(self):
        doc = {"text": "Art. 1 conteúdo do regulamento", "regulation_id": "doc-1"}
        result = DocumentEvaluator._build_eval_text(doc)
        assert "Art. 1 conteúdo do regulamento" in result

    def test_includes_title_from_metadata(self):
        doc = {
            "text": "Art. 1 texto",
            "regulation_id": "ICA-100-12",
            "metadata": {
                "title": "Instrução sobre Certificação de Pilotos",
                "type": "ICA",
                "number": "100-12",
                "authority": "ANAC",
            },
        }
        result = DocumentEvaluator._build_eval_text(doc)
        assert "Instrução sobre Certificação de Pilotos" in result
        assert "ICA" in result
        assert "nº 100-12" in result
        assert "ANAC" in result
        assert "Art. 1 texto" in result

    def test_falls_back_to_doc_title(self):
        doc = {
            "text": "corpo do texto",
            "title": "Título no nível raiz",
            "metadata": {},
        }
        result = DocumentEvaluator._build_eval_text(doc)
        assert "Título no nível raiz" in result

    def test_truncates_to_max_tokens(self):
        long_text = " ".join(["palavra"] * 600)
        doc = {"text": long_text}
        result = DocumentEvaluator._build_eval_text(doc, max_tokens=100)
        assert len(result.split()) <= 100

    def test_empty_metadata_uses_text_only(self):
        doc = {"text": "apenas texto", "metadata": {}}
        result = DocumentEvaluator._build_eval_text(doc)
        assert result == "apenas texto"

    def test_no_text_with_metadata(self):
        doc = {
            "text": "",
            "metadata": {"title": "Título", "authority": "ANAC"},
        }
        result = DocumentEvaluator._build_eval_text(doc)
        assert "Título" in result
        assert "ANAC" in result

    def test_prefix_before_body(self):
        doc = {
            "text": "corpo do artigo",
            "metadata": {"title": "Meu Título"},
        }
        result = DocumentEvaluator._build_eval_text(doc)
        title_pos = result.index("Meu Título")
        body_pos = result.index("corpo do artigo")
        assert title_pos < body_pos


class TestLazyLoading:
    def test_model_not_loaded_at_init(self):
        ev = DocumentEvaluator(threshold=50)
        assert ev._model is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
