"""Tests for the ResponseGenerator module."""

from unittest.mock import MagicMock

import pytest

from search.generator.generator import ResponseGenerator
from search.generator.prompts import build_generator_context
from search.shared.schemas import EvaluatedDocument


SAMPLE_EVALUATED = [
    EvaluatedDocument(
        document={
            "text": "Art. 1 texto do regulamento",
            "regulation_id": "ICA-100-12",
            "score": 0.9,
            "version": "2023",
            "metadata": {"authority": "ANAC"},
        },
        relevance_score=85.0,
        query_text="certificação",
    ),
    EvaluatedDocument(
        document={
            "text": "Art. 2 outro regulamento",
            "regulation_id": "DCA-200-5",
            "score": 0.8,
            "metadata": {},
        },
        relevance_score=72.0,
        query_text="certificação",
    ),
]


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.model_name = "test-model"
    llm.generate.return_value = "Resposta baseada nos documentos."
    return llm


@pytest.fixture
def generator(mock_llm):
    return ResponseGenerator(llm=mock_llm, grounded_only=True, timeout=10)


class TestGenerateBasic:
    def test_returns_answer_string(self, generator):
        result = generator.generate(SAMPLE_EVALUATED, "minha pergunta")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_does_not_append_references(self, generator, mock_llm):
        """References are rendered by the frontend, not appended by the generator."""
        mock_llm.generate.return_value = "Resposta."
        result = generator.generate(SAMPLE_EVALUATED, "pergunta")

        assert "Fontes:" not in result


class TestGroundedOnlyOverride:
    def test_override_per_call(self, generator, mock_llm):
        mock_llm.generate.return_value = "Resposta."

        result_grounded = generator.generate(SAMPLE_EVALUATED, "q", grounded_only=True)
        result_ungrounded = generator.generate(SAMPLE_EVALUATED, "q", grounded_only=False)

        assert isinstance(result_grounded, str)
        assert isinstance(result_ungrounded, str)


class TestStreamingGeneration:
    def test_stream_returns_generator(self, generator, mock_llm):
        mock_llm.generate.return_value = iter(["chunk1", "chunk2"])

        result = generator.generate(SAMPLE_EVALUATED, "q", stream=True)

        chunks = list(result)
        assert len(chunks) >= 2


class TestPromptBuilders:
    def test_build_context_includes_metadata(self):
        docs = [
            {
                "text": "Content",
                "regulation_id": "ICA-1",
                "metadata": {
                    "type": "ICA", "number": "100-12",
                    "authority": "DECEA", "title": "Regras de Tráfego Aéreo",
                },
            },
        ]
        context = build_generator_context(docs)

        assert "ICA 100-12" in context
        assert "Content" in context
        assert "ICA-1" not in context
        assert "DECEA" in context
        assert "Regras de Tráfego Aéreo" in context

class TestSystemPromptSelection:
    def test_grounded_no_history(self, generator):
        prompt = generator._select_system_prompt(grounded=True, has_history=False)
        assert "ONLY" in prompt

    def test_ungrounded_no_history(self, generator):
        prompt = generator._select_system_prompt(grounded=False, has_history=False)
        assert "Prioritize" in prompt

    def test_grounded_with_history(self, generator):
        prompt = generator._select_system_prompt(grounded=True, has_history=True)
        assert "conversation" in prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
