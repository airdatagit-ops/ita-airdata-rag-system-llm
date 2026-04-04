"""Tests for the ResponseGenerator module."""

from unittest.mock import MagicMock, patch

import pytest

from search.generator.generator import ResponseGenerator
from search.generator.prompts import (
    GENERATOR_GROUNDED_SYSTEM_PROMPT,
    GENERATOR_UNGROUNDED_SYSTEM_PROMPT,
    build_generator_context,
    build_references_section,
)
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
    with patch("search.generator.generator.LlamaModel", return_value=mock_llm):
        return ResponseGenerator(llm=mock_llm, grounded_only=True, timeout=10)


class TestGenerateBasic:
    def test_returns_answer_string(self, generator):
        result = generator.generate(SAMPLE_EVALUATED, "minha pergunta")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_appends_references_when_grounded(self, generator, mock_llm):
        mock_llm.generate.return_value = "Resposta."
        result = generator.generate(SAMPLE_EVALUATED, "pergunta")

        assert "Fontes:" in result
        assert "ICA-100-12" in result
        assert "DCA-200-5" in result

    def test_no_references_when_ungrounded(self, mock_llm):
        with patch("search.generator.generator.LlamaModel", return_value=mock_llm):
            gen = ResponseGenerator(llm=mock_llm, grounded_only=False, timeout=10)

        mock_llm.generate.return_value = "Resposta com conhecimento."
        result = gen.generate(SAMPLE_EVALUATED, "pergunta", grounded_only=False)

        assert "Fontes:" not in result


class TestGroundedOnlyOverride:
    def test_override_per_call(self, generator, mock_llm):
        mock_llm.generate.return_value = "Resposta."

        result_grounded = generator.generate(SAMPLE_EVALUATED, "q", grounded_only=True)
        result_ungrounded = generator.generate(SAMPLE_EVALUATED, "q", grounded_only=False)

        assert "Fontes:" in result_grounded
        assert "Fontes:" not in result_ungrounded


class TestStreamingGeneration:
    def test_stream_returns_generator(self, generator, mock_llm):
        mock_llm.generate.return_value = iter(["chunk1", "chunk2"])

        result = generator.generate(SAMPLE_EVALUATED, "q", stream=True)

        chunks = list(result)
        assert len(chunks) >= 2


class TestPromptBuilders:
    def test_build_context_includes_metadata(self):
        docs = [
            {"text": "Content", "regulation_id": "ICA-1", "version": "2024"},
        ]
        context = build_generator_context(docs, scores=[85.0])

        assert "ICA-1" in context
        assert "Versão 2024" in context
        assert "Relevância: 85" in context

    def test_build_references_deduplicates(self):
        docs = [
            {"regulation_id": "ICA-1", "metadata": {"authority": "ANAC"}},
            {"regulation_id": "ICA-1", "metadata": {"authority": "ANAC"}},
            {"regulation_id": "ICA-2", "metadata": {}},
        ]
        refs = build_references_section(docs)

        assert refs.count("ICA-1") == 1
        assert "ICA-2" in refs

    def test_build_references_empty_docs(self):
        assert build_references_section([]) == ""


class TestSystemPromptSelection:
    def test_grounded_no_history(self, generator):
        prompt = generator._select_system_prompt(grounded=True, has_history=False)
        assert "APENAS" in prompt

    def test_ungrounded_no_history(self, generator):
        prompt = generator._select_system_prompt(grounded=False, has_history=False)
        assert "Priorize" in prompt

    def test_grounded_with_history(self, generator):
        prompt = generator._select_system_prompt(grounded=True, has_history=True)
        assert "conversa" in prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
