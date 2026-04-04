"""Tests for the centralized prompts module."""

import pytest

from search.prompts import (
    SYSTEM_PROMPT,
    build_context_string,
    build_rag_prompt,
    build_chat_prompt,
    build_search_query,
)


class TestSystemPrompt:
    def test_is_non_empty_string(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 50

    def test_mentions_aviation(self):
        assert "aviação" in SYSTEM_PROMPT.lower()


class TestBuildContextString:
    def test_single_document(self):
        docs = [{"text": "Art. 1º Texto.", "regulation_id": "lei-123"}]
        result = build_context_string(docs)
        assert "[lei-123]" in result
        assert "Art. 1º Texto." in result

    def test_multiple_documents(self):
        docs = [
            {"text": "Texto A", "regulation_id": "doc-a"},
            {"text": "Texto B", "regulation_id": "doc-b"},
        ]
        result = build_context_string(docs)
        assert "[doc-a]" in result
        assert "[doc-b]" in result
        assert "Texto A" in result
        assert "Texto B" in result

    def test_with_version(self):
        docs = [{"text": "T", "regulation_id": "r1", "version": "2024-01-01"}]
        result = build_context_string(docs)
        assert "Versão 2024-01-01" in result

    def test_missing_regulation_id_uses_fallback(self):
        docs = [{"text": "T"}]
        result = build_context_string(docs)
        assert "[documento-1]" in result

    def test_empty_list(self):
        assert build_context_string([]) == ""


class TestBuildRagPrompt:
    def test_contains_query_and_context(self):
        result = build_rag_prompt("Minha pergunta?", "Contexto aqui")
        assert "Minha pergunta?" in result
        assert "Contexto aqui" in result
        assert "NORMAS REGULATÓRIAS" in result
        assert "PERGUNTA" in result

    def test_contains_response_section(self):
        result = build_rag_prompt("q", "c")
        assert "RESPOSTA" in result


class TestBuildChatPrompt:
    def test_with_context_and_history(self):
        history = [
            {"role": "user", "content": "Olá"},
            {"role": "assistant", "content": "Oi!"},
        ]
        result = build_chat_prompt("Nova pergunta?", "Norma X", history)
        assert "HISTÓRICO" in result
        assert "USER: Olá" in result
        assert "ASSISTANT: Oi!" in result
        assert "Norma X" in result
        assert "Nova pergunta?" in result

    def test_without_context(self):
        result = build_chat_prompt("Pergunta?", None, [])
        assert "NORMAS REGULATÓRIAS" not in result
        assert "Pergunta?" in result

    def test_without_history(self):
        result = build_chat_prompt("Pergunta?", "Ctx", [])
        assert "HISTÓRICO" not in result
        assert "Ctx" in result

    def test_without_context_nor_history(self):
        result = build_chat_prompt("Pergunta?", None, [])
        assert "PERGUNTA ATUAL" in result
        assert "RESPOSTA" in result


class TestBuildSearchQuery:
    def test_no_history_returns_message_as_is(self):
        assert build_search_query("idade mínima?", []) == "idade mínima?"

    def test_prepends_recent_user_messages(self):
        history = [
            {"role": "user", "content": "requisitos para pilotos"},
            {"role": "assistant", "content": "Os requisitos são..."},
            {"role": "user", "content": "e para comerciais?"},
            {"role": "assistant", "content": "Para comerciais..."},
        ]
        result = build_search_query("qual a idade mínima?", history)
        assert "requisitos para pilotos" in result
        assert "e para comerciais?" in result
        assert "qual a idade mínima?" in result
        assert "Os requisitos são" not in result

    def test_limits_history_messages(self):
        history = [
            {"role": "user", "content": f"msg-{i}"}
            for i in range(10)
        ]
        result = build_search_query("atual", history, max_history_messages=2)
        assert "msg-8" in result
        assert "msg-9" in result
        assert "msg-7" not in result
        assert "atual" in result

    def test_only_assistant_messages_returns_current(self):
        history = [
            {"role": "assistant", "content": "Oi!"},
        ]
        assert build_search_query("pergunta", history) == "pergunta"
