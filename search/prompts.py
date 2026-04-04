"""Centralized prompt templates for the RAG pipeline."""

from typing import Dict, List


SYSTEM_PROMPT = (
    "Você é um assistente especializado em regulamentação de aviação civil brasileira. "
    "Responda sempre em português, de forma clara e precisa. "
    "Quando normas regulatórias forem fornecidas, baseie suas respostas apenas nelas e "
    "cite as fontes (número da lei/regulamento e artigo). "
    "Se a informação necessária não estiver nos documentos fornecidos, diga claramente. "
    "Se não houver normas fornecidas, responda com seu conhecimento geral sobre aviação "
    "e sugira ao usuário ativar a busca com RAG para obter informações precisas."
)


def build_context_string(documents: List[Dict]) -> str:
    """Build formatted context string from retrieved documents."""
    parts = []
    for i, doc in enumerate(documents, 1):
        text = doc.get("text", "")
        reg_id = doc.get("regulation_id", f"documento-{i}")
        version = doc.get("version", "")

        header = f"[{reg_id}"
        if version:
            header += f" - Versão {version}"
        header += "]"

        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


def build_rag_prompt(query: str, context: str) -> str:
    """Build a standalone RAG prompt (no conversation history)."""
    return (
        f"=== NORMAS REGULATÓRIAS ===\n{context}\n\n"
        f"=== PERGUNTA ===\n{query}\n\n"
        f"=== RESPOSTA ===\nBaseado nas normas fornecidas:\n"
    )


def build_chat_prompt(
    query: str,
    context: str | None,
    history: List[Dict[str, str]],
) -> str:
    """Build a chat prompt with optional RAG context and conversation history."""
    parts: list[str] = []

    if history:
        parts.append("=== HISTÓRICO DA CONVERSA ===")
        for msg in history:
            parts.append(f"{msg['role'].upper()}: {msg['content']}")
        parts.append("")

    if context:
        parts.append(f"=== NORMAS REGULATÓRIAS ===\n{context}\n")

    parts.append(f"=== PERGUNTA ATUAL ===\n{query}\n")

    if context:
        parts.append("=== RESPOSTA ===\nBaseado nas normas fornecidas e no contexto da conversa:\n")
    else:
        parts.append("=== RESPOSTA ===\n")

    return "\n".join(parts)
