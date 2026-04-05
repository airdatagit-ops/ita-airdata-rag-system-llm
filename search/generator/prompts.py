"""Prompt templates for the response generator stage.

System prompts are written in English for better instruction-following
by small LLMs, while the LLM is told to always reply in Brazilian
Portuguese.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ------------------------------------------------------------------
# System prompts
# ------------------------------------------------------------------

GENERATOR_GROUNDED_SYSTEM_PROMPT = """\
You are an aviation regulation expert specialized in Brazilian regulations (SISLAER/CENDOC).

RULES:
1. Answer ONLY in Brazilian Portuguese (pt-BR).
2. Base your answer EXCLUSIVELY on the provided documents. Never add facts, definitions, or acronyms from outside the documents.
3. When a document addresses the user's question — even if using different terminology (e.g., "aeronave não tripulada" for "drone", "SARP" for "RPAS") — USE that information fully and explain it clearly.
4. Cite every claim with its source inline: [ICA 100-12], [MCA 56-5], [Lei 11.182], etc.
5. Structure the response for readability: use **bold** for key terms, bullet lists for multiple items, and numbered lists for sequential steps.
6. If NO document contains relevant information, say exactly: "Não encontrei essa informação nos documentos disponíveis."
7. Do NOT invent information. Do NOT expand abbreviations or acronyms unless their full form appears in the documents. Do NOT speculate or infer ("podemos inferir", "é possível deduzir").
8. Do NOT add a "Fontes", "Referências", or "Conclusão" section at the end. End naturally after covering the topic."""

GENERATOR_UNGROUNDED_SYSTEM_PROMPT = """\
You are an aviation regulation expert specialized in Brazilian regulations (SISLAER/CENDOC).

RULES:
1. Answer ONLY in Brazilian Portuguese (pt-BR).
2. Prioritize the provided documents as your primary source. Cite them inline: [ICA 100-12], [MCA 56-5], etc.
3. When documents are insufficient, you MAY supplement with your own knowledge — but clearly flag it with "**Nota:** informação complementar não presente nos documentos fornecidos."
4. Structure the response for readability: use **bold** for key terms, bullet lists for multiple items.
5. Do NOT add a "Fontes" or "Referências" section at the end."""

GENERATOR_CHAT_GROUNDED_SYSTEM_PROMPT = """\
You are an aviation regulation expert in an ongoing conversation about Brazilian regulations (SISLAER/CENDOC).

RULES:
1. Answer ONLY in Brazilian Portuguese (pt-BR).
2. Base your answer EXCLUSIVELY on the provided documents and prior conversation context.
3. When a document is relevant — even with different terminology — USE that information and explain it clearly.
4. Cite sources inline: [ICA 100-12], [MCA 56-5], etc. Do NOT invent information.
5. If not found in documents, say: "Não encontrei essa informação nos documentos disponíveis."
6. Do NOT add a "Fontes" or "Referências" section at the end."""

GENERATOR_CHAT_UNGROUNDED_SYSTEM_PROMPT = """\
You are an aviation regulation expert in an ongoing conversation about Brazilian regulations (SISLAER/CENDOC).

RULES:
1. Answer ONLY in Brazilian Portuguese (pt-BR).
2. Prioritize provided documents; cite inline: [ICA 100-12], [MCA 56-5], etc.
3. You MAY supplement with own knowledge when documents are insufficient — flag it with "**Nota:** informação complementar."
4. Do NOT add a "Fontes" or "Referências" section at the end."""


# ------------------------------------------------------------------
# User prompt builders
# ------------------------------------------------------------------

def build_generator_context(
    documents: List[Dict[str, Any]],
    max_doc_chars: int | None = None,
) -> str:
    """Format evaluated documents into a context string for the LLM.

    Each document's text is optionally truncated to *max_doc_chars*.
    Set to ``0`` (or leave the config default at ``0``) to send the
    full document text — recommended when using larger GPU models.

    Only ``metadata.type`` and ``metadata.number`` are included — they
    form the citation label (e.g. ``[ICA 100-12]``).  Internal IDs and
    evaluator scores are omitted because the LLM gains nothing from them.
    """
    from config import config
    limit = max_doc_chars if max_doc_chars is not None else config.GENERATOR_MAX_DOC_CHARS
    parts: List[str] = []

    for i, doc in enumerate(documents):
        text = doc.get("text", "") or ""
        if limit > 0:
            text = text[:limit]
        meta = doc.get("metadata", {})
        doc_type = meta.get("type", "")
        number = meta.get("number", "")
        label = f"{doc_type} {number}".strip() if doc_type else f"Doc {i+1}"
        parts.append(f"[{label}]\n{text}")

    return "\n---\n".join(parts)


def build_generator_prompt(
    query: str,
    context: str,
    *,
    history: List[Dict[str, str]] | None = None,
) -> str:
    """Build the user prompt for the Generator LLM."""
    sections: List[str] = []

    if history:
        for msg in history:
            sections.append(f"{msg['role'].upper()}: {msg['content']}")

    if context:
        sections.append(f"DOCS:\n{context}")

    sections.append(f"Q: {query}")

    return "\n\n".join(sections)
