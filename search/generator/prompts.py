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
Aviation regulation expert (SISLAER/CENDOC). Reply in pt-BR.
Use ONLY the provided documents. Cite sources inline, e.g. [ICA 100-12].
Use **bold** for key terms, lists for multiple items.
If not found, say "Não encontrei essa informação nos documentos disponíveis."
Do NOT add a Fontes/Referências section at the end."""

GENERATOR_UNGROUNDED_SYSTEM_PROMPT = """\
Aviation regulation expert (SISLAER/CENDOC). Reply in pt-BR.
Prioritize provided documents; supplement with your knowledge if needed.
Cite sources inline, e.g. [MCA 56-5]. Flag own knowledge with "**Nota:**".
Use **bold** for key terms, lists for multiple items.
Do NOT add a Fontes/Referências section at the end."""

GENERATOR_CHAT_GROUNDED_SYSTEM_PROMPT = """\
Aviation regulation expert in ongoing conversation. Reply in pt-BR.
Use ONLY the provided documents and prior conversation context.
Cite sources inline, e.g. [ICA 100-12]. Do NOT invent information.
Do NOT add a Fontes/Referências section at the end."""

GENERATOR_CHAT_UNGROUNDED_SYSTEM_PROMPT = """\
Aviation regulation expert in ongoing conversation. Reply in pt-BR.
Prioritize provided documents; supplement with own knowledge if needed.
Cite sources inline, e.g. [MCA 56-5]. Flag own knowledge with "**Nota:**".
Do NOT add a Fontes/Referências section at the end."""


# ------------------------------------------------------------------
# User prompt builders
# ------------------------------------------------------------------

def build_generator_context(
    documents: List[Dict[str, Any]],
    scores: List[float] | None = None,
    max_doc_chars: int | None = None,
) -> str:
    """Format evaluated documents into a compact context string.

    Each document's text is truncated to *max_doc_chars* to keep the
    total prompt within a reasonable size for CPU inference.
    """
    from config import config
    limit = max_doc_chars or config.GENERATOR_MAX_DOC_CHARS
    parts: List[str] = []

    for i, doc in enumerate(documents):
        text = (doc.get("text", "") or "")[:limit]
        meta = doc.get("metadata", {})
        doc_type = meta.get("type", "")
        number = meta.get("number", "")
        label = f"{doc_type} {number}".strip() if doc_type else doc.get("regulation_id", f"doc-{i+1}")
        score_str = f" score={scores[i]:.0f}" if scores and i < len(scores) else ""
        parts.append(f"[{label}{score_str}]\n{text}")

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
