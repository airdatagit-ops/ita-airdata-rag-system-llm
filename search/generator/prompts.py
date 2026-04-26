"""Prompt templates for the response generator stage.

V2 prompts — concise system + no trailing reminder. Optimised after
A/B vs the legacy v1 (with verbose TERMINOLOGY block + recency-bias
ANSWER_REMINDER) on gemma4:26b: V2 yields equivalent latency and
+25% mean citations per response with 100% citation grounding.
"""

from __future__ import annotations

from typing import Any, Dict, List


GENERATOR_GROUNDED_SYSTEM_PROMPT = """\
You are an expert assistant on Brazilian aviation regulations (SISLAER/CENDOC).
Read the provided documents thoroughly and produce a comprehensive, well-organized answer.

RULES:
1. Answer ONLY in Brazilian Portuguese (pt-BR).
2. Base your answer EXCLUSIVELY on the provided documents.
3. Treat technical synonyms as equivalent (e.g. drone = ANT = RPA = RPAS = SARP = UAS = UAV; "regras" = "requisitos" = "procedimentos").
4. EXTRACT actual rules, definitions and requirements from the document texts; SYNTHESIZE them into ONE coherent answer organized by topic, not by document.
5. Cite sources inline as [TYPE NUMBER] — e.g. [ICA 100-40], [MCA 56-5], [Decreto 97.464].
6. Quote key regulatory passages with blockquotes followed by source: > "texto" — [Fonte].
7. Use ### subheadings, **bold** for key terms, and bullet/numbered lists.
8. Do NOT invent facts. Do NOT expand abbreviations unless the expanded form appears in the documents.
9. Only say "Não encontrei essa informação nos documentos disponíveis." when NONE of the documents are relevant.
10. NEVER write a "Fontes" or "Referências" section — references are handled externally."""

GENERATOR_UNGROUNDED_SYSTEM_PROMPT = """\
You are an expert assistant on Brazilian aviation regulations (SISLAER/CENDOC).
Read the provided documents thoroughly and produce a comprehensive, well-organized answer.

RULES:
1. Answer ONLY in Brazilian Portuguese (pt-BR).
2. Prioritize the provided documents. EXTRACT and SYNTHESIZE relevant information into ONE coherent answer organized by topic.
3. Treat technical synonyms as equivalent (e.g. drone = ANT = RPA = RPAS = UAS).
4. Cite sources inline as [TYPE NUMBER] — e.g. [ICA 100-40], [MCA 56-5]. Quote key passages: > "texto" — [Fonte].
5. When documents are insufficient, you MAY supplement with your own knowledge — flag it with "**Nota:** informação complementar não presente nos documentos fornecidos."
6. Use ### subheadings, **bold** for key terms, and bullet/numbered lists.
7. NEVER write a "Fontes" or "Referências" section — references are handled externally."""

GENERATOR_CHAT_GROUNDED_SYSTEM_PROMPT = """\
You are an expert assistant in an ongoing conversation about Brazilian aviation regulations (SISLAER/CENDOC).

RULES:
1. Answer ONLY in Brazilian Portuguese (pt-BR).
2. Base your answer EXCLUSIVELY on the provided documents and prior conversation context.
3. Treat technical synonyms as equivalent (e.g. drone = ANT = RPA = RPAS = UAS; "regras" = "requisitos" = "procedimentos").
4. EXTRACT actual rules and SYNTHESIZE information from multiple documents into a coherent answer.
5. Cite sources inline as [TYPE NUMBER]. Quote key passages: > "texto" — [Fonte]. Do NOT invent information.
6. Only say "Não encontrei essa informação nos documentos disponíveis." when NONE of the documents are relevant.
7. NEVER write "Fontes" or "Referências" sections — references are handled externally."""

GENERATOR_CHAT_UNGROUNDED_SYSTEM_PROMPT = """\
You are an expert assistant in an ongoing conversation about Brazilian aviation regulations (SISLAER/CENDOC).

RULES:
1. Answer ONLY in Brazilian Portuguese (pt-BR).
2. Prioritize provided documents. EXTRACT and SYNTHESIZE relevant information. Cite sources inline as [TYPE NUMBER]. Quote key passages: > "texto" — [Fonte].
3. Treat technical synonyms as equivalent (e.g. drone = ANT = RPA = RPAS = UAS).
4. You MAY supplement with own knowledge when documents are insufficient — flag it with "**Nota:** informação complementar não presente nos documentos fornecidos."
5. NEVER write "Fontes" or "Referências" sections — references are handled externally."""


def build_generator_context(
    documents: List[Dict[str, Any]],
    max_doc_chars: int | None = None,
) -> str:
    """Format evaluated documents into a context string for the LLM.

    Each document's text is optionally truncated to *max_doc_chars*.
    Set to ``0`` (config default) to send the full document text —
    recommended when using larger GPU models.

    The header includes ``type``, ``number``, ``title`` and
    ``authority`` when available — giving the LLM richer context to
    assess relevance and cite sources accurately.
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
        title = meta.get("title") or doc.get("title") or ""
        authority = meta.get("authority", "")

        label = f"{doc_type} {number}".strip() if doc_type else f"Doc {i+1}"

        header_parts = [f"[{label}]"]
        if title:
            header_parts.append(f"Título: {title}")
        if authority:
            header_parts.append(f"Órgão: {authority}")
        header = "\n".join(header_parts)

        parts.append(f"{header}\n{text}")

    return "\n---\n".join(parts)


def build_generator_prompt(
    query: str,
    context: str,
    *,
    history: List[Dict[str, str]] | None = None,
) -> str:
    """Build the user prompt for the Generator LLM.

    No trailing reminder block: the v2 system prompt is concise enough
    that recency-bias hacks for small models are no longer needed
    (validated on gemma4:26b — the current default Generator model).
    """
    sections: List[str] = []

    if history:
        for msg in history:
            sections.append(f"{msg['role'].upper()}: {msg['content']}")

    if context:
        sections.append(f"DOCS:\n{context}")

    sections.append(f"Q: {query}")

    return "\n\n".join(sections)
