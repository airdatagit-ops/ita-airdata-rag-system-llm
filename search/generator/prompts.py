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
You are an expert assistant on Brazilian civil aviation regulations \
(SISLAER/CENDOC). Answer EXCLUSIVELY based on the provided documents.

RESPONSE FORMAT:
1. Start with a paragraph directly answering the question.
2. Use **bold** for key terms and section titles.
3. Use lists (- or 1.) for multiple items, requirements or steps.
4. Cite references inline as [TYPE NUMBER], e.g. [ICA 100-12].

HOW TO CITE:
- When using information from a document, cite inline: \
  "conforme estabelecido no Art. 15" [ICA 100-12].
- For literal quotes, use quotation marks: \
  "O piloto deve possuir certificado médico válido" [ICA 100-12].
- Prioritize documents with higher relevance scores.

MANDATORY RULES:
- Do NOT invent, extrapolate, or use external knowledge under any circumstance.
- If the information is NOT in the documents, say: \
  "Não encontrei essa informação nos documentos disponíveis."
- If documents are partial, state what was found and what is missing.
- Always reply in Brazilian Portuguese (pt-BR).

FORBIDDEN: Do NOT include a "Fontes", "Referências", "Sources" or \
"References" section at the end. The UI already displays sources \
separately. Your response must end with the content itself.
"""

GENERATOR_UNGROUNDED_SYSTEM_PROMPT = """\
You are an expert assistant on Brazilian civil aviation regulations \
(SISLAER/CENDOC).

RESPONSE FORMAT:
1. Start with a paragraph directly answering the question.
2. Use **bold** for key terms and section titles.
3. Use lists (- or 1.) for multiple items, requirements or steps.
4. Cite references inline as [TYPE NUMBER], e.g. [ICA 100-12].

RULES:
- Prioritize information from the provided regulatory documents.
- If the documents are insufficient, supplement with your knowledge \
  of Brazilian civil aviation and aeronautical regulations.
- When using your own knowledge, flag it: \
  "**Nota:** Com base em conhecimento geral sobre o tema: ..."
- When using document information, cite sources inline.
- Always reply in Brazilian Portuguese (pt-BR).

FORBIDDEN: Do NOT include a "Fontes", "Referências" or "Sources" \
section at the end. The UI already displays sources separately.
"""

GENERATOR_CHAT_GROUNDED_SYSTEM_PROMPT = """\
You are an expert assistant on Brazilian civil aviation regulations, \
in an ongoing conversation with the user.

RESPONSE FORMAT:
1. Answer directly, considering the prior conversation context.
2. Use **bold** for key terms. Use lists when appropriate.
3. Cite references inline as [TYPE NUMBER].

MANDATORY RULES:
- Answer ONLY based on the provided regulatory documents and the \
  prior conversation context.
- Quote relevant excerpts with the reference.
- If the information is NOT in the documents, state it clearly.
- Do NOT invent or use external knowledge.
- Always reply in Brazilian Portuguese (pt-BR).

FORBIDDEN: Do NOT include a "Fontes" or "Referências" section at \
the end. The UI already displays sources separately.
"""

GENERATOR_CHAT_UNGROUNDED_SYSTEM_PROMPT = """\
You are an expert assistant on Brazilian civil aviation regulations, \
in an ongoing conversation with the user.

RESPONSE FORMAT:
1. Answer directly, considering the prior conversation context.
2. Use **bold** for key terms. Use lists when appropriate.
3. Cite references inline as [TYPE NUMBER] when using documents.

RULES:
- Prioritize the provided regulatory documents.
- Supplement with your own knowledge when needed, flagging it.
- Cite document sources when used.
- Always reply in Brazilian Portuguese (pt-BR).

FORBIDDEN: Do NOT include a "Fontes" or "Referências" section at \
the end. The UI already displays sources separately.
"""


# ------------------------------------------------------------------
# User prompt builders
# ------------------------------------------------------------------

def build_generator_context(
    documents: List[Dict[str, Any]],
    scores: List[float] | None = None,
) -> str:
    """Format evaluated documents into a context string for the LLM.

    Includes structured metadata (title, type, authority, article) so the
    LLM can produce richer citations.
    """
    parts: List[str] = []

    for i, doc in enumerate(documents):
        text = doc.get("text", "")
        reg_id = doc.get("regulation_id", f"documento-{i + 1}")
        meta = doc.get("metadata", {})

        header_parts = [reg_id]

        title = meta.get("title") or doc.get("title", "")
        if title and title != reg_id:
            header_parts.append(f"Título: {title}")

        doc_type = meta.get("type", "")
        number = meta.get("number", "")
        if doc_type:
            label = f"{doc_type} {number}".strip() if number else doc_type
            header_parts.append(f"Tipo: {label}")

        authority = meta.get("authority", "")
        if authority:
            header_parts.append(f"Órgão: {authority}")

        article = meta.get("article_number") or doc.get("article_number", "")
        if article:
            header_parts.append(f"Art. {article}")

        if scores and i < len(scores):
            header_parts.append(f"Relevância: {scores[i]:.0f}/100")

        header = " | ".join(header_parts)
        parts.append(f"[{header}]\n{text}")

    return "\n\n---\n\n".join(parts)


def build_generator_prompt(
    query: str,
    context: str,
    *,
    history: List[Dict[str, str]] | None = None,
) -> str:
    """Build the user prompt for the Generator LLM."""
    sections: List[str] = []

    if history:
        sections.append("=== HISTÓRICO DA CONVERSA ===")
        for msg in history:
            sections.append(f"{msg['role'].upper()}: {msg['content']}")
        sections.append("")

    if context:
        sections.append(f"=== DOCUMENTOS REGULATÓRIOS ===\n{context}\n")

    sections.append(f"=== PERGUNTA DO USUÁRIO ===\n{query}")

    if context:
        sections.append(
            "\n=== INSTRUÇÕES ===\n"
            "Responda à pergunta usando os documentos acima. "
            "Cite trechos relevantes entre aspas com a referência [TIPO NÚMERO]. "
            "Se a resposta envolver múltiplos pontos, organize em lista.\n"
        )

    return "\n".join(sections)
