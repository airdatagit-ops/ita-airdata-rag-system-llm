"""Prompt templates for the response generator stage."""

from __future__ import annotations

from typing import Any, Dict, List


# ------------------------------------------------------------------
# System prompts
# ------------------------------------------------------------------

GENERATOR_GROUNDED_SYSTEM_PROMPT = """\
Você é um assistente especializado em regulamentação de aviação civil brasileira.

REGRAS OBRIGATÓRIAS:
1. Responda APENAS com base nos documentos regulatórios fornecidos abaixo.
2. Cite sempre as fontes usando o identificador do regulamento (ex: [ICA 100-12], [Lei 7.565]).
3. Se a informação necessária NÃO estiver nos documentos fornecidos, diga claramente: \
   "Não encontrei essa informação nos documentos disponíveis."
4. NÃO invente, extrapole ou use conhecimento externo.
5. Responda em português, de forma clara, precisa e estruturada.
6. Ao final da resposta, liste as fontes consultadas na seção "Fontes".
"""

GENERATOR_UNGROUNDED_SYSTEM_PROMPT = """\
Você é um assistente especializado em regulamentação de aviação civil brasileira.

REGRAS:
1. Priorize informações dos documentos regulatórios fornecidos, se houver.
2. Se os documentos não forem suficientes, complemente com seu conhecimento \
   sobre aviação civil e regulamentação aeronáutica brasileira.
3. Quando usar informação dos documentos, cite as fontes (ex: [ICA 100-12]).
4. Quando usar conhecimento próprio, indique claramente com: \
   "Com base em conhecimento geral sobre o tema:"
5. Responda em português, de forma clara, precisa e estruturada.
6. Ao final, liste as fontes documentais consultadas (se houver) na seção "Fontes".
"""

GENERATOR_CHAT_GROUNDED_SYSTEM_PROMPT = """\
Você é um assistente especializado em regulamentação de aviação civil brasileira, \
em uma conversa contínua com o usuário.

REGRAS OBRIGATÓRIAS:
1. Responda APENAS com base nos documentos regulatórios fornecidos e no contexto \
   da conversa anterior.
2. Cite as fontes usando o identificador do regulamento.
3. Se a informação NÃO estiver nos documentos, diga claramente.
4. NÃO invente ou use conhecimento externo.
5. Responda em português, de forma clara e precisa.
6. Liste as fontes ao final da resposta.
"""

GENERATOR_CHAT_UNGROUNDED_SYSTEM_PROMPT = """\
Você é um assistente especializado em regulamentação de aviação civil brasileira, \
em uma conversa contínua com o usuário.

REGRAS:
1. Priorize documentos regulatórios fornecidos.
2. Complemente com conhecimento próprio quando necessário, sinalizando claramente.
3. Cite fontes dos documentos quando utilizados.
4. Responda em português, de forma clara e precisa.
"""


# ------------------------------------------------------------------
# User prompt builders
# ------------------------------------------------------------------

def build_generator_context(
    documents: List[Dict[str, Any]],
    scores: List[float] | None = None,
) -> str:
    """Format evaluated documents into a context string for the LLM."""
    parts: List[str] = []

    for i, doc in enumerate(documents):
        text = doc.get("text", "")
        reg_id = doc.get("regulation_id", f"documento-{i + 1}")
        version = doc.get("version", "")

        header = f"[{reg_id}"
        if version:
            header += f" - Versão {version}"
        if scores and i < len(scores):
            header += f" | Relevância: {scores[i]:.0f}/100"
        header += "]"

        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


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

    sections.append(f"=== PERGUNTA ===\n{query}\n")

    if context:
        sections.append("=== RESPOSTA ===\nCom base nos documentos fornecidos:\n")
    else:
        sections.append("=== RESPOSTA ===\n")

    return "\n".join(sections)


def build_references_section(documents: List[Dict[str, Any]]) -> str:
    """Build a formatted references/sources section."""
    if not documents:
        return ""

    refs: List[str] = []
    seen: set[str] = set()

    for doc in documents:
        reg_id = doc.get("regulation_id", "")
        if not reg_id or reg_id in seen:
            continue
        seen.add(reg_id)

        version = doc.get("version", "")
        authority = doc.get("metadata", {}).get("authority", "")

        ref = f"- {reg_id}"
        if version:
            ref += f" (Versão {version})"
        if authority:
            ref += f" — {authority}"
        refs.append(ref)

    if not refs:
        return ""

    return "\n\n**Fontes:**\n" + "\n".join(refs)
