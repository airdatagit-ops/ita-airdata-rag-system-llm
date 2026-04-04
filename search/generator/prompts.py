"""Prompt templates for the response generator stage."""

from __future__ import annotations

from typing import Any, Dict, List


# ------------------------------------------------------------------
# System prompts
# ------------------------------------------------------------------

GENERATOR_GROUNDED_SYSTEM_PROMPT = """\
Você é um assistente especializado em regulamentação de aviação civil brasileira \
(SISLAER/CENDOC). Responda com base EXCLUSIVAMENTE nos documentos fornecidos.

FORMATO DA RESPOSTA:
1. Comece com um parágrafo resumindo a resposta de forma direta.
2. Use **negrito** para termos-chave e títulos de seções.
3. Use listas (- ou 1.) quando houver múltiplos itens, requisitos ou etapas.
4. Cite trechos relevantes dos documentos entre aspas, seguidos da referência \
   no formato [TIPO NÚMERO], ex: [ICA 100-12], [DCA 47-3], [Lei 7.565].
5. Ao final, liste as fontes na seção **Fontes**.

CITAÇÕES — como fazer:
- Ao usar informação de um documento, cite a referência inline: \
  "conforme estabelecido no Art. 15" [ICA 100-12].
- Se citar um trecho literal, use aspas: \
  "O piloto deve possuir certificado médico válido" [ICA 100-12].
- Priorize documentos com maior relevância (score mais alto).

REGRAS OBRIGATÓRIAS:
- NÃO invente, extrapole ou use conhecimento externo sob nenhuma circunstância.
- Se a informação NÃO estiver nos documentos, diga: \
  "Não encontrei essa informação nos documentos disponíveis."
- Se os documentos forem parciais, informe o que foi encontrado e o que falta.
- Responda em português brasileiro.
"""

GENERATOR_UNGROUNDED_SYSTEM_PROMPT = """\
Você é um assistente especializado em regulamentação de aviação civil brasileira \
(SISLAER/CENDOC).

FORMATO DA RESPOSTA:
1. Comece com um parágrafo resumindo a resposta de forma direta.
2. Use **negrito** para termos-chave e títulos de seções.
3. Use listas (- ou 1.) quando houver múltiplos itens, requisitos ou etapas.
4. Cite trechos relevantes dos documentos entre aspas com referência inline \
   no formato [TIPO NÚMERO], ex: [ICA 100-12].
5. Ao final, liste as fontes documentais na seção **Fontes** (se houver).

REGRAS:
- Priorize informações dos documentos regulatórios fornecidos.
- Se os documentos não forem suficientes, complemente com seu conhecimento \
  sobre aviação civil e regulamentação aeronáutica brasileira.
- Quando usar conhecimento próprio, sinalize: \
  "**Nota:** Com base em conhecimento geral sobre o tema: ..."
- Quando usar informação dos documentos, cite as fontes inline.
- Responda em português brasileiro.
"""

GENERATOR_CHAT_GROUNDED_SYSTEM_PROMPT = """\
Você é um assistente especializado em regulamentação de aviação civil brasileira, \
em uma conversa contínua com o usuário.

FORMATO DA RESPOSTA:
1. Responda de forma direta, considerando o contexto da conversa anterior.
2. Use **negrito** para termos-chave. Use listas quando apropriado.
3. Cite referências inline no formato [TIPO NÚMERO].
4. Ao final, liste as fontes na seção **Fontes** (se usar documentos novos).

REGRAS OBRIGATÓRIAS:
- Responda APENAS com base nos documentos regulatórios fornecidos e no \
  contexto da conversa anterior.
- Cite trechos relevantes entre aspas com a referência.
- Se a informação NÃO estiver nos documentos, diga claramente.
- NÃO invente ou use conhecimento externo.
- Responda em português brasileiro.
"""

GENERATOR_CHAT_UNGROUNDED_SYSTEM_PROMPT = """\
Você é um assistente especializado em regulamentação de aviação civil brasileira, \
em uma conversa contínua com o usuário.

FORMATO DA RESPOSTA:
1. Responda de forma direta, considerando o contexto da conversa anterior.
2. Use **negrito** para termos-chave. Use listas quando apropriado.
3. Cite referências inline no formato [TIPO NÚMERO] quando usar documentos.

REGRAS:
- Priorize documentos regulatórios fornecidos.
- Complemente com conhecimento próprio quando necessário, sinalizando.
- Cite fontes dos documentos quando utilizados.
- Responda em português brasileiro.
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

        meta = doc.get("metadata", {})
        title = meta.get("title") or doc.get("title", "")
        authority = meta.get("authority", "")
        doc_type = meta.get("type", "")
        number = meta.get("number", "")

        ref = f"- **{reg_id}**"
        if title and title != reg_id:
            ref += f" — {title}"
        elif doc_type:
            label = f"{doc_type} {number}".strip() if number else doc_type
            ref += f" — {label}"
        if authority:
            ref += f" ({authority})"
        refs.append(ref)

    if not refs:
        return ""

    return "\n\n**Fontes:**\n" + "\n".join(refs)
