"""Prompt templates for the query rewriter stage.

The schema block injected into the system prompt is generated at import
time from the Pydantic models so it stays in sync with the code.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from search.shared.schemas import (
    RewrittenQuery,
    SearchFilter,
    SearchSort,
    FilterOperator,
    SortOrder,
)

# ------------------------------------------------------------------
# Valid values for filter fields (kept in sync with rewriter.py)
# ------------------------------------------------------------------

VALID_FILTER_FIELDS: Dict[str, Dict[str, Any]] = {
    "metadata.type": {
        "description": "Tipo de documento normativo",
        "operator": "eq",
        "values": sorted([
            "ICA", "DCA", "MCA", "NSCA", "RCA", "PCA", "FCA",
            "TCA", "BCA", "BMA", "IMA", "RICA", "ROCA", "decreto",
        ]),
    },
    "metadata.authority": {
        "description": "Órgão emissor do documento",
        "operator": "eq",
        "values": sorted([
            "DECEA", "EMAER", "GABAER", "CENIPA", "DCTA", "COMGAP",
            "DIRSA", "CIMAER", "DIRAD", "DIRINFRA", "DEPENS",
        ]),
    },
    "effective_date": {
        "description": "Data de vigência do documento (formato ISO 8601)",
        "operator": "gte | lte",
        "values": "string ISO 8601 (ex: 2024-01-01)",
    },
}

VALID_FACET_TYPES = ["general", "temporal", "authority", "document_type"]


def _build_schema_block() -> str:
    """Generate a human-readable schema description from Pydantic models."""
    filter_schema = json.dumps(
        SearchFilter.model_json_schema(), indent=2, ensure_ascii=False,
    )
    sort_schema = json.dumps(
        SearchSort.model_json_schema(), indent=2, ensure_ascii=False,
    )
    query_schema = json.dumps(
        RewrittenQuery.model_json_schema(), indent=2, ensure_ascii=False,
    )

    fields_block = "\n".join(
        f'  - field: "{field}"\n'
        f'    Descrição: {info["description"]}\n'
        f'    Operadores: {info["operator"]}\n'
        f'    Valores: {info["values"]}'
        for field, info in VALID_FILTER_FIELDS.items()
    )

    return (
        f"=== SCHEMA DO OUTPUT (Pydantic) ===\n"
        f"Cada elemento do JSON array segue este schema:\n"
        f"{query_schema}\n\n"
        f"Schema de SearchFilter:\n{filter_schema}\n\n"
        f"Schema de SearchSort:\n{sort_schema}\n\n"
        f"=== CAMPOS DE FILTRO PERMITIDOS ===\n"
        f"Só use os campos abaixo. Qualquer outro será descartado.\n\n"
        f"{fields_block}\n\n"
        f"facet_type: um de {VALID_FACET_TYPES}"
    )


REWRITER_SCHEMA_BLOCK = _build_schema_block()

# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
Você é um módulo de reescrita de consultas para um sistema de busca vetorial em \
regulamentações de aviação civil brasileira (SISLAER/CENDOC).

TAREFA: Transformar a consulta do usuário em 1 a <<MAX_QUERIES>> sub-consultas \
otimizadas para busca semântica.

REGRAS OBRIGATÓRIAS:
1. Reescreva para melhorar clareza, corrigir erros e expandir siglas comuns \
   (ex: RBAC → Regulamento Brasileiro de Aviação Civil).
2. Se a consulta for complexa, decomponha em sub-consultas (uma por faceta).
3. Mantenha cada consulta concisa (máximo 500 caracteres).
4. Responda APENAS com JSON válido conforme o schema abaixo. \
   Sem markdown, sem texto antes ou depois.

FILTROS — regras estritas:
- SÓ gere filtros quando o usuário mencionar EXPLICITAMENTE um tipo ou órgão.
- Na dúvida, NÃO filtre. Deixe "filters": [] e confie na busca semântica.
- Use APENAS os campos e valores listados no schema abaixo.

PROIBIDO:
- Inventar campos ou valores que não estejam no schema.
- Colocar filtros quando o usuário NÃO especificou tipo/órgão/data.
- Gerar texto fora do JSON.

""" + REWRITER_SCHEMA_BLOCK


def build_system_prompt(max_queries: int) -> str:
    """Build system prompt with max_queries substituted."""
    return _SYSTEM_PROMPT_TEMPLATE.replace("<<MAX_QUERIES>>", str(max_queries))

_USER_PROMPT_TEMPLATE = """\
Consulta: "<<QUERY>>"

Responda com um JSON array seguindo o schema acima. Exemplos:

Consulta simples (sem filtros):
[{"text": "regras para operação de drones no espaço aéreo brasileiro", \
"filters": [], "sorts": [], "facet_type": "general"}]

Consulta com tipo explícito:
[{"text": "requisitos para certificação de pilotos", "filters": [], \
"sorts": [], "facet_type": "general"}, \
{"text": "requisitos para certificação de pilotos em ICAs", \
"filters": [{"field": "metadata.type", "operator": "eq", "value": "ICA"}], \
"sorts": [], "facet_type": "document_type"}]

Consulta com órgão explícito:
[{"text": "normas do DECEA sobre espaço aéreo", \
"filters": [{"field": "metadata.authority", "operator": "eq", "value": "DECEA"}], \
"sorts": [], "facet_type": "authority"}]

Agora reescreva a consulta acima:
"""


def build_user_prompt(query: str) -> str:
    """Build user prompt with query substituted."""
    return _USER_PROMPT_TEMPLATE.replace("<<QUERY>>", query)
