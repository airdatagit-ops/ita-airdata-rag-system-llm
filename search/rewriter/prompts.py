"""Prompt templates for the query rewriter stage.

System prompt is in English for better instruction-following by small
LLMs. Values for types and authorities are loaded dynamically from the
``FilterRegistry`` so the prompt stays in sync with the actual data.
"""

from __future__ import annotations

from search.shared.schemas import filter_registry


def build_system_prompt(max_queries: int) -> str:
    """Build the system prompt with current filter values."""
    types = ", ".join(filter_registry.prompt_types)
    authorities = ", ".join(filter_registry.prompt_authorities)

    return f"""\
You improve search queries about Brazilian aviation regulations.
Rules:
- Fix typos and grammar errors
- Expand abbreviations (RBAC → Regulamento Brasileiro de Aviação Civil)
- KEEP the original topic — do NOT add new subjects
- Do NOT change the meaning of the query
- Return at least one query
- Only add filters when the user explicitly mentions a document type or authority
- Leave filters and sorts as empty arrays when not needed

Valid filter fields and values:
  "metadata.type" (operator "eq"): {types}
  "metadata.authority" (operator "eq"): {authorities}

Example: "certficação de piltos" → text="certificação de pilotos na aviação civil", filters=[], sorts=[]
Example: "regras para drones" → text="regras para operação de drones", filters=[], sorts=[]
Example: "ICAs do DECEA sobre meteorologia" → text="instruções do DECEA sobre meteorologia aeronáutica", filters=[field=metadata.type value=ICA, field=metadata.authority value=DECEA]"""


def build_user_prompt(query: str) -> str:
    """Build user prompt with the query only."""
    return query
