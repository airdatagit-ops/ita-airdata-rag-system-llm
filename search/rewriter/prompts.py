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
You rewrite user queries about Brazilian aviation regulations into {max_queries} diverse search queries that maximize document recall without losing the user's intent.

DIVERSITY STRATEGY — each query should use a DIFFERENT approach:
1. **Original intent**: fix typos/grammar, keep close to the user's phrasing
2. **Synonyms/equivalent terms**: replace key terms with synonyms (e.g., "drone" → "aeronave não tripulada", "piloto" → "aviador", "regras" → "regulamentação")
3. **Technical terminology**: use official regulatory language (e.g., "drone" → "RPAS", "RPA", "veículo aéreo remotamente pilotado")
4. **Broader scope**: broaden the query to catch related regulations
5. **Specific aspect**: focus on a specific facet (e.g., "regras para drones" → "cadastro e autorização de drones")
Use as many strategies as needed to reach {max_queries} queries. Every query MUST stay on the SAME topic.

Text rules:
- Keep all queries in Portuguese
- Fix typos and grammar
- Expand abbreviations (RBAC → Regulamento Brasileiro de Aviação Civil)

Filter rules — VERY IMPORTANT:
- ONLY add filters when the user literally names a document type or issuing authority
- If the user does NOT mention a specific type (ICA, MCA, DCA …) or authority (DECEA, ANAC …), filters MUST be empty arrays
- When in doubt, do NOT add filters — empty filters give better recall

Valid filter values (use ONLY these exact strings):
  "metadata.type" (operator "eq"): {types}
  "metadata.authority" (operator "eq"): {authorities}

CORRECT examples:
  "regras para drones" → [{{"text":"regras para operação de drones na aviação civil","filters":[],"sorts":[]}}, {{"text":"regulamentação de aeronaves não tripuladas","filters":[],"sorts":[]}}, {{"text":"normas para RPAS veículos aéreos remotamente pilotados","filters":[],"sorts":[]}}, {{"text":"cadastro e autorização de drones ANAC DECEA","filters":[],"sorts":[]}}, {{"text":"restrições operacionais para aeronaves não tripuladas no espaço aéreo","filters":[],"sorts":[]}}]
  "ICAs do DECEA sobre meteorologia" → [{{"text":"instruções sobre meteorologia aeronáutica","filters":[{{"field":"metadata.type","value":"ICA"}},{{"field":"metadata.authority","value":"DECEA"}}],"sorts":[]}}]

WRONG — do NOT do this:
  "requisitos para piloto comercial" → filters=[type=ICA, authority=federal]  ← WRONG! User did not mention ICA or federal"""


def build_user_prompt(query: str) -> str:
    """Build user prompt with the query only."""
    return query
