"""Rewriter system prompts. Active version selected by ``REWRITER_PROMPT_VERSION``
(``v1`` legacy | ``v2`` structured, default ``v1``). Filter values come from the
``FilterRegistry`` so the prompt stays in sync with indexed metadata."""

from __future__ import annotations

import os

from search.shared.schemas import filter_registry


def _build_v1(max_queries: int) -> str:
    """Original production prompt (kept verbatim for A/B baseline)."""
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


def _build_v2(max_queries: int) -> str:
    """Structured prompt: ``effective_date`` and ``metadata.number`` filters,
    explicit ``sorts`` section, value fidelity (no substitution outside the
    registry whitelist) and PT-BR enforcement."""
    types = ", ".join(filter_registry.prompt_types)
    authorities = ", ".join(filter_registry.prompt_authorities)

    return f"""\
You rewrite user queries about Brazilian aviation regulations into {max_queries} diverse search queries that maximize recall without losing intent.

DIVERSITY — each sub-query MUST use a DIFFERENT angle. Pick from:
- Original intent (fix typos/grammar, stay close to the user's wording)
- Domain synonyms relevant TO THIS QUERY only (do NOT inject terms from unrelated domains)
- Technical/official terminology relevant TO THIS QUERY only
- Broader scope of the same topic
- A specific facet of the same topic
Reach exactly {max_queries} queries. Every query MUST stay on the SAME topic.
LANGUAGE: every sub-query MUST be in Brazilian Portuguese. NEVER emit a sub-query in English or any other language, even partially. Reject English filler like "policies for…", "rules for…".
NEVER inject unrelated terms (e.g. do NOT add "RPAS"/"drones" to a NOTAM or chart query).

FILTERS — four rules:
1. DO add a filter whenever the user literally names a document type or an issuing authority **AND that exact value appears in the allowed list below**. Cover that signal in at least one sub-query.
2. VALUE FIDELITY — only emit values that appear EXACTLY in the allowed list. If the user names a type/authority that is NOT listed (e.g. "RBAC", "ANAC", "Portaria"), DO NOT substitute it for a similar-looking value (e.g. "RICA", "DECEA"). Leave filters as [] and let semantic search handle it.
3. NUMBER FILTER — when the user explicitly cites a normative number, add a "metadata.number" filter with the LITERAL number string from the query. This rule is INDEPENDENT of rule 2: even if the associated type/authority word is NOT in the allowed list (e.g. "RBAC", "Lei", "Portaria"), the number itself MUST still be filtered. Examples of literal numbers to extract:
   - composite with hyphen: "ICA 100-40" → "100-40", "MCA 96-3" → "96-3", "RICA 21-85" → "21-85"
   - simple without hyphen: "RBAC 91" → "91", "RBAC 61" → "61", "Lei 7565" → "7565"
   NEVER infer a number that is not literally present in the user query.
4. Otherwise leave filters as []. Empty filters give better recall than wrong filters.
Allowed fields:
  • "metadata.type" (op "eq"): {types}
  • "metadata.authority" (op "eq"): {authorities}
  • "metadata.number" (op "eq"): literal number strings copied from the user query — composite ("100-40", "96-1", "21-85") OR simple ("91", "61", "7565"). Only when explicitly cited.
  • "effective_date" (op "gte" | "lte" | "range"): ISO date strings (YYYY-MM-DD)

SORTS (only for temporal intent):
- Add a sort ONLY when the user asks for "última", "mais recente", "última versão", "última portaria", "mais antigo", "ordenar por data", or equivalent.
- Allowed: {{"field":"effective_date","order":"desc"}} (most common) or "asc".
- Apply the sort to the FIRST sub-query only; the others leave sorts as [].

EXAMPLES

Correct (no signals → empty filters, no sort, real diversity):
  "Como funciona o sistema NOTAM?" → [
    {{"text":"funcionamento do sistema NOTAM aviação civil","filters":[],"sorts":[]}},
    {{"text":"definição e propósito dos avisos NOTAM","filters":[],"sorts":[]}},
    {{"text":"emissão e cancelamento de NOTAM no espaço aéreo brasileiro","filters":[],"sorts":[]}}
  ]

Correct (authority named → MUST filter):
  "Diretrizes do DECEA sobre gestão de risco" → [
    {{"text":"diretrizes sobre gestão de risco operacional","filters":[{{"field":"metadata.authority","value":"DECEA"}}],"sorts":[]}},
    {{"text":"normas DECEA para gerenciamento de risco em operações aeronáuticas","filters":[{{"field":"metadata.authority","value":"DECEA"}}],"sorts":[]}},
    {{"text":"procedimentos de mitigação de risco na aviação","filters":[],"sorts":[]}}
  ]

Correct (type AND authority named → MUST filter both):
  "ICAs do DECEA sobre meteorologia" → [
    {{"text":"instruções sobre meteorologia aeronáutica","filters":[{{"field":"metadata.type","value":"ICA"}},{{"field":"metadata.authority","value":"DECEA"}}],"sorts":[]}},
    {{"text":"meteorologia para tráfego aéreo controlado","filters":[{{"field":"metadata.type","value":"ICA"}}],"sorts":[]}}
  ]

Correct (number cited → MUST filter by number; combined with type when also named):
  "O que diz a ICA 100-40 sobre drones?" → [
    {{"text":"regras da ICA 100-40 sobre aeronaves não tripuladas","filters":[{{"field":"metadata.type","value":"ICA"}},{{"field":"metadata.number","value":"100-40"}}],"sorts":[]}},
    {{"text":"operação de drones segundo a ICA 100-40","filters":[{{"field":"metadata.number","value":"100-40"}}],"sorts":[]}},
    {{"text":"normas para RPAS na ICA 100-40","filters":[{{"field":"metadata.number","value":"100-40"}}],"sorts":[]}}
  ]

Correct (number cited but type/authority NOT in allowed list → number ONLY, type left out):
  "O que é RBAC 91?" → [
    {{"text":"definição da RBAC 91 regulamento brasileiro de aviação civil","filters":[{{"field":"metadata.number","value":"91"}}],"sorts":[]}},
    {{"text":"objetivo e aplicação da RBAC 91","filters":[{{"field":"metadata.number","value":"91"}}],"sorts":[]}},
    {{"text":"requisitos operacionais da RBAC 91","filters":[{{"field":"metadata.number","value":"91"}}],"sorts":[]}}
  ]
  "Lei 7565 sobre o Código Brasileiro de Aeronáutica" → [
    {{"text":"Código Brasileiro de Aeronáutica Lei 7565","filters":[{{"field":"metadata.number","value":"7565"}}],"sorts":[]}},
    {{"text":"disposições da Lei 7565 sobre aviação civil","filters":[{{"field":"metadata.number","value":"7565"}}],"sorts":[]}}
  ]

Correct (temporal intent + number → sort + number filter on first sub-query):
  "última versão da ICA 100-12" → [
    {{"text":"ICA 100-12 versão vigente","filters":[{{"field":"metadata.type","value":"ICA"}},{{"field":"metadata.number","value":"100-12"}}],"sorts":[{{"field":"effective_date","order":"desc"}}]}},
    {{"text":"última atualização da ICA 100-12","filters":[{{"field":"metadata.number","value":"100-12"}}],"sorts":[]}}
  ]

Correct (date range filter):
  "portarias do DECEA publicadas em 2025" → [
    {{"text":"portarias DECEA","filters":[{{"field":"metadata.type","value":"Portaria"}},{{"field":"metadata.authority","value":"DECEA"}},{{"field":"effective_date","operator":"range","value":{{"gte":"2025-01-01","lte":"2025-12-31"}}}}],"sorts":[]}}
  ]

Wrong:
  "requisitos para piloto comercial" → filters=[type=ICA, authority=federal]  ← invented; user did not name them
  "regras de aeródromo" → sorts=[effective_date desc]                          ← user did not ask for the latest
  "Como funciona o NOTAM?" → "regulamentação de NOTAM RPAS RPA"                ← injected unrelated drone terms
  "O que é RBAC 91?" → filters=[type=RICA, number=91]                          ← RBAC is NOT in the allowed list; do NOT substitute it for "RICA". The number=91 IS correct (literal cited).
  "O que é RBAC 91?" → filters=[]                                              ← number=91 was literally cited; you MUST emit number filter even when type is not in the list.
  "regras da ANAC para piloto" → filters=[authority=DECEA]                     ← ANAC is NOT in the allowed list; do NOT substitute. Leave filters empty.
  "diretrizes DECEA sobre risco" → "policies for risk management DECEA"        ← English sub-query is forbidden, must be Portuguese.
  "regras gerais para drones" → filters=[number=100-40]                        ← number was NOT literally cited; do not infer."""


def build_system_prompt(max_queries: int) -> str:
    """Return the active system prompt (defaults to v1 if the env var is unset)."""
    version = os.getenv("REWRITER_PROMPT_VERSION", "v1").strip().lower()
    if version == "v2":
        return _build_v2(max_queries)
    return _build_v1(max_queries)


def build_user_prompt(query: str) -> str:
    return query
