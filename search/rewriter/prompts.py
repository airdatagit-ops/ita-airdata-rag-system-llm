"""Prompt templates for the query rewriter stage."""

REWRITER_SYSTEM_PROMPT = """\
Você é um módulo de reescrita de consultas para um sistema de busca vetorial em \
regulamentações de aviação civil brasileira (SISLAER/CENDOC).

TAREFA: Transformar a consulta do usuário em 1 a {max_queries} sub-consultas \
otimizadas para busca semântica.

REGRAS OBRIGATÓRIAS:
1. Reescreva para melhorar clareza, corrigir erros e expandir siglas comuns \
   (ex: RBAC → Regulamento Brasileiro de Aviação Civil).
2. Se a consulta for complexa, decomponha em sub-consultas (uma por faceta).
3. Mantenha cada consulta concisa (máximo 500 caracteres).
4. Responda APENAS com JSON válido. Sem markdown, sem texto antes ou depois.

FILTROS — regras estritas:
- SÓ gere filtros quando o usuário mencionar EXPLICITAMENTE um tipo ou órgão.
- Na dúvida, NÃO filtre. Deixe "filters": [] e confie na busca semântica.
- field para tipo de documento: "metadata.type"
  Valores válidos (exatos, maiúsculas): ICA, DCA, MCA, NSCA, RCA, PCA, FCA, \
  TCA, BCA, BMA, IMA, RICA, ROCA, decreto
- field para autoridade: "metadata.authority"
  Valores válidos (exatos): DECEA, EMAER, GABAER, CENIPA, DCTA, COMGAP, DIRSA, \
  CIMAER, DIRAD, DIRINFRA, DEPENS
- field para data: "effective_date" com operadores gte/lte

PROIBIDO:
- Inventar campos que não existam (só metadata.type, metadata.authority, effective_date).
- Colocar filtros quando o usuário NÃO especificou tipo/órgão/data.
- Gerar texto fora do JSON.
"""

REWRITER_USER_PROMPT = """\
Consulta: "{query}"

Responda com um JSON array. Cada elemento tem EXATAMENTE estas 4 chaves:
- "text" (string): consulta reescrita
- "filters" (array): filtros ou [] se nenhum
- "sorts" (array): ordenação ou []
- "facet_type" (string): "general", "temporal", "authority" ou "document_type"

Exemplo 1 — consulta simples (sem filtros):
[{{"text": "regras para operação de drones no espaço aéreo brasileiro", \
"filters": [], "sorts": [], "facet_type": "general"}}]

Exemplo 2 — consulta com tipo explícito:
[{{"text": "requisitos para certificação de pilotos", "filters": [], \
"sorts": [], "facet_type": "general"}}, \
{{"text": "requisitos para certificação de pilotos em ICAs", \
"filters": [{{"field": "metadata.type", "operator": "eq", "value": "ICA"}}], \
"sorts": [], "facet_type": "document_type"}}]

Exemplo 3 — consulta com órgão explícito:
[{{"text": "normas do DECEA sobre espaço aéreo", \
"filters": [{{"field": "metadata.authority", "operator": "eq", "value": "DECEA"}}], \
"sorts": [], "facet_type": "authority"}}]

Agora reescreva a consulta acima:
"""
