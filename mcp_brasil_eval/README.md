# mcp-brasil-eval

CLI Python autocontido que faz a ponte entre uma pergunta em linguagem
natural e o servidor [`mcp-brasil`](https://github.com/Mcp-Brasil/mcp-brasil)
(70 APIs publicas brasileiras), usando Claude (Anthropic) como agente que
escolhe e chama as tools.

Objetivo: poder testar a qualidade do `mcp-brasil` em perguntas do dominio
legislativo/regulatorio, para futura comparacao com o RAG normativo deste
repositorio. Sem comparacao automatica nesta versao.

Tudo dentro deste diretorio. Nao toca o restante do projeto.

## Pre-requisitos

- Python 3.11+
- Acesso a internet (a 1a execucao baixa `mcp-brasil` via `uvx`).
- Uma `ANTHROPIC_API_KEY`.

## Instalacao (3 passos)

```bash
cd mcp_brasil_eval
bash setup.sh
# edite .env e preencha ANTHROPIC_API_KEY
```

`setup.sh` cria `.venv/`, instala dependencias e copia `.env.example` para
`.env` se ainda nao existir.

## Uso

Sempre com o venv ativo:

```bash
source .venv/bin/activate
```

### One-shot

```bash
python ask.py "Liste publicacoes no DOU da ANAC nos ultimos 30 dias sobre RBAC 91"
```

### REPL

```bash
python ask.py
> Quais projetos de lei sobre aviacao civil tramitam na Camara em 2026?
> exit
```

Cada pergunta no REPL e independente (sem historico entre perguntas).

## Escopos de tools

Controla quais tools do `mcp-brasil` ficam disponiveis para o Claude.

| Flag | Tools enviadas ao modelo | Custo aprox. (Sonnet 4.5) | Quando usar |
|------|--------------------------|---------------------------|-------------|
| `--scope legislation` (default) | ~68: DOU, Camara, Senado, jurisprudencia, TCU + 5 meta | ~$0.05 / pergunta | Avaliacao alinhada com RAG normativo |
| `--scope smart` | 7 meta-tools (`search_tools`/`call_tool` proxiam as 312 nativas) | ~$0.05-0.15 / pergunta | Cobrir todas as features sem inflar payload |
| `--scope all` | ~317 (todas as features ativas no servidor) | ~$2-4 / pergunta, 10-30s/turno | Apenas debug / exploracao |

```bash
python ask.py --scope smart "Qual a Selic atual?"
python ask.py --scope all "Compare gastos com saude em SP e MG"
```

Detalhe tecnico: `legislation` e `all` rodam o `mcp-brasil` com
`MCP_BRASIL_TOOL_SEARCH=none` (todas as tools nativas listadas);
`smart` usa o default `bm25` do `mcp-brasil`, em que o servidor expoe
so 7 meta-tools e o proprio Claude descobre/invoca via `search_tools` +
`call_tool`.

## Outras flags

```
--model <id>           Modelo Anthropic (default: claude-sonnet-4-5-20250929)
--max-tool-iters N     Max iteracoes do tool-use loop (default 15)
--verbose              Mostra cada tool call no stderr (debug)
--list-tools           Lista todas as tools do servidor e sai
```

## Exemplos de perguntas (dominio legislativo/regulatorio)

```bash
python ask.py "Quais foram as ultimas Resolucoes da ANAC publicadas no DOU em 2026?"
python ask.py "Cite acordaos do TCU sobre concessao de aeroportos nos ultimos 3 anos"
python ask.py "Existe algum PL na Camara propondo alteracao no Codigo Brasileiro de Aeronautica?"
python ask.py --scope smart "Quais foram os ultimos atos do DECEA publicados no DOU?"
```

Importante: `mcp-brasil` nao tem fonte ANAC normativa direta (RBACs por
numero), mas o `diario_oficial` (DOU) cobre a publicacao oficial dos atos.
Resultados em perguntas muito especificas podem ser parciais.

## Como funciona

1. `ask.py` sobe o `mcp-brasil` via `uvx --from mcp-brasil python -m
   mcp_brasil.server` (stdio).
2. Faz `list_tools()` no servidor.
3. Filtra para o escopo escolhido.
4. Manda a pergunta para a API da Anthropic com as tools convertidas para
   o formato `{name, description, input_schema}`.
5. Loop: enquanto o modelo retorna `stop_reason=tool_use`, executa a tool
   via MCP (`session.call_tool`) e devolve o resultado como `tool_result`.
6. Quando o modelo termina (`stop_reason=end_turn`), imprime a resposta.

## Limitacoes conhecidas

- Sem cache de respostas. Sem historico multi-turn no REPL.
- Sem comparacao automatica com o RAG do repositorio.
- Datasets opt-in (`anac_rab`, `anac_vra`, etc.) nao sao habilitados por
  default. Para ativar, exporte `MCP_BRASIL_DATASETS=...` antes de rodar.
