# Documentação Completa - Aviation RAG System

> Documentação detalhada de todo o sistema Aviation RAG. Para um guia rápido, consulte o [QUICKSTART.md](QUICKSTART.md). Para uma introdução, consulte o [START_HERE.md](START_HERE.md).

---

## Índice

1. [Visão Geral do Sistema](#1-visão-geral-do-sistema)
2. [Pré-requisitos e Dependências Externas](#2-pré-requisitos-e-dependências-externas)
3. [Configuração Central (.env)](#3-configuração-central-env)
4. [Banco Vetorial (Qdrant)](#4-banco-vetorial-qdrant)
5. [Modelo de Embeddings (Legal-BERTimbau)](#5-modelo-de-embeddings-legal-bertimbau)
6. [LLM (Ollama)](#6-llm-ollama)
7. [Extração de Documentos Normativos](#7-extração-de-documentos-normativos)
8. [Pipeline de Ingestão](#8-pipeline-de-ingestão)
9. [API RAG (Backend)](#9-api-rag-backend)
10. [Interface Web (Frontend)](#10-interface-web-frontend)
11. [Scripts Utilitários](#11-scripts-utilitários)
12. [Deploy em Produção (systemctl + Nginx)](#12-deploy-em-produção-systemctl--nginx)
13. [Resolução de Problemas](#13-resolução-de-problemas)

---

## 1. Visão Geral do Sistema

O Aviation RAG System é uma plataforma que combina:

- **Busca semântica vetorial** — Encontra trechos de documentos similares à pergunta do usuário usando embeddings
- **Geração aumentada por recuperação (RAG)** — Usa os trechos recuperados como contexto para um LLM gerar respostas fundamentadas
- **Chat conversacional** — Mantém histórico de conversa por sessão, com streaming em tempo real

### Fluxo de uma consulta RAG:

```
Pergunta do usuário
        │
        ▼
  1. Embedding da pergunta (Legal-BERTimbau)
        │
        ▼
  2. Busca vetorial no Qdrant (top-K documentos similares)
        │
        ▼
  3. Construção do prompt (contexto + pergunta + histórico)
        │
        ▼
  4. Geração de resposta pelo LLM (Ollama)
        │
        ▼
  Resposta com citações de fontes
```

### Componentes do sistema:

| Componente | Tecnologia | Localização |
|-----------|------------|-------------|
| Configuração central | Pydantic Settings | `config.py` |
| API RAG | FastAPI | `api/server.py` |
| Busca vetorial | Qdrant Client | `search/vector_search.py` |
| Pipeline RAG | VectorSearch + LLM | `search/rag.py` |
| Embeddings | Legal-BERTimbau (sentence-transformers) | `models/embeddings.py` |
| LLM | Ollama (llama3, phi3, etc.) | `models/llm.py` |
| Banco vetorial | Qdrant | `database/qdrant_manager.py` |
| Scraper DECEA | Selenium + BeautifulSoup | `parsers/decea_scraper.py` |
| Scraper LexML | Requests + BeautifulSoup | `parsers/lexml_scraper.py` |
| Ingestão | Chunking + Embedding + Upload | `pipeline/ingestion.py` |
| Interface Web | FastAPI + Jinja2 | `web/main.py` |

---

## 2. Pré-requisitos e Dependências Externas

### 2.1. Serviços que precisam estar rodando

| Serviço | Porta Padrão | Instalação |
|---------|-------------|------------|
| **Qdrant** | 6333 | Docker: `docker run -d -p 6333:6333 qdrant/qdrant` |
| **Ollama** | 11434 | `curl -fsSL https://ollama.com/install.sh \| sh` |

### 2.2. Dependências Python do sistema principal

O arquivo `requirements.txt` na raiz contém todas as dependências. As principais são:

| Biblioteca | Função |
|-----------|--------|
| `fastapi` | Framework web da API |
| `uvicorn` | Servidor ASGI |
| `qdrant-client` | Cliente Python para Qdrant |
| `sentence-transformers` | Carregamento do modelo de embeddings |
| `torch` | PyTorch (backend do modelo de embeddings) |
| `ollama` | Cliente Python para Ollama |
| `beautifulsoup4` | Parsing HTML dos scrapers |
| `selenium` | Automação de navegador (scraper DECEA) |
| `requests` | Requisições HTTP |
| `pydantic`, `pydantic-settings` | Validação de dados e configurações |
| `loguru` | Logging estruturado |
| `slowapi` | Rate limiting na API |

### 2.3. Dependências Python da interface web

O `web/requirements.txt` é um conjunto menor e independente. Veja a documentação em `web/docs/`.

### 2.4. Instalação

```bash
cd aviation-rag-system/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Configuração Central (.env)

Toda a configuração do sistema é centralizada no arquivo `.env` na raiz do projeto, carregada pela classe `Settings` em `config.py`.

### Como configurar:

```bash
cp env.example .env
# Edite o .env
```

### Referência completa de variáveis:

#### Segurança da API

| Variável | Tipo | Padrão | Descrição |
|----------|------|--------|-----------|
| `API_KEY` | string | — | **Obrigatório.** Chave de autenticação. Gere com: `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `CORS_ORIGINS` | string | `http://localhost:3000,http://localhost:8080` | Origens permitidas (separadas por vírgula) |
| `RATE_LIMIT` | int | `100` | Limite de requisições por minuto |

#### Qdrant

| Variável | Tipo | Padrão | Descrição |
|----------|------|--------|-----------|
| `QDRANT_HOST` | string | `localhost` | Host do servidor Qdrant |
| `QDRANT_PORT` | int | `6333` | Porta do Qdrant |
| `QDRANT_COLLECTION_NAME` | string | `aviation_regulations` | Nome da coleção |
| `QDRANT_API_KEY` | string | — | Chave API (apenas para Qdrant Cloud) |

#### Ollama / LLM

| Variável | Tipo | Padrão | Descrição |
|----------|------|--------|-----------|
| `OLLAMA_HOST` | string | `http://localhost:11434` | URL do servidor Ollama |
| `OLLAMA_MODEL` | string | `llama3.2:3b` | Modelo LLM padrão |
| `LLM_TEMPERATURE` | float | `0.3` | Temperatura de geração (0=determinístico, 2=criativo) |
| `LLM_TOP_P` | float | `0.9` | Nucleus sampling |
| `LLM_MAX_TOKENS` | int | `500` | Máximo de tokens por resposta |

#### Modelo de Embeddings

| Variável | Tipo | Padrão | Descrição |
|----------|------|--------|-----------|
| `EMBEDDING_MODEL` | string | `rufimelo/Legal-BERTimbau-sts-large-ma-v3` | Modelo HuggingFace |
| `EMBEDDING_BATCH_SIZE` | int | `32` | Batch size para encoding |
| `EMBEDDING_MAX_LENGTH` | int | `512` | Comprimento máximo de sequência |
| `EMBEDDING_DIMENSION` | int | `1024` | Dimensão dos vetores |

#### Busca

| Variável | Tipo | Padrão | Descrição |
|----------|------|--------|-----------|
| `SEARCH_TOP_K` | int | `5` | Número de resultados retornados |
| `SEARCH_SCORE_THRESHOLD` | float | `0.3` | Score mínimo de similaridade (0-1) |
| `HNSW_M` | int | `16` | Parâmetro M do índice HNSW |
| `HNSW_EF_CONSTRUCT` | int | `100` | Parâmetro ef_construct do HNSW |
| `HNSW_EF_SEARCH` | int | `64` | Parâmetro ef para busca no HNSW |

#### Chunking

| Variável | Tipo | Padrão | Descrição |
|----------|------|--------|-----------|
| `CHUNK_MAX_TOKENS` | int | `512` | Máximo de tokens por chunk |
| `CHUNK_OVERLAP` | int | `50` | Overlap entre chunks (tokens) |

#### Servidor da API

| Variável | Tipo | Padrão | Descrição |
|----------|------|--------|-----------|
| `API_HOST` | string | `127.0.0.1` | Host do servidor API |
| `API_PORT` | int | `8083` | Porta do servidor API |
| `RELOAD` | bool | `true` | Hot-reload em desenvolvimento |

#### LexML

| Variável | Tipo | Padrão | Descrição |
|----------|------|--------|-----------|
| `LEXML_API_URL` | string | `https://www.lexml.gov.br/sru` | URL da API LexML |
| `LEXML_KEYWORDS` | string | `aviação,aeronave,ANAC,...` | Palavras-chave para busca |

---

## 4. Banco Vetorial (Qdrant)

O Qdrant é o banco de dados vetorial que armazena os embeddings dos documentos normativos. Ele permite buscas por similaridade semântica e filtros temporais.

### 4.1. Inicialização

O Qdrant precisa estar rodando antes de qualquer operação. A forma mais simples é via Docker:

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 4.2. Criação da coleção

O script `setup_qdrant.py` cria a coleção com a configuração correta:

```bash
python -m scripts.setup_qdrant
```

Isso cria a coleção `aviation_regulations` com:
- Vetores de **1024 dimensões** (dimensão do Legal-BERTimbau)
- Distância **Cosine**
- Índice HNSW otimizado (M=16, ef_construct=100)
- Índices de payload para: `regulation_id`, `effective_date`, `expiry_date`, `source`, `doc_type`

### 4.3. Operações do QdrantManager

A classe `QdrantManager` (`database/qdrant_manager.py`) encapsula todas as operações:

| Método | Descrição |
|--------|-----------|
| `create_collection()` | Cria a coleção com índices |
| `upsert_points()` | Insere/atualiza vetores |
| `search()` | Busca por similaridade |
| `search_temporal()` | Busca com filtro de data |
| `get_collection_info()` | Estatísticas da coleção |
| `delete_collection()` | Remove a coleção |

### 4.4. Inspeção

Para inspecionar os dados no Qdrant:

```bash
python -m scripts.inspect_qdrant
```

Este script mostra:
- Informações da coleção (total de vetores, status)
- Estrutura dos campos (payload) de cada registro
- Exemplos de dados armazenados

### 4.5. Reset completo

Para apagar todos os dados e recriar a coleção:

```bash
python -m scripts.reset_database --confirm
```

Opções adicionais:

```bash
# Resetar e re-ingerir apenas documentos DECEA
python -m scripts.reset_database --confirm --only-decea

# Resetar e re-ingerir apenas documentos LexML
python -m scripts.reset_database --confirm --only-lexml

# Resetar sem re-ingerir (apenas limpa o banco)
python -m scripts.reset_database --confirm --skip-ingest

# Limpar também o rastreador de documentos
python -m scripts.reset_database --confirm --clear-tracker
```

---

## 5. Modelo de Embeddings (Legal-BERTimbau)

O sistema usa o modelo **rufimelo/Legal-BERTimbau-sts-large-ma-v3**, um modelo de sentence-transformers treinado especificamente para textos jurídicos em português brasileiro.

### Características:

| Propriedade | Valor |
|-------------|-------|
| Modelo base | BERTimbau Large |
| Treinamento | Textos jurídicos em PT-BR |
| Dimensão dos vetores | 1024 |
| Max sequence length | 512 tokens |
| Distância | Cosine similarity |

### Como funciona:

A classe `EmbeddingModel` (`models/embeddings.py`) carrega o modelo na inicialização e oferece:

```python
from models.embeddings import EmbeddingModel

model = EmbeddingModel()

# Encoding de um texto
vector = model.encode("Quais são os requisitos para pilotos?")
# Retorna: numpy array de shape (1024,)

# Encoding em batch
vectors = model.encode(["texto 1", "texto 2", "texto 3"])
# Retorna: numpy array de shape (3, 1024)
```

### GPU vs CPU:

- O modelo detecta automaticamente se CUDA está disponível
- Com **GPU**: encoding rápido (~100 textos/segundo)
- Com **CPU**: encoding lento (~5 textos/segundo), mas funcional

O cache do modelo é armazenado em `models_cache/`.

---

## 6. LLM (Ollama)

O sistema usa **Ollama** para rodar modelos de linguagem localmente. O Ollama gerencia o download, carregamento e execução dos modelos.

### 6.1. Instalação e configuração

```bash
# Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Baixar modelos
ollama pull llama3.2:3b    # Modelo leve (3B parâmetros)
ollama pull llama3.1:8b    # Modelo médio (8B parâmetros)
```

### 6.2. Modelos suportados

Qualquer modelo disponível no Ollama funciona. O modelo padrão é configurado em `OLLAMA_MODEL` no `.env`. Modelos testados:

| Modelo | Tamanho | Observação |
|--------|---------|------------|
| `llama3.2:3b` | ~2GB | Rápido, bom para testes |
| `llama3.1:8b` | ~4.7GB | Melhor qualidade, mais lento |
| `phi3:3.8b` | ~2.3GB | Alternativa leve da Microsoft |

### 6.3. Troca de modelo em tempo real

A API permite trocar o modelo LLM em tempo de execução, sem reiniciar o servidor:

```bash
curl -X POST -H "X-API-Key: SUA_CHAVE" -H "Content-Type: application/json" \
  -d '{"model_name": "llama3.1:8b"}' \
  http://127.0.0.1:8083/api/models/change
```

Na interface web, há um dropdown na sidebar do chat para trocar o modelo.

### 6.4. Classe LlamaModel

A classe `LlamaModel` (`models/llm.py`) oferece:

| Método | Descrição |
|--------|-----------|
| `generate()` | Geração de texto com prompt simples (suporta streaming) |
| `generate_with_context()` | Geração RAG (prompt + contexto de documentos) |
| `chat()` | Chat com histórico de mensagens (suporta streaming) |

---

## 7. Extração de Documentos Normativos

O sistema extrai documentos de duas fontes principais:

### 7.1. DECEA (Instruções de Comando da Aeronáutica)

**Scraper:** `parsers/decea_scraper.py`
**Script de ingestão:** `scripts/ingest_decea.py`

O scraper DECEA usa **Selenium** para acessar o portal de publicações do DECEA (`publicacoes.decea.mil.br`), que renderiza conteúdo via JavaScript. Ele:

1. Navega pelo índice de publicações
2. Filtra por tipo de documento (ICA, MCA, PCA, etc.)
3. Baixa os PDFs originais
4. Extrai o texto dos PDFs
5. Salva como JSON em `data/decea/`

**Como executar:**

```bash
# Ingerir ICAs (Instruções de Comando da Aeronáutica)
python -m scripts.ingest_decea --doc-types ICA --limit 50

# Ingerir ICAs e MCAs
python -m scripts.ingest_decea --doc-types ICA,MCA --limit 100

# Ingerir documentos específicos por slug
python -m scripts.ingest_decea --slugs ICA-63-47,ICA-100-12,ICA-100-37

# Pular download e usar JSONs existentes
python -m scripts.ingest_decea --skip-download

# Sem extrair texto dos PDFs (usar apenas a descrição)
python -m scripts.ingest_decea --doc-types ICA --no-text
```

**Parâmetros disponíveis:**

| Parâmetro | Descrição |
|-----------|-----------|
| `--doc-types` | Tipos de documento separados por vírgula (ICA, MCA, PCA, DCA, TCA, CIRCEA, NSCA) |
| `--slugs` | Slugs específicos separados por vírgula |
| `--keywords` | Palavras-chave para filtro |
| `--limit` | Máximo de documentos |
| `--download-dir` | Diretório de saída (padrão: `./data/decea`) |
| `--skip-download` | Usar apenas JSONs já existentes |
| `--no-text` | Não extrair texto dos PDFs |

**Requisito:** Chrome/Chromium instalado (para Selenium).

### 7.2. LexML (Legislação Federal)

**Scraper:** `parsers/lexml_scraper.py`
**Script de ingestão:** `scripts/ingest_lexml.py`

O scraper LexML usa **web scraping** (requests + BeautifulSoup) para buscar documentos no portal LexML Brasil. Ele:

1. Busca documentos por palavras-chave na interface web do LexML
2. Extrai metadados (título, URN, tipo, data, autoria)
3. Baixa o conteúdo textual dos documentos via normas.leg.br
4. Salva como JSON em `data/lexml/`
5. Registra no rastreador para evitar re-downloads futuros

**Como executar:**

```bash
# Ingerir com palavras-chave padrão (definidas no .env)
python -m scripts.ingest_lexml --limit 100

# Ingerir com palavras-chave específicas
python -m scripts.ingest_lexml --keywords "aviação,ANAC,aeroporto" --limit 50

# Forçar re-download de documentos já baixados
python -m scripts.ingest_lexml --force-download --limit 50

# Limpar rastreador e baixar tudo de novo
python -m scripts.ingest_lexml --clear-tracker --limit 100

# Pular download e ingerir apenas JSONs existentes
python -m scripts.ingest_lexml --skip-download
```

**Parâmetros disponíveis:**

| Parâmetro | Descrição |
|-----------|-----------|
| `--keywords` | Palavras-chave separadas por vírgula |
| `--limit` | Máximo de documentos |
| `--download-dir` | Diretório de saída (padrão: `./data/lexml`) |
| `--skip-download` | Usar apenas JSONs já existentes |
| `--force-download` | Reebaixar mesmo se já existir |
| `--clear-tracker` | Limpar rastreador antes de iniciar |

### 7.3. PDFs Locais

**Script:** `scripts/ingest_pdfs.py`

Para ingerir PDFs que já estejam em um diretório local:

```bash
# Ingerir PDFs de um diretório
python -m scripts.ingest_pdfs --source ./meus-pdfs/

# Buscar recursivamente em subdiretórios
python -m scripts.ingest_pdfs --source ./meus-pdfs/ --recursive
```

### 7.4. Rastreamento de Documentos

O `DocumentTracker` (`parsers/document_tracker.py`) mantém um registro em `data/document_tracker.json` com:

- URNs de documentos já baixados
- URLs já visitadas
- Hashes de conteúdo (para detectar duplicatas)
- Timestamps de download

Isso evita re-downloads desnecessários em execuções subsequentes.

---

## 8. Pipeline de Ingestão

O pipeline de ingestão (`pipeline/ingestion.py`) é responsável por processar documentos e armazená-los no Qdrant.

### Etapas do pipeline:

```
Documento JSON/XML/PDF
        │
        ▼
  1. Parsing (extrair texto e metadados)
        │
        ▼
  2. Chunking (dividir em trechos de ~512 tokens)
        │    - ArticleChunker: para legislação (divide por artigos)
        │    - ICAChunker: para ICAs (divide por seções/capítulos)
        ▼
  3. Embedding (gerar vetor de 1024 dimensões para cada chunk)
        │
        ▼
  4. Upload (upsert no Qdrant com metadados)
```

### Chunkers disponíveis:

| Chunker | Uso | Estratégia |
|---------|-----|-----------|
| `ArticleChunker` | Legislação (LexML) | Divide por artigos, parágrafos e incisos |
| `ICAChunker` | Documentos DECEA | Divide por seções, capítulos e títulos |

### Metadados armazenados por chunk:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `text` | string | Texto do trecho |
| `regulation_id` | string | ID da regulamentação (ex: "RBAC 61", "ICA-100-12") |
| `source` | string | Fonte (lexml, decea) |
| `doc_type` | string | Tipo de documento |
| `effective_date` | datetime | Data de início da vigência |
| `expiry_date` | datetime | Data de fim da vigência |
| `version` | string | Versão do documento |
| `metadata` | dict | Metadados adicionais |

---

## 9. API RAG (Backend)

A API RAG (`api/server.py`) é o coração do sistema. É um servidor FastAPI que expõe todos os endpoints de busca, chat e gerenciamento.

### 9.1. Endpoints

#### Busca e RAG

| Método | Endpoint | Descrição | Autenticação |
|--------|----------|-----------|-------------|
| POST | `/api/search-regulations` | Busca RAG completa (busca + LLM) | API Key |
| POST | `/api/vector-search` | Busca vetorial pura (sem LLM) | API Key |

#### Chat

| Método | Endpoint | Descrição | Autenticação |
|--------|----------|-----------|-------------|
| POST | `/api/chat` | Chat com resposta completa | API Key |
| POST | `/api/chat/stream` | Chat com streaming SSE | API Key |

#### Sessões

| Método | Endpoint | Descrição | Autenticação |
|--------|----------|-----------|-------------|
| GET | `/api/chat/session/{id}` | Info de uma sessão | API Key |
| DELETE | `/api/chat/session/{id}` | Deletar sessão | API Key |
| POST | `/api/chat/session/{id}/clear` | Limpar histórico da sessão | API Key |
| GET | `/api/chat/sessions/stats` | Estatísticas de sessões | API Key |

#### Modelos

| Método | Endpoint | Descrição | Autenticação |
|--------|----------|-----------|-------------|
| GET | `/api/models` | Listar modelos disponíveis | API Key |
| POST | `/api/models/change` | Trocar modelo ativo | API Key |

#### Sistema

| Método | Endpoint | Descrição | Autenticação |
|--------|----------|-----------|-------------|
| GET | `/` | Info básica da API | — |
| GET | `/health` | Health check | — |
| GET | `/stats` | Estatísticas do Qdrant + documentos | API Key |

### 9.2. Autenticação

Todas as requisições autenticadas necessitam do header:

```
X-API-Key: sua-chave-aqui
```

A chave é validada contra `API_KEY` do `.env` pelo middleware em `api/auth.py`.

### 9.3. Rate Limiting

A API utiliza `slowapi` para limitar requisições. O limite padrão é `100 requisições/minuto` por IP, configurável via `RATE_LIMIT` no `.env`.

### 9.4. Gerenciador de Sessões

O `SessionManager` (`api/session_manager.py`) gerencia sessões de chat em memória:

- Sessões com TTL (expiram automaticamente)
- Thread-safe
- Histórico de mensagens com limite configurável (50 mensagens por sessão)
- Cleanup automático em background
- Context window configurável (quantas mensagens anteriores enviar ao LLM)

### 9.5. Como executar a API

```bash
# Na raiz do projeto, com o venv ativado:
python -m api.server

# Ou com uvicorn diretamente:
uvicorn api.server:app --host 127.0.0.1 --port 8083 --reload
```

### 9.6. Exemplo de uso via curl

```bash
# Health check
curl http://127.0.0.1:8083/health

# Estatísticas
curl -H "X-API-Key: SUA_CHAVE" http://127.0.0.1:8083/stats

# Busca vetorial (sem LLM)
curl -X POST -H "X-API-Key: SUA_CHAVE" -H "Content-Type: application/json" \
  -d '{"query": "requisitos para certificação de pilotos", "limit": 5}' \
  http://127.0.0.1:8083/api/vector-search

# Chat com RAG
curl -X POST -H "X-API-Key: SUA_CHAVE" -H "Content-Type: application/json" \
  -d '{"message": "Quais são os requisitos para pilotos comerciais?", "use_rag": true}' \
  http://127.0.0.1:8083/api/chat

# Listar modelos
curl -H "X-API-Key: SUA_CHAVE" http://127.0.0.1:8083/api/models
```

---

## 10. Interface Web (Frontend)

A interface web é um projeto FastAPI/Jinja2 separado que se comunica com a API RAG via HTTP.

Para documentação completa da interface web, consulte `web/`:

- `web/START_HERE.md` — Visão geral da web
- `web/QUICKSTART.md` — Guia rápido da web
- `web/README.md` — Documentação completa da web

### Resumo de como executar:

```bash
cd web/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp env.example .env
# Editar .env (configurar API_BASE_URL e API_KEY)
python main.py
```

---

## 11. Scripts Utilitários

> **IMPORTANTE:** Todos os scripts da pasta `scripts/` **devem** ser executados com `python -m scripts.nome` a partir da raiz do projeto. Executar diretamente com `python scripts/nome.py` causará `ModuleNotFoundError` porque as importações internas (`from config import config`, `from models.embeddings import ...`, etc.) não serão resolvidas corretamente.

### Resumo dos scripts:

| Script | Comando | Descrição |
|--------|---------|-----------|
| `setup_qdrant.py` | `python -m scripts.setup_qdrant` | Inicializa a coleção no Qdrant |
| `ingest_decea.py` | `python -m scripts.ingest_decea` | Baixa e ingere documentos DECEA |
| `ingest_lexml.py` | `python -m scripts.ingest_lexml` | Baixa e ingere documentos LexML |
| `ingest_pdfs.py` | `python -m scripts.ingest_pdfs --source DIR` | Ingere PDFs de um diretório |
| `reset_database.py` | `python -m scripts.reset_database --confirm` | Reseta o banco vetorial |
| `inspect_qdrant.py` | `python -m scripts.inspect_qdrant` | Inspeciona dados do Qdrant |
| `test_system.py` | `python -m scripts.test_system` | Testa todos os componentes |
| `test_chatbot.py` | `python -m scripts.test_chatbot` | Testa os endpoints do chatbot |

### Exemplos detalhados:

```bash
# Preparação: estar na raiz com venv ativado
cd aviation-rag-system/
source venv/bin/activate

# 1. Criar a coleção no Qdrant
python -m scripts.setup_qdrant

# 2. Ingerir documentos DECEA
python -m scripts.ingest_decea --doc-types ICA,MCA --limit 100

# 3. Ingerir documentos LexML
python -m scripts.ingest_lexml --keywords "aviação,ANAC" --limit 200

# 4. Ingerir PDFs locais
python -m scripts.ingest_pdfs --source ./data/originals --recursive

# 5. Verificar o que foi indexado
python -m scripts.inspect_qdrant

# 6. Testar todo o sistema
python -m scripts.test_system

# 7. Testar o chatbot (precisa da API rodando)
python -m scripts.test_chatbot

# 8. Resetar tudo e re-ingerir
python -m scripts.reset_database --confirm --clear-tracker
```

---

## 12. Deploy em Produção (systemctl + Nginx)

O sistema está configurado para rodar em produção usando **systemd** para gerenciamento de processos e **Nginx** como reverse proxy.

### 12.1. Serviço da API RAG (`ragapi.service`)

Este serviço roda o backend da API (busca vetorial + LLM).

Arquivo: `/etc/systemd/system/ragapi.service`

```ini
[Unit]
Description=API - RAG (Uvicorn)
After=network.target

[Service]
Type=simple
User=jean
Group=jean
WorkingDirectory=/home/jean/RAGSystem/aviation-rag-system

# Ativa o venv automaticamente
Environment="PATH=/home/jean/RAGSystem/aviation-rag-system/venv/bin"

# Comando de execução
ExecStart=/home/jean/RAGSystem/aviation-rag-system/venv/bin/uvicorn api.server:app --host 127.0.0.1 --port 8083

Restart=always
RestartSec=5

# Logs
StandardOutput=journal
StandardError=journal
```

### 12.2. Serviço da Interface Web (`ragweb.service`)

Este serviço roda o frontend web.

Arquivo: `/etc/systemd/system/ragweb.service`

```ini
[Unit]
Description= RAG Web aplication(Uvicorn)
After=network.target

[Service]
Type=simple
User=jean
Group=jean
WorkingDirectory=/home/jean/RAGSystem/aviation-rag-system/web

# Ativa o venv automaticamente
Environment="PATH=/home/jean/RAGSystem/aviation-rag-system/web/venv/bin"

# Comando de execução
ExecStart=/home/jean/RAGSystem/aviation-rag-system/web/venv/bin/python main.py

Restart=always
RestartSec=5

# Logs
StandardOutput=journal
StandardError=journal
```

### 12.3. Comandos de gerenciamento (systemctl)

```bash
# Habilitar os serviços (iniciar automaticamente no boot)
sudo systemctl enable ragapi
sudo systemctl enable ragweb

# Iniciar os serviços
sudo systemctl start ragapi
sudo systemctl start ragweb

# Verificar status
sudo systemctl status ragapi
sudo systemctl status ragweb

# Ver logs em tempo real
sudo journalctl -u ragapi -f
sudo journalctl -u ragweb -f

# Reiniciar após alterações no código
sudo systemctl restart ragapi
sudo systemctl restart ragweb

# Parar os serviços
sudo systemctl stop ragapi
sudo systemctl stop ragweb
```

### 12.4. Nginx (Reverse Proxy)

O Nginx atua como reverse proxy, recebendo as requisições na porta 80 e redirecionando para os serviços internos.

Arquivo: `/etc/nginx/sites-enabled/default`

```nginx
server {
    listen 80 default_server;
    server_name _;

    # Redirect /ragweb to /ragweb/
    location /ragweb {
        return 302 /ragweb/;
    }

    # Interface Web (porta 8082)
    location /ragweb/ {
        proxy_pass http://127.0.0.1:8082/;

        proxy_http_version 1.1;
        proxy_set_header Host                   $host;
        proxy_set_header X-Real-IP              $remote_addr;
        proxy_set_header X-Forwarded-for        $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto      $scheme;
        proxy_redirect off;

        # Timeouts adequados para streaming e RAG
        proxy_read_timeout 180s;
        proxy_connect_timeout 60s;
        proxy_send_timeout 180s;
    }

    # Redirect /ragapi to /ragapi/
    location /ragapi {
        return 302 /ragapi/;
    }

    # API RAG (porta 8083)
    location /ragapi/ {
        proxy_pass http://127.0.0.1:8083/;

        proxy_http_version 1.1;
        proxy_set_header Host                   $host;
        proxy_set_header X-Real-IP              $remote_addr;
        proxy_set_header X-Forwarded-for        $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto      $scheme;
        proxy_redirect off;

        # Timeouts adequados para RAG + LLM
        proxy_read_timeout 180s;
        proxy_connect_timeout 60s;
        proxy_send_timeout 180s;
    }
}
```

### 12.5. Comandos Nginx

```bash
# Testar configuração
sudo nginx -t

# Recarregar configuração (sem downtime)
sudo systemctl reload nginx

# Reiniciar Nginx
sudo systemctl restart nginx

# Ver logs
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log
```

### 12.6. URLs de acesso em produção

Com a configuração acima, os serviços ficam acessíveis em:

| Serviço | URL |
|---------|-----|
| Interface Web | `http://SEU_IP/ragweb/` |
| API RAG | `http://SEU_IP/ragapi/` |
| Health (Web) | `http://SEU_IP/ragweb/health` |
| Health (API) | `http://SEU_IP/ragapi/health` |
| Estatísticas | `http://SEU_IP/ragapi/stats` (requer API Key) |

### 12.7. Ordem de inicialização em produção

A ordem recomendada para inicializar todos os serviços é:

```bash
# 1. Qdrant (se não for Docker com restart: always)
docker start qdrant

# 2. Ollama (geralmente já roda como serviço)
sudo systemctl start ollama

# 3. API RAG
sudo systemctl start ragapi

# 4. Interface Web
sudo systemctl start ragweb

# 5. Nginx (geralmente já roda)
sudo systemctl start nginx
```

---

## 13. Resolução de Problemas

### Erros de importação ao executar scripts

```
ModuleNotFoundError: No module named 'config'
```

**Causa:** O script foi executado diretamente (`python scripts/arquivo.py`) em vez de como módulo.

**Solução:** Use `python -m scripts.arquivo` a partir da raiz do projeto:

```bash
cd aviation-rag-system/
source venv/bin/activate
python -m scripts.setup_qdrant  # ✅ Correto
```

### Qdrant não conecta

```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Causa:** Qdrant não está rodando.

**Solução:**

```bash
docker ps | grep qdrant  # Verificar se está rodando
docker start qdrant       # Iniciar
curl http://localhost:6333/healthz  # Testar
```

### Ollama não conecta

```
ConnectionError: HTTPConnectionPool(host='localhost', port=11434)
```

**Causa:** Ollama não está rodando ou o modelo não está carregado.

**Solução:**

```bash
systemctl status ollama   # Verificar status
ollama list                # Listar modelos disponíveis
ollama pull llama3.2:3b    # Baixar modelo se necessário
```

### Erro de CUDA / GPU

```
RuntimeError: CUDA out of memory
```

**Causa:** GPU sem memória suficiente para o modelo de embeddings.

**Solução:** O modelo automaticamente cai para CPU se CUDA não estiver disponível. Para forçar CPU:

```bash
export CUDA_VISIBLE_DEVICES=""  # Desabilitar GPU
```

### Timeout na API de chat

```
504 Gateway Time-out
```

**Causa:** O LLM está demorando mais que o timeout do Nginx (180s).

**Solução:** Aumente os timeouts no Nginx ou use um modelo LLM menor:

```env
OLLAMA_MODEL=llama3.2:3b  # Modelo menor = mais rápido
```

### Erro de autenticação

```
{"detail": "Invalid API Key"}
```

**Causa:** A chave API enviada no header `X-API-Key` não corresponde ao `API_KEY` no `.env`.

**Solução:** Verifique se a chave é a mesma em ambos os `.env` (raiz e `web/.env`).

### Serviço systemd não inicia

```bash
sudo journalctl -u ragapi -n 50  # Ver últimas 50 linhas de log
sudo journalctl -u ragweb -n 50
```

Causas comuns:
- Caminho do venv incorreto no `Environment`
- `WorkingDirectory` incorreto
- Dependência (Qdrant, Ollama) não disponível

---

## Referências

- [START_HERE.md](START_HERE.md) — Introdução e arquitetura
- [QUICKSTART.md](QUICKSTART.md) — Guia rápido de execução
- [web/](../web/START_HERE.md) — Documentação da interface web
- [FastAPI](https://fastapi.tiangolo.com)
- [Qdrant](https://qdrant.tech/documentation)
- [Ollama](https://ollama.com)
- [Legal-BERTimbau](https://huggingface.co/rufimelo/Legal-BERTimbau-sts-large-ma-v3)
- [Portal AirData](https://www.airdata.ita.br)
- [GitHub AirData](https://github.com/ita-airdata)
