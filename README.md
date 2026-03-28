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
12. [Avaliação de Qualidade](#12-avaliação-de-qualidade)
13. [Testes e Automação (Makefile)](#13-testes-e-automação-makefile)
14. [Deploy em Produção (systemctl + Nginx)](#14-deploy-em-produção-systemctl--nginx)
15. [Resolução de Problemas](#15-resolução-de-problemas)

---

## 1. Visão Geral do Sistema

O Aviation RAG System é uma plataforma que combina:

- **Busca semântica vetorial** — Encontra trechos de documentos similares à pergunta do usuário usando embeddings (dense vectors)
- **Busca por keywords (opcional)** — Busca BM25 via sparse vectors para termos exatos, siglas e referências a artigos
- **Busca híbrida (opcional)** — Combina busca semântica + keywords usando Reciprocal Rank Fusion (RRF)
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
| Scraper base (ABC) | Interface async + registry | `crawler/scrapers/base.py` |
| Scraper DECEA | Requests + BeautifulSoup (async via to_thread) | `crawler/scrapers/decea_scraper.py` |
| Scraper LexML | aiohttp (async) + BeautifulSoup | `crawler/scrapers/lexml_scraper.py` |
| Scraper PDF (local) | PDFParser (pdfplumber/PyMuPDF) | `crawler/scrapers/pdf_scraper.py` |
| Document Store | SQLite registry + change detection | `pipeline/document_store.py` |
| Embedding Store | Parquet-backed vector storage | `pipeline/embedding_store.py` |
| Avaliação (retrieval) | Golden Set + Métricas IR | `evaluation/evaluate_retrieval.py` |
| Avaliação (geração) | Heurísticas de qualidade LLM | `evaluation/evaluate_generation.py` |
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
| `lxml` | Parser HTML rápido para scraper DECEA |
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
| `SEARCH_SCORE_THRESHOLD` | float | `0.3` | Score mínimo de similaridade (apenas busca dense-only) |
| `SEARCH_DENSE_ENABLED` | bool | `true` | Habilita busca semântica (dense vectors) |
| `SEARCH_SPARSE_ENABLED` | bool | `false` | Habilita busca por keywords/BM25 (sparse vectors via fastembed) |
| `SPARSE_EMBEDDING_MODEL` | string | `Qdrant/bm25` | Modelo de sparse embeddings (usado quando `SEARCH_SPARSE_ENABLED=true`) |
| `HNSW_M` | int | `16` | Parâmetro M do índice HNSW |
| `HNSW_EF_CONSTRUCT` | int | `100` | Parâmetro ef_construct do HNSW |
| `HNSW_EF_SEARCH` | int | `64` | Parâmetro ef para busca no HNSW |

> **Busca híbrida:** Quando ambos `SEARCH_DENSE_ENABLED` e `SEARCH_SPARSE_ENABLED` estão habilitados, o sistema combina os resultados usando Reciprocal Rank Fusion (RRF) via Qdrant Query API. Isso melhora a recuperação de termos exatos (siglas, artigos, nomes de ICAs) que a busca semântica pura pode perder. A coleção deve ser recriada ao habilitar sparse pela primeira vez (`make index RECREATE=1`).

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
| `LEXML_MAX_RATE` | int | `5` | Máximo de requisições por segundo (rate limiting do scraper async) |

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

A coleção é criada automaticamente pela fase de indexação:

```bash
make index RECREATE=1
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
make index RECREATE=1
```

Para forçar re-coleta e reconstrução completa:

```bash
make collect FORCE=1    # apaga docs da store e re-coleta
make embed FORCE=1      # re-gera todos os embeddings
make index RECREATE=1   # recria a coleção no Qdrant
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

**Scraper:** `crawler/scrapers/decea_scraper.py`

O scraper DECEA herda de `BaseScraper` e usa **HTTP direto** (`requests` + `BeautifulSoup`) para acessar o portal de publicações do DECEA (`publicacoes.decea.mil.br`). O portal usa Next.js com Server-Side Rendering, o que permite extrair todo o conteúdo sem navegador. Ele:

1. Parseia o índice de publicações via HTML estático
2. Filtra por tipo de documento (ICA, MCA, PCA, DCA, CIRCEA, NSCA, etc.)
3. Extrai URLs de PDFs assinadas (S3) de cada página de publicação
4. Baixa os PDFs originais em paralelo (via `asyncio.Semaphore` + `to_thread`)
5. Extrai texto dos PDFs (PyMuPDF, pdfplumber, OCR fallback)
6. Armazena conteúdo e metadados no SQLite (`data/store.db`)

**Como executar:**

```bash
make collect SOURCES=decea                     # Coleta todos os tipos
make collect SOURCES=decea LIMIT=50            # Limita a 50 docs
make collect SOURCES=decea DOC_TYPES=ICA,MCA   # Apenas ICA e MCA
make collect SOURCES=decea CHECK=1             # Verifica alterações
make collect SOURCES=decea FORCE=1             # Re-coleta do zero
```

### 7.2. LexML (Legislação Federal)

**Scraper:** `crawler/scrapers/lexml_scraper.py`

O scraper LexML herda de `BaseScraper` e usa **aiohttp (assíncrono)** + BeautifulSoup para buscar documentos no portal LexML Brasil com downloads paralelos. Ele:

1. Busca documentos por palavras-chave individualmente na interface web do LexML (paginação automática)
2. Extrai metadados (título, URN, tipo, data, autoria)
3. Segue links para Senado ou Planalto e extrai o texto integral
4. Baixa múltiplos documentos em paralelo (semáforo configurável)
5. Deduplica resultados por URN entre keywords
6. Armazena conteúdo e metadados no SQLite (`data/store.db`)

Por padrão, apenas legislação **federal** é coletada (`f2-localidade=Brasil`), evitando documentos estaduais/municipais cujos portais não são suportados para extração de texto integral. Para incluir todas as localidades, use `ALL_LOCALITIES=1`.

**Como executar:**

```bash
make collect SOURCES=lexml                             # Coleta legislação federal (padrão)
make collect SOURCES=lexml ALL_LOCALITIES=1             # Inclui estadual/municipal
make collect SOURCES=lexml LIMIT=50                    # Limita a 50 docs por keyword
make collect SOURCES=lexml KEYWORDS="ANAC,portaria"    # Keywords específicas
make collect SOURCES=lexml CHECK=1                     # Verifica alterações
make collect SOURCES=lexml FORCE=1                     # Re-coleta do zero
```

### 7.3. PDFs Locais

Para coletar PDFs que já estejam em um diretório local:

```bash
make collect SOURCES=pdf                           # PDFs do diretório padrão (./data/pdfs)
make collect SOURCES=pdf PDF_DIR=./meus-pdfs/      # Diretório customizado
```

### 7.4. Arquitetura de Scrapers

Todos os scrapers herdam de `BaseScraper` (`crawler/scrapers/base.py`), que define uma interface async unificada:

| Método | Descrição |
|--------|-----------|
| `search(**kwargs)` | Busca documentos e retorna metadados |
| `fetch_document(doc)` | Extrai conteúdo completo de um documento, retorna `ScrapedDocument` |
| `fetch_all(docs, concurrency)` | Fetch paralelo com `asyncio.Semaphore` (herdado, pode ser sobrescrito) |
| `make_doc_id(doc)` | Gera ID estável e filesystem-safe |
| `save_original_file(...)` | Salva arquivo original (PDF/HTML) + metadata JSON |

Os scrapers são registrados automaticamente via decorator `@register_scraper` e descobertos pelo registry em `crawler/scrapers/__init__.py`:

```python
from crawler.scrapers import get_scraper, list_scrapers

scraper = get_scraper("decea")  # instancia DECEAScraper
names = list_scrapers()          # ["decea", "lexml", "pdf"]
```

Para adicionar um novo scraper, basta criar uma classe em `crawler/scrapers/` que herde de `BaseScraper` e use `@register_scraper`.

### 7.5. Rastreamento de Documentos

O `DocumentStore` (`pipeline/document_store.py`) mantém um registro em SQLite (`data/store.db`) com:

- Conteúdo completo de cada documento
- Hash SHA256 do conteúdo (para detectar alterações)
- Metadados (título, URL, URN, tipo, source)
- Timestamps de coleta e atualização
- Log de embeddings gerados

Isso evita re-downloads desnecessários em execuções subsequentes.

---

## 8. Pipeline de Ingestão (Arquitetura 3 Fases)

O pipeline de ingestão foi reestruturado em **3 fases independentes e idempotentes**, cada uma executável separadamente via Makefile. Apenas documentos/embeddings cujo conteúdo mudou são reprocessados (detecção via hash SHA256).

### Armazenamento

| Componente | Tecnologia | Caminho |
|------------|-----------|---------|
| Registro de documentos | SQLite | `data/store.db` |
| Embeddings densos | Parquet (Snappy) | `data/embeddings/dense/` |
| Embeddings esparsos | Parquet (Snappy) | `data/embeddings/sparse/` |
| Busca vetorial | Qdrant | `localhost:6333` |

### Fase 1: Collect (`make collect`)

Executa os scrapers (LexML, DECEA, PDFs locais) e persiste documentos no SQLite. Três modos de coleta controlam o comportamento em re-runs:

| Modo | Comando | Comportamento |
|---|---|---|
| **Default** | `make collect` | Pula documentos que já existem no SQLite (sem HTTP). Apenas novos são baixados. Re-runs instantâneos. |
| **Check** | `make collect CHECK=1` | Re-baixa todos os documentos e recalcula hash. Atualiza apenas os que mudaram na fonte. |
| **Force** | `make collect FORCE=1` | Apaga todos os documentos da fonte no SQLite e re-coleta do zero. |

```bash
make collect                              # re-run rápido (pula existentes)
make collect CHECK=1                      # verificar mudanças nas fontes
make collect FORCE=1                      # apagar e re-coletar tudo
make collect SOURCES=lexml LIMIT=50       # apenas LexML, 50 docs
make collect SOURCES=decea                # apenas DECEA (todos os tipos)
make collect SOURCES=pdf PDF_DIR=./data/pdfs  # PDFs locais
make collect SOURCES=lexml,decea,pdf      # todas as fontes
```

### Fase 2: Embed (`make embed`)

Lê documentos do SQLite, aplica validação/limpeza/chunking, gera embeddings e salva em Parquet. Suporta modos `dense`, `sparse` ou `hybrid`.

```bash
make embed                    # Incremental, modo do config
make embed MODE=dense         # Apenas dense
make embed MODE=hybrid        # Dense + sparse
make embed FORCE=1            # Re-gerar tudo
```

### Fase 3: Index (`make index`)

Carrega embeddings do Parquet e faz bulk-upsert no Qdrant. Nenhum modelo de embedding é carregado, fase puramente I/O.

```bash
make index                    # Push para Qdrant
make index RECREATE=1         # Recriar coleção antes
```

### Pipeline completo

```bash
make pipeline                          # collect + embed + index
make pipeline MODE=hybrid RECREATE=1   # Full rebuild com busca híbrida
```

### Analytics (`make query` / `make explore`)

Duas interfaces para explorar os documentos coletados no SQLite:

#### Console SQL (`make query`)

```bash
make query                                           # Console SQL interativo (REPL)
make query SQL="SELECT source, doc_type, COUNT(*) FROM documents GROUP BY source, doc_type"
```

O console SQL suporta comandos especiais: `\tables`, `\schema`, `\sources`, `\types`, `\counts`.

Exemplo de consulta:

```bash
make query SQL="SELECT source, COUNT(*) AS total, ROUND(AVG(LENGTH(content))) AS avg_chars FROM documents GROUP BY source ORDER BY total DESC"
```

```
+--------+-------+-----------+
| source | total | avg_chars |
+--------+-------+-----------+
| lexml  | 3029  | 12450.0   |
| decea  | 447   | 38721.0   |
+--------+-------+-----------+
```

#### Interface Web (`make explore`)

```bash
make explore    # Abre o Datasette no navegador (http://localhost:8001)
```

O Datasette oferece uma interface web completa para navegar tabelas, aplicar filtros visuais, executar SQL arbitrário e exportar resultados em JSON/CSV. O banco abre em **modo read-only** (`--immutable`).

**Autenticação:** O acesso requer login. Ao abrir, você será redirecionado para a página de login. Usuários configurados:

| Usuário | Descrição |
|---------|-----------|
| `admin` | Administrador |
| `berg`  | Usuário padrão |

**Adicionando novos usuários:**

1. Gere o hash da senha:

```bash
python -c "from datasette_auth_passwords import hash_password; print(hash_password('minha_senha'))"
```

2. Adicione ao `metadata.yml`:

```yaml
plugins:
  datasette-auth-passwords:
    novousuario_password_hash: "pbkdf2_sha256$480000$..."
```

3. Autorize o acesso na seção `allow`:

```yaml
allow:
  id:
    - admin
    - berg
    - novousuario
```

**Deploy em produção (Nginx):**

```bash
python -m datasette serve --immutable data/store.db --metadata metadata.yml \
  --host 127.0.0.1 --port 8001 --setting base_url /datasette/ --cors
```

```nginx
location /datasette/ {
    proxy_pass http://127.0.0.1:8001/datasette/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

### Fluxo detalhado:

```
  FASE 1: COLLECT                    FASE 2: EMBED                    FASE 3: INDEX
  ─────────────────                  ────────────────                  ────────────────
  Web Scrapers                       SQLite → Chunking                Parquet → Qdrant
  (DECEA, LexML)                     → Embedding → Parquet
        │                                  │                                │
        ▼                                  ▼                                ▼
  SHA256 hash check              Quality Gate + TextCleaner          Bulk upsert com
        │                                  │                          indexação desativada
        ▼                                  ▼                                │
  SQLite (store.db)              ArticleChunker/ICAChunker                  ▼
  INSERT/UPDATE/SKIP                       │                          Qdrant collection
                                           ▼                          (dense/sparse/hybrid)
                                  EmbeddingModel (GPU)
                                  + SparseEncoder (BM25)
                                           │
                                           ▼
                                  Parquet (data/embeddings/)
                                  + embedding_log (SQLite)
```

> **Nota:** A limpeza de texto é aplicada em memória durante a Fase 2 (embed), antes do chunking e embedding. Os dados brutos nunca são modificados.

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

**Pipeline 3 fases (recomendado):**

| Script | Comando | Descrição |
|--------|---------|-----------|
| `collect.py` | `python -m scripts.collect` | Fase 1: coleta documentos de todas as fontes no SQLite (`make collect`). Modos: default (skip), --check, --force |
| `embed.py` | `python -m scripts.embed` | Fase 2: gera embeddings incrementais em Parquet (`make embed`) |
| `index.py` | `python -m scripts.index` | Fase 3: carrega embeddings no Qdrant (`make index`) |
| `query.py` | `python -m scripts.query` | Console SQL interativo para explorar o SQLite (`make query`) |

**Utilitários:**

| Script | Comando | Descrição |
|--------|---------|-----------|
| `validate_data.py` | `python -m scripts.validate_data` | Valida qualidade e limpeza dos documentos (`make validate-data`) |
| `inspect_qdrant.py` | `python -m scripts.inspect_qdrant` | Inspeciona dados do Qdrant |
| `test_system.py` | `python -m scripts.test_system` | Testa todos os componentes |
| `test_chatbot.py` | `python -m scripts.test_chatbot` | Testa os endpoints do chatbot |

> Para avaliação de qualidade da busca, veja a [Seção 12](#12-avaliação-de-qualidade).

### Exemplos detalhados:

```bash
# Preparação: estar na raiz com venv ativado
cd aviation-rag-system/
source venv/bin/activate

# 1. Coletar documentos (Fase 1)
make collect                                  # Coleta DECEA + LexML
make collect SOURCES=decea LIMIT=50           # Apenas DECEA, 50 docs
make collect SOURCES=pdf PDF_DIR=./meus-pdfs  # PDFs locais

# 2. Gerar embeddings (Fase 2)
make embed MODE=sparse                        # Apenas sparse (sem GPU)
make embed MODE=hybrid                        # Dense + sparse (GPU)

# 3. Indexar no Qdrant (Fase 3)
make index                                    # Upsert incremental
make index RECREATE=1                         # Recria a coleção

# 4. Pipeline completo (3 fases em sequência)
make pipeline

# 5. Verificar o que foi indexado
python -m scripts.inspect_qdrant

# 6. Consultar documentos coletados
make query SQL="SELECT source, COUNT(*) n FROM documents GROUP BY source"
make explore                                  # Web UI (Datasette)

# 7. Testar todo o sistema
python -m scripts.test_system

# 8. Resetar tudo e re-coletar
make collect FORCE=1                          # Apaga e re-coleta
make embed FORCE=1                            # Re-gera embeddings
make index RECREATE=1                         # Recria Qdrant

# 9. Validar qualidade dos documentos
python -m scripts.validate_data --report-only
```

---

## 12. Avaliação de Qualidade

O sistema inclui dois módulos de avaliação independentes: um para a **busca semântica (retrieval)** e outro para a **qualidade de geração do LLM**. Ambos compartilham o mesmo golden set e podem ser executados via `Makefile` ou diretamente.

### 12.1. Conceitos

A avaliação é baseada em um **golden set** (`evaluation/golden_set.csv`) — um conjunto curado de queries com documentos esperados como resposta, cada um classificado por relevância:

| Relevância | Peso NDCG | Significado |
|------------|-----------|-------------|
| `relevant` | 2 | O documento responde diretamente à query |
| `moderate` | 1 | O documento é relacionado mas não responde completamente |
| `irrelevant` | 0 | Documento de cobertura (não deve existir no banco) |

Uma mesma query pode ter múltiplas linhas quando mais de um documento é esperado. A coluna `category` distingue queries de `retrieval` (devem encontrar documentos) de queries de `coverage` (testam que o sistema rejeita corretamente buscas sem resultado).

O golden set atual contém **60 queries** cobrindo 30+ ICAs diferentes, incluindo 5 queries de cobertura para documentos não indexados.

### 12.2. Avaliação de Retrieval

Mede a qualidade da busca semântica — se os documentos certos estão sendo encontrados e bem ranqueados.

**Métricas:**

| Métrica | O que mede |
|---------|------------|
| **Hit Rate@K** | % de queries onde ao menos um documento relevante aparece nos top K resultados |
| **MRR** (Mean Reciprocal Rank) | Média de 1/posição do primeiro documento relevante |
| **NDCG@K** | Qualidade do ranking considerando relevância gradual |
| **Precision@K** | % dos K documentos retornados que são relevantes |
| **Recall** | % dos documentos relevantes esperados que foram retornados |
| **Coverage** | % das queries de cobertura corretamente sem resultado |

**Como executar:**

```bash
# Via Makefile (recomendado)
make eval-retrieval
make eval-retrieval K=10 WORKERS=8

# Direto
python -m evaluation.evaluate_retrieval --k 5 --workers 4
python -m evaluation.evaluate_retrieval --golden-set outro.csv --quiet
```

**Parâmetros:**

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--k` | `5` | Documentos a recuperar por query |
| `--workers` | `4` | Threads paralelas para busca no Qdrant |
| `--golden-set` | `evaluation/golden_set.csv` | Caminho do golden set |
| `--output` | `evaluation/results` | Diretório de saída |
| `--quiet` | `false` | Suprime relatório no terminal |

### 12.3. Avaliação de Geração (Heurísticas)

Mede a qualidade das respostas do LLM usando heurísticas — sem necessidade, por enquanto, de golden answers ou de outro LLM como juiz.

**Métricas:**

| Métrica | O que mede |
|---------|------------|
| **Empty Rate** | % de respostas onde o LLM diz "não encontrei" |
| **Citation Rate** | % de respostas que citam documentos (ex: ICA-100-47) |
| **Hedging Rate** | % de respostas com linguagem evasiva ("possivelmente", "talvez") |
| **Mean/Median Length** | Comprimento médio das respostas (em tokens) |

**Como executar:**

```bash
# Via Makefile (recomendado)
make eval-generation
make eval-generation K=3 SAMPLE=10

# Direto
python -m evaluation.evaluate_generation --k 5
python -m evaluation.evaluate_generation --sample 15 --quiet
```

**Parâmetros:**

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--k` | `5` | Documentos a recuperar por query como contexto |
| `--golden-set` | `evaluation/golden_set.csv` | Caminho do golden set |
| `--output` | `evaluation/results` | Diretório de saída |
| `--sample` | todos | Limitar a N queries (para testes rápidos) |
| `--quiet` | `false` | Suprime relatório no terminal |

### 12.4. Executar ambas as avaliações

```bash
make eval                    # retrieval + geração com defaults
make eval K=10               # ambas com K=10
```

### 12.5. Saída

Cada avaliação produz três artefatos em `evaluation/results/`:

| Arquivo | Conteúdo |
|---------|----------|
| `summary_*.csv` | Métricas agregadas |
| `details_*.csv` | Resultado por query |
| `results_*.json` | Dados completos para análise programática |

### 12.6. Otimização de performance

Ambos os scripts são otimizados para execução rápida:

1. **Batch encoding** — Todas as queries são codificadas em uma única chamada ao modelo de embeddings
2. **Busca paralela (retrieval)** — Buscas no Qdrant via `ThreadPoolExecutor` concorrente
3. **Separação de responsabilidades** — Usam `EmbeddingModel` e `QdrantManager` diretamente para controle sobre batching e paralelismo

A avaliação de geração executa chamadas ao LLM sequencialmente, já que o Ollama não se beneficia de requisições paralelas.

### 12.7. Como expandir o golden set

1. Identifique o documento relevante no Qdrant:

```bash
python -m scripts.inspect_qdrant
```

2. Adicione uma linha no `evaluation/golden_set.csv`:

```csv
Q061,Minha nova pergunta sobre o tema X?,ICA-XX-YY-artZZ,relevant,retrieval,Justificativa breve
```

3. Se a query já existe e um novo documento também é relevante, adicione outra linha com o mesmo `query_id`:

```csv
Q061,Minha nova pergunta sobre o tema X?,ICA-XX-YY-artWW,moderate,retrieval,Doc secundário
```

### 12.8. Interpretando os resultados

| Cenário | O que fazer |
|---------|-------------|
| Hit Rate baixo (<50%) | Investigar qualidade dos chunks e do texto extraído |
| MRR baixo (<0.3) | Documentos relevantes em posições baixas. Ajustar embeddings ou score threshold |
| NDCG baixo com Hit Rate alto | Ranking ruim — revisar modelo de embeddings |
| Coverage <100% | Falsos positivos. Ajustar score threshold |
| Empty Rate alto (>50%) | Documentos recuperados não contêm a informação. Melhorar chunking ou expandir base |
| Citation Rate baixo | LLM não cita fontes. Ajustar prompt |
| Hedging Rate alto | LLM inseguro nas respostas. Verificar qualidade do contexto recuperado |

---

## 13. Testes e Automação (Makefile)

### 13.1. Estrutura de testes

Os testes unitários ficam em `tests/`, organizados por domínio:

```
tests/
├── __init__.py
├── crawler/
│   └── scrapers/
│       ├── test_base_scraper.py     # BaseScraper ABC, registry, ScrapedDocument
│       ├── test_decea_scraper.py    # DECEAScraper (sync + async)
│       └── test_lexml_scraper.py    # LexMLScraper (async)
├── evaluation/
│   ├── test_evaluate_retrieval.py
│   └── test_evaluate_generation.py
├── pipeline/
│   ├── test_document_store.py
│   ├── test_embedding_store.py
│   └── test_text_cleaner.py
└── ...
```

Todos os testes usam **mocks** para isolar dependências externas (Qdrant, Ollama, modelo de embeddings), garantindo execução rápida e sem necessidade de serviços rodando.

### 13.2. Executar testes

```bash
# Todos os testes
make test

# Testes de um diretório específico
make test FILE=tests/evaluation/

# Teste de um arquivo específico
make test FILE=tests/evaluation/test_evaluate_retrieval.py

# Direto via pytest
python -m pytest tests/ -v --tb=short
```

### 13.3. Comandos do Makefile

**Pipeline 3 fases:**

| Comando | Descrição |
|---------|-----------|
| `make collect` | Fase 1: coleta novos documentos (pula existentes, re-run instantâneo) |
| `make collect CHECK=1` | Fase 1: re-baixa tudo e verifica hashes (detecta mudanças na fonte) |
| `make collect FORCE=1` | Fase 1: apaga docs da fonte e re-coleta do zero |
| `make embed` | Fase 2: gera embeddings incrementais em Parquet |
| `make index` | Fase 3: carrega embeddings no Qdrant |
| `make pipeline` | Executa as 3 fases em sequência |
| `make query` | Console SQL interativo para explorar documentos |
| `make explore` | Interface web (datasette) para explorar o SQLite |

**Avaliação e utilitários:**

| Comando | Descrição |
|---------|-----------|
| `make help` | Lista todos os comandos disponíveis |
| `make test` | Executa todos os testes unitários |
| `make test FILE=<path>` | Executa testes de um arquivo ou diretório |
| `make eval` | Executa ambas as avaliações (retrieval + geração) |
| `make eval-retrieval` | Avaliação de retrieval |
| `make eval-generation` | Avaliação de geração |
| `make clean` | Remove arquivos de resultado das avaliações |

**Parâmetros configuráveis:**

| Parâmetro | Padrão | Uso |
|-----------|--------|-----|
| `SOURCES` | `lexml,decea` | `make collect SOURCES=lexml` |
| `CHECK` | — | `make collect CHECK=1` (verificar hashes) |
| `FORCE` | — | `make collect FORCE=1` (re-coletar) / `make embed FORCE=1` |
| `MODE` | config | `make embed MODE=hybrid` |
| `RECREATE` | — | `make index RECREATE=1` |
| `SQL` | — | `make query SQL='SELECT ...'` |
| `K` | `5` | `make eval-retrieval K=10` |
| `WORKERS` | `4` | `make eval-retrieval WORKERS=8` |
| `SAMPLE` | todos | `make eval-generation SAMPLE=10` |
| `FILE` | `tests/` | `make test FILE=tests/evaluation/` |
| `LIMIT` | `0` (sem limite) | `make collect LIMIT=50` |
| `CONCURRENCY` | `10` | `make collect CONCURRENCY=3` |
| `KEYWORDS` | — | `make collect KEYWORDS='ANAC,portaria'` |
| `PDF_DIR` | `./data/pdfs` | `make collect SOURCES=pdf PDF_DIR=./meus_pdfs` |
| `BATCH_SIZE` | config | `make embed BATCH_SIZE=64` |

---

## 14. Deploy em Produção (systemctl + Nginx)

O sistema está configurado para rodar em produção usando **systemd** para gerenciamento de processos e **Nginx** como reverse proxy.

### 14.1. Serviço da API RAG (`ragapi.service`)

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

### 14.2. Serviço da Interface Web (`ragweb.service`)

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

### 14.3. Comandos de gerenciamento (systemctl)

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

### 14.4. Nginx (Reverse Proxy)

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

### 14.5. Comandos Nginx

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

### 14.6. URLs de acesso em produção

Com a configuração acima, os serviços ficam acessíveis em:

| Serviço | URL |
|---------|-----|
| Interface Web | `http://SEU_IP/ragweb/` |
| API RAG | `http://SEU_IP/ragapi/` |
| Health (Web) | `http://SEU_IP/ragweb/health` |
| Health (API) | `http://SEU_IP/ragapi/health` |
| Estatísticas | `http://SEU_IP/ragapi/stats` (requer API Key) |

### 14.7. Ordem de inicialização em produção

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

## 15. Resolução de Problemas

### Erros de importação ao executar scripts

```
ModuleNotFoundError: No module named 'config'
```

**Causa:** O script foi executado diretamente (`python scripts/arquivo.py`) em vez de como módulo.

**Solução:** Use `python -m scripts.arquivo` a partir da raiz do projeto:

```bash
cd aviation-rag-system/
source venv/bin/activate
python -m scripts.collect  # ✅ Correto
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
