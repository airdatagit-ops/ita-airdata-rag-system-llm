# Guia Rápido de Execução - Aviation RAG System

> Este guia mostra os passos mínimos para colocar o sistema completo em funcionamento. Para documentação detalhada, consulte o [README.md](README.md).

---

## Visão Geral

O sistema possui **3 serviços principais** que precisam estar rodando:

| Serviço | Porta Padrão | Descrição |
|---------|-------------|-----------|
| **Qdrant** | 6333 | Banco de dados vetorial |
| **Ollama** | 11434 | Servidor local de LLMs |
| **API RAG** | 8083 | Backend da API (busca + chat + LLM) |
| **Web App** | 8082 | Interface web para o usuário |

A ordem de inicialização deve ser: **Qdrant → Ollama → API RAG → Web App**.

---

## 1. Iniciar o Qdrant

Se via Docker (recomendado):

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Verifique se está rodando:

```bash
curl http://localhost:6333/healthz
```

---

## 2. Iniciar o Ollama

Instale o Ollama (se ainda não instalado):

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Baixe um modelo (por exemplo, `llama3.2:3b`):

```bash
ollama pull llama3.2:3b
```

Verifique se está rodando:

```bash
ollama list
```

---

## 3. Configurar e Executar a API RAG

### 3.1. Navegue até a raiz do projeto

```bash
cd aviation-rag-system/
```

### 3.2. Crie e ative o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3.3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3.4. Configure o `.env`

```bash
cp env.example .env
# Edite o .env com suas configurações (API_KEY, QDRANT_HOST, OLLAMA_MODEL, etc.)
```

Variáveis essenciais a configurar:

| Variável | Exemplo | Descrição |
|----------|---------|-----------|
| `API_KEY` | (gerar aleatório) | Chave de autenticação da API |
| `QDRANT_HOST` | `localhost` | Host do Qdrant |
| `QDRANT_PORT` | `6333` | Porta do Qdrant |
| `OLLAMA_HOST` | `http://localhost:11434` | URL do Ollama |
| `OLLAMA_MODEL` | `llama3.2:3b` | Modelo LLM padrão |
| `API_HOST` | `127.0.0.1` | Host da API |
| `API_PORT` | `8083` | Porta da API |

### 3.5. Inicialize o banco vetorial (primeira execução)

```bash
python -m scripts.setup_qdrant
```

### 3.6. Ingira documentos (primeira execução)

```bash
# Documentos DECEA (ICA, MCA, etc.)
python -m scripts.ingest_decea --doc-types ICA --limit 50

# Documentos LexML (legislação)
python -m scripts.ingest_lexml --limit 100
```

### 3.7. Teste o sistema

```bash
python -m scripts.test_system
```

### 3.8. Execute a API

```bash
python -m api.server
```

Ou diretamente com uvicorn:

```bash
uvicorn api.server:app --host 127.0.0.1 --port 8083 --reload
```

Verifique:

```bash
curl -H "X-API-Key: SUA_CHAVE" http://127.0.0.1:8083/stats
```

---

## 4. Configurar e Executar a Interface Web

### 4.1. Navegue até o diretório web

```bash
cd web/
```

### 4.2. Crie e ative o ambiente virtual da web

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4.3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4.4. Configure o `.env` da web

```bash
cp env.example .env
# Edite o .env — a variável mais importante é API_BASE_URL (deve apontar para a API RAG)
```

### 4.5. Execute a aplicação web

```bash
python main.py
```

### 4.6. Acesse no navegador

```
http://127.0.0.1:8082/ragweb/
```

---

## 5. Verificação Rápida

| Componente | Comando de Verificação |
|-----------|----------------------|
| Qdrant | `curl http://localhost:6333/healthz` |
| Ollama | `ollama list` |
| API RAG | `curl -H "X-API-Key: SUA_CHAVE" http://127.0.0.1:8083/health` |
| Web App | `curl http://127.0.0.1:8082/ragweb/health` |

---

## Aviso Importante sobre Execução de Scripts

> **Todos os scripts da pasta `scripts/` devem ser executados com `python -m scripts.nome_do_script`** a partir da raiz do projeto. Executar diretamente com `python scripts/nome.py` causará erros de importação, pois os módulos internos (`config`, `models`, `database`, etc.) não serão encontrados.

Exemplo correto:

```bash
# Estando na raiz do projeto: aviation-rag-system/
python -m scripts.setup_qdrant
python -m scripts.ingest_decea
python -m scripts.test_system
```

Exemplo **incorreto** (vai dar erro):

```bash
python scripts/setup_qdrant.py     # ❌ ModuleNotFoundError
cd scripts && python setup_qdrant.py  # ❌ ModuleNotFoundError
```

---

## Referências

- [START_HERE.md](START_HERE.md) — Visão geral e arquitetura
- [README.md](README.md) — Documentação completa e detalhada
- [web/docs/](../web/docs/START_HERE.md) — Documentação da interface web
