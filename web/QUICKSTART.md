# Guia Rápido de Execução - AirData RAG Web Interface

> Este guia contém os passos mínimos necessários para colocar a aplicação web em funcionamento. Para informações mais detalhadas, consulte o [README.md](README.md).

---

## Resumo

A execução da aplicação web se resume a **3 etapas**:

1. Criar e ativar um ambiente virtual Python
2. Instalar as dependências do `requirements.txt`
3. Executar o `main.py`

---

## Passo a Passo

### 1. Navegue até o diretório do projeto web

```bash
cd /caminho/para/aviation-rag-system/web
```

### 2. Crie um ambiente virtual Python

```bash
python3 -m venv venv
```

> **Nota:** Se o comando `python3` não estiver disponível, tente `python`. O importante é usar Python 3.8 ou superior.

### 3. Ative o ambiente virtual

**Linux / macOS:**
```bash
source venv/bin/activate
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

Após a ativação, você verá `(venv)` no início da linha de comando, indicando que o ambiente virtual está ativo.

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

Este comando instalará todas as bibliotecas necessárias, incluindo:
- **FastAPI** — framework web
- **Uvicorn** — servidor ASGI
- **Jinja2** — motor de templates HTML
- **httpx** — cliente HTTP assíncrono (para comunicação com a API backend)
- **pydantic-settings** — gerenciamento de configurações
- **loguru** — sistema de logging
- **python-dotenv** — carregamento de variáveis de ambiente

### 5. Configure as variáveis de ambiente

Copie o arquivo de exemplo e edite com suas configurações:

```bash
cp env.example .env
```

Abra o arquivo `.env` em um editor de texto e configure:

```env
# Endereço e porta em que o servidor web vai rodar
HOST=127.0.0.1
PORT=8082
RELOAD=True

# Prefixo de rota (usado quando atrás de um reverse proxy)
ROOT_PATH=/ragweb

# URL da API RAG backend (deve estar rodando e acessível)
API_BASE_URL=http://localhost:8000

# Chave de autenticação da API RAG
API_KEY=sua-chave-api-aqui

# Informações da aplicação
APP_NAME=Aviation RAG Web Interface
APP_VERSION=1.0.0
```

**Variáveis essenciais que você precisa configurar:**

| Variável | O que configurar |
|----------|-----------------|
| `HOST` | IP onde o servidor vai escutar (`127.0.0.1` para acesso local, `0.0.0.0` para acesso externo) |
| `PORT` | Porta do servidor web (ex: `8082`) |
| `API_BASE_URL` | URL completa da API RAG backend (ex: `http://161.24.29.22/ragapi`) |
| `API_KEY` | Chave de API para autenticação com o backend |
| `ROOT_PATH` | Prefixo de URL se o app estiver atrás de um reverse proxy (ex: `/ragweb`). Deixe vazio (`""`) se não usar proxy |

### 6. Execute a aplicação

```bash
python main.py
```

A saída esperada será semelhante a:

```
INFO:     Uvicorn running on http://127.0.0.1:8082 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx] using WatchFiles
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 7. Acesse no navegador

Abra o navegador e acesse:

```
http://127.0.0.1:8082/ragweb/
```

> **Atenção:** A URL completa inclui o `ROOT_PATH` configurado no `.env`. Se você configurou `ROOT_PATH=/ragweb`, a URL base será `http://HOST:PORT/ragweb/`. Se `ROOT_PATH` estiver vazio, acesse `http://HOST:PORT/`.

---

## Verificação Rápida

Após iniciar a aplicação, você pode verificar se está funcionando:

| Verificação | URL | Resultado Esperado |
|-------------|-----|-------------------|
| Health check | `http://HOST:PORT/ragweb/health` | JSON: `{"status": "healthy", ...}` |
| Página inicial | `http://HOST:PORT/ragweb/` | Página HTML com cards de funcionalidades |
| Página sobre | `http://HOST:PORT/ragweb/sobre` | Página HTML com informações do sistema |

> As páginas de **Chat**, **Busca** e **Estatísticas** necessitam que a API RAG backend esteja acessível. Se ela não estiver rodando, essas páginas exibirão mensagens de erro.

---

## Encerrando a Aplicação

Para parar o servidor, pressione `Ctrl+C` no terminal onde ele está rodando.

Para desativar o ambiente virtual:

```bash
deactivate
```

---

## Problemas Comuns

### "ModuleNotFoundError: No module named 'fastapi'"
O ambiente virtual não está ativado ou as dependências não foram instaladas. Execute:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "Connection refused" ao acessar Chat ou Busca
A API RAG backend não está rodando ou a variável `API_BASE_URL` no `.env` está incorreta. Verifique se a API está acessível:
```bash
curl http://SEU_API_BASE_URL/stats
```

### "API Error: 401" ou "Unauthorized"
A chave de API (`API_KEY`) no `.env` está incorreta. Verifique com o administrador da API backend qual é a chave correta.

### Página carrega mas sem dados
Verifique o terminal do servidor — erros de conexão com a API backend serão exibidos nos logs do Loguru.

---

## Referências

- [START_HERE.md](START_HERE.md) — Visão geral e arquitetura do projeto
- [README.md](README.md) — Documentação completa e detalhada
