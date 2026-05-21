# Documentação Completa - AirData RAG Web Interface

> Documentação detalhada da aplicação web do sistema AirData RAG. Para um guia rápido de execução, consulte o [QUICKSTART.md](QUICKSTART.md). Para uma introdução ao projeto, consulte o [START_HERE.md](START_HERE.md).

---

## Índice

1. [Visão Geral](#1-visão-geral)
2. [Tecnologias Utilizadas](#2-tecnologias-utilizadas)
3. [Instalação e Execução](#3-instalação-e-execução)
4. [Configuração (Variáveis de Ambiente)](#4-configuração-variáveis-de-ambiente)
5. [Estrutura do Código-Fonte](#5-estrutura-do-código-fonte)
6. [Rotas e Páginas](#6-rotas-e-páginas)
7. [Funcionalidades Detalhadas](#7-funcionalidades-detalhadas)
8. [Frontend (Templates | CSS | JavaScript)](#8-frontend-templates--css--javascript)
9. [Comunicação com a API Backend](#9-comunicação-com-a-api-backend)
10. [Histórico de Chat (Persistência Local)](#10-histórico-de-chat-persistência-local)
11. [Tema Claro e Escuro](#11-tema-claro-e-escuro)
12. [Deploy em Produção](#12-deploy-em-produção)
13. [Resolução de Problemas](#13-resolução-de-problemas)

---

## 1. Visão Geral

A **AirData RAG Web Interface** é uma aplicação web que serve como interface gráfica para o sistema de consulta de regulamentações aeronáuticas brasileiras. Ela permite que usuários realizem perguntas em linguagem natural, consultem documentos por busca vetorial, visualizem estatísticas do banco de dados e avaliem a qualidade das respostas geradas.

A aplicação **não executa nenhuma lógica de IA localmente**. Ela funciona exclusivamente como um **frontend** que se comunica com uma API RAG backend via HTTP. Todas as operações de busca vetorial, geração de texto com LLM e gerenciamento de modelos são realizadas pela API backend.

### O que esta aplicação faz:

- Renderiza páginas HTML com templates Jinja2
- Faz proxy de requisições do navegador para a API RAG backend
- Suporta streaming de respostas (Server-Sent Events - SSE)
- Persiste o histórico de conversas localmente em arquivos JSON
- Permite ao usuário avaliar respostas com sistema de estrelas (1-5)
- Suporta tema claro e escuro com persistência via `localStorage`

### O que esta aplicação **não** faz:

- Não executa modelos de linguagem (LLM)
- Não realiza busca vetorial diretamente
- Não se conecta ao Qdrant ou Ollama diretamente
- Não processa ou indexa documentos

---

## 2. Tecnologias Utilizadas

| Tecnologia | Versão | Função |
|-----------|--------|--------|
| **Python** | 3.8+ | Linguagem da aplicação |
| **FastAPI** | 0.109.0 | Framework web assíncrono |
| **Uvicorn** | 0.27.0 | Servidor ASGI (HTTP) |
| **Jinja2** | 3.1.3 | Motor de templates HTML |
| **httpx** | 0.26.0 | Cliente HTTP assíncrono para comunicação com a API backend |
| **Pydantic** | 2.12.5 | Validação de dados e modelos |
| **pydantic-settings** | 2.1.0 | Gerenciamento de configurações via `.env` |
| **python-dotenv** | 1.0.1 | Carregamento de variáveis de ambiente de arquivos `.env` |
| **Loguru** | 0.7.2 | Sistema de logging estruturado |
| **python-multipart** | 0.0.6 | Processamento de formulários HTML (POST com `Form`) |

---

## 3. Instalação e Execução

### 3.1. Criar ambiente virtual

```bash
cd web/
python3 -m venv venv
```

### 3.2. Ativar ambiente virtual

```bash
# Linux / macOS
source venv/bin/activate

# Windows (CMD)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### 3.3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3.4. Configurar variáveis de ambiente

```bash
cp env.example .env
# Edite o arquivo .env conforme necessário (veja Seção 4)
```

### 3.5. Executar a aplicação

```bash
python main.py
```

Alternativamente, usando uvicorn diretamente:

```bash
uvicorn main:app --host 127.0.0.1 --port 8082 --reload
```

### 3.6. Acessar no navegador

```
http://127.0.0.1:8082/ragweb/
```

> A URL inclui o prefixo configurado em `ROOT_PATH`. Ajuste conforme sua configuração.

---

## 4. Configuração (Variáveis de Ambiente)

Todas as configurações são lidas do arquivo `.env` na raiz do diretório `web/`. A classe `Settings` em `app/config.py` carrega essas variáveis automaticamente na inicialização.

### Referência completa de variáveis:

| Variável | Tipo | Padrão | Descrição |
|----------|------|--------|-----------|
| `HOST` | string | `127.0.0.1` | Endereço IP em que o servidor vai escutar. Use `0.0.0.0` para aceitar conexões externas. |
| `PORT` | inteiro | `8082` | Porta HTTP do servidor web. |
| `RELOAD` | booleano | `True` | Se `True`, o servidor reinicia automaticamente ao detectar mudanças nos arquivos (útil em desenvolvimento). |
| `ROOT_PATH` | string | `/ragweb` | Prefixo de URL para quando a aplicação está atrás de um reverse proxy. Caso acesse diretamente, deixe como string vazia `""`. |
| `API_BASE_URL` | string | — | **Obrigatório.** URL completa da API RAG backend (ex: `http://localhost:8000` ou `http://161.24.29.22/ragapi`). |
| `API_KEY` | string | — | **Obrigatório.** Chave de autenticação para a API RAG backend. Enviada no header `X-API-Key`. |
| `APP_NAME` | string | `Aviation RAG Web Interface` | Nome exibido internamente pela aplicação. |
| `APP_VERSION` | string | `1.0.0` | Versão da aplicação. |

#### Autenticacao

| Variavel | Tipo | Padrao | Descricao |
|----------|------|--------|-----------|
| `AUTH_MODE` | string | `api_key` | Modo de autenticacao. Use `api_key` para manter o login estatico; use `drupal_oauth2` para Drupal OAuth2 quando configurado. |
| `SESSION_SECRET_KEY` | string | `API_KEY` | Segredo usado para assinar sessao e cookies. |
| `WEB_LOGIN_ENABLED` | booleano | `True` | Habilita o login estatico quando Drupal OAuth2 nao esta ativo. |
| `WEB_LOGIN_USERNAME` | string | `airdata` | Usuario do login estatico temporario. |
| `WEB_LOGIN_PASSWORD` | string | `AirData-M7q9-V2x4-Kp31` | Senha do login estatico temporario. Trocar ao sair do modo apresentacao. |
| `WEB_LOGIN_TOKEN_TTL_SECONDS` | inteiro | `28800` | Tempo de vida do JWT local, em segundos. |
| `WEB_LOGIN_COOKIE_NAME` | string | `airdata_auth` | Nome do cookie usado pelo login estatico. |
| `DRUPAL_OAUTH_BASE_URL` | string | - | URL base do Drupal, sem barra final. |
| `DRUPAL_OAUTH_CLIENT_ID` | string | - | Client ID do OAuth2 no Drupal. |
| `DRUPAL_OAUTH_CLIENT_SECRET` | string | - | Client secret do OAuth2 no Drupal. |
| `DRUPAL_OAUTH_AUTHORIZE_URL` | string | `<base>/oauth/authorize` | Endpoint de autorizacao OAuth2. |
| `DRUPAL_OAUTH_TOKEN_URL` | string | `<base>/oauth/token` | Endpoint de token OAuth2. |
| `DRUPAL_OAUTH_USERINFO_URL` | string | `<base>/oauth/userinfo` | Endpoint de dados do usuario autenticado. |
| `DRUPAL_OAUTH_CALLBACK_PATH` | string | `/auth/callback` | Callback exposto pela web e registrado no Drupal. |
| `DRUPAL_OAUTH_SCOPES` | string | `openid profile email` | Escopos solicitados durante o login Drupal. |

#### Modos de login

| Cenario | Configuracao principal | Comportamento |
|---------|------------------------|---------------|
| Apresentacao / Drupal indisponivel | `AUTH_MODE=api_key` | Exibe a tela de login estatica e cria um JWT local em cookie HTTP-only. |
| Drupal OAuth2 ativo | `AUTH_MODE=drupal_oauth2` + `DRUPAL_OAUTH_BASE_URL` + `DRUPAL_OAUTH_CLIENT_ID` | Redireciona `/login` para o Drupal e desabilita o login estatico. |

Exemplo para login estatico:

```env
AUTH_MODE=api_key
WEB_LOGIN_ENABLED=true
WEB_LOGIN_USERNAME=airdata
WEB_LOGIN_PASSWORD=troque-esta-senha
SESSION_SECRET_KEY=troque-este-segredo
```

Exemplo para Drupal OAuth2:

```env
AUTH_MODE=drupal_oauth2
SESSION_SECRET_KEY=troque-este-segredo
DRUPAL_OAUTH_BASE_URL=https://www.airdata.ita.br
DRUPAL_OAUTH_CLIENT_ID=id-chat
DRUPAL_OAUTH_CLIENT_SECRET=<secret>
DRUPAL_OAUTH_AUTHORIZE_URL=https://www.airdata.ita.br/oauth/authorize
DRUPAL_OAUTH_TOKEN_URL=https://www.airdata.ita.br/oauth/token
DRUPAL_OAUTH_USERINFO_URL=https://www.airdata.ita.br/oauth/userinfo
```

### Exemplo de arquivo `.env`:

```env
HOST=127.0.0.1
PORT=8082
RELOAD=True
ROOT_PATH=/ragweb

API_BASE_URL=http://161.24.29.22/ragapi
API_KEY=L2B7zzrla7pV0Ro2Bc5ipwxf3-H8EKwxWz0kX1qMsa0

APP_NAME=Aviation RAG Web Interface
APP_VERSION=1.0.0
```

### Como a configuração é carregada

O arquivo `app/config.py` define uma classe `Settings` que herda de `pydantic_settings.BaseSettings`. Na inicialização, o `dotenv` carrega as variáveis do arquivo `.env` e a classe as disponibiliza como atributos tipados:

```python
from app.config import settings

print(settings.API_BASE_URL)  # http://161.24.29.22/ragapi
print(settings.PORT)          # 8082
```

O objeto `settings` é importado em `main.py` e utilizado para configurar o servidor, o cliente HTTP e os metadados da aplicação.

---

## 5. Estrutura do Código-Fonte

### `main.py` — Aplicação Principal

Este é o **único arquivo de código da aplicação**. Ele contém:

- **Inicialização do FastAPI** — Cria a instância `app` com nome, versão e `root_path`
- **Montagem de arquivos estáticos** — Serve CSS, JS e imagens de `static/`
- **Configuração de templates** — Inicializa Jinja2 apontando para `templates/`
- **Cliente HTTP** — Instância `httpx.AsyncClient` com timeout de 180 segundos para comunicação com a API backend
- **Rotas de páginas** — Endpoints GET que renderizam templates HTML
- **Rotas de API (proxy)** — Endpoints que fazem proxy de requisições para a API backend
- **Persistência de histórico** — Função `_save_chat_history()` que salva conversas em JSON
- **Servidor** — Bloco `if __name__ == "__main__"` que inicia o Uvicorn

### `app/config.py` — Configurações

Classe `Settings` que lê variáveis de ambiente e as disponibiliza de forma tipada e validada.

### `templates/` — Templates HTML

Templates Jinja2 com herança. Todos herdam de `base.html`, que define:
- Estrutura HTML base (`<head>`, `<body>`)
- Header com navegação e logo
- Botão de alternância de tema (claro/escuro)
- Footer institucional
- Blocos extensíveis: `title`, `extra_styles`, `content`, `extra_scripts`

### `static/` — Arquivos Estáticos

- `css/` — Um arquivo CSS por página, mais `base.css` (global) e `custom.css` (personalizações)
- `js/app.js` — JavaScript utilitário (validação de formulários, botões de cópia, highlight)
- `images/` — Logos e imagens

### `chat_history/` — Histórico Persistente

Diretório criado automaticamente. Cada sessão de chat gera um arquivo JSON com:
- ID da sessão
- Título (gerado automaticamente da primeira mensagem do usuário)
- Lista de mensagens (role, content, timestamp, ratings)
- Metadados (modelo usado, fontes consultadas, tempo de processamento)

---

## 6. Rotas e Páginas

### Rotas de Páginas (retornam HTML)

| Método | Rota | Função | Template | Descrição |
|--------|------|--------|----------|-----------|
| GET | `/` | `home()` | `index.html` | Página inicial com cards de funcionalidades |
| GET | `/chat` | `chat_page()` | `chat.html` | Interface de chat com LLM. Carrega lista de modelos da API |
| GET | `/pesquisa` | `search_page()` | `search.html` | Formulário de busca vetorial |
| POST | `/pesquisa` | `search_post()` | `search.html` | Processamento da busca vetorial (envia para API e exibe resultados) |
| GET | `/estatisticas` | `stats_page()` | `stats.html` | Estatísticas do banco vetorial (carrega dados da API) |
| GET | `/sobre` | `about_page()` | `about.html` | Página informativa sobre o sistema |

### Rotas de API (proxy para o backend, retornam JSON)

| Método | Rota | Função | Descrição |
|--------|------|--------|-----------|
| POST | `/api/chat/send` | `send_chat_message()` | Envia mensagem para o chat (resposta completa) |
| POST | `/api/chat/stream` | `stream_chat_message()` | Envia mensagem com streaming (SSE) |
| POST | `/api/chat/feedback` | `submit_feedback()` | Registra feedback (thumbs / star / comment / clear) |
| POST | `/api/chat/rate` | `rate_message()` | **Deprecated** — alias legado de `/api/chat/feedback` (mapeia para `kind=star`) |
| POST | `/api/models/change` | `change_model_proxy()` | Troca o modelo LLM ativo |
| GET | `/api/models` | `get_models_proxy()` | Lista modelos LLM disponíveis |
| GET | `/api/chat/history` | `get_chat_history()` | Lista todas as sessões salvas |
| GET | `/api/chat/history/{session_id}` | `get_session_history()` | Carrega uma sessão específica |
| GET | `/health` | `health()` | Health check do servidor |

---

## 7. Funcionalidades Detalhadas

### 7.1. Chat com LLM

A página de chat (`/chat`) oferece uma experiência conversacional completa:

**Interface:**
- Campo de texto com auto-resize para digitação de mensagens
- Botão de envio (também funciona com Enter; Shift+Enter para nova linha)
- Indicador de digitação animado enquanto aguarda resposta
- Mensagens exibidas em bolhas estilizadas (usuário à direita, assistente à esquerda)

**Sidebar:**
- Seletor de modelo LLM (dropdown carregado da API)
- Toggle "Usar RAG" — quando ativado, o sistema busca documentos relevantes antes de gerar a resposta
- Botão "Nova Conversa" — limpa a conversa atual
- Histórico de conversas — lista sessões anteriores, clicáveis para recarregar

**Streaming:**
- As respostas do LLM são recebidas via Server-Sent Events (SSE)
- Cada token é exibido em tempo real na interface, palavra por palavra
- Ao final do streaming, as fontes e metadados são adicionados à mensagem

**Fontes:**
- Quando RAG está ativado, as fontes consultadas são exibidas abaixo da resposta
- Cada fonte é clicável e abre um modal com o texto completo do trecho
- Exibe o `regulation_id`, score de relevância e preview do texto

**Avaliação:**
- Cada resposta do assistente pode ser avaliada em 5 categorias com estrelas (1-5):
  - Precisão Factual
  - Completude
  - Clareza
  - Qualidade das Citações
  - Relevância
- As avaliações são salvas no histórico JSON local

**Troca de Modelo:**
- O usuário pode trocar o modelo LLM a qualquer momento pelo dropdown
- Uma mensagem de sistema é exibida confirmando a troca
- A troca é enviada para a API backend via proxy

### 7.2. Busca de Regulamentações

A página de busca (`/pesquisa`) realiza consultas diretas ao banco vetorial, **sem utilizar nenhum modelo de linguagem**. É uma busca puramente vetorial (por similaridade semântica):

**Formulário:**
- Campo de texto para a pergunta (obrigatório)
- Campo de data (opcional) — filtra regulamentações vigentes em uma data específica
- Seletor de número de resultados (3, 5, 10 ou 15)
- Campo de score mínimo (0.0 a 1.0) — filtra resultados abaixo de um limiar de relevância

**Resultados:**
- Estatísticas de performance (tempo de busca em ms, total de resultados)
- Lista de documentos encontrados, cada um exibindo:
  - ID da regulamentação
  - Score de relevância
  - Texto do trecho relevante
  - Metadados (versão, data efetiva, data de expiração)
  - Metadados completos em formato JSON (expansível)

**Diferença para o Chat:**
- O chat usa RAG completo: busca vetorial + geração de resposta com LLM
- A busca usa apenas a busca vetorial: retorna os trechos mais similares sem processar com LLM
- A busca é significativamente mais rápida por não envolver geração de texto

### 7.3. Estatísticas

A página de estatísticas (`/estatisticas`) exibe informações em tempo real do sistema:

- Status do sistema (healthy/error)
- Total de vetores indexados no Qdrant
- Total de pontos no banco
- Taxa de indexação
- Contagem de documentos por tipo (leis, ICAs, RBACs, portarias, etc.)
- Contagem de documentos por fonte (LexML, DECEA)

### 7.4. Tema Claro e Escuro

- Botão de alternância no header (ícone lua/sol)
- Preferência salva no `localStorage` do navegador
- Atributo `data-theme="dark"` aplicado no `<html>`
- Cada arquivo CSS possui regras específicas para `[data-theme="dark"]`

---

## 8. Frontend (Templates | CSS | JavaScript)

### 8.1. Sistema de Templates (Jinja2)

Todos os templates herdam de `base.html` usando `{% extends "base.html" %}`. Os blocos disponíveis são:

| Bloco | Descrição |
|-------|-----------|
| `{% block title %}` | Título da aba do navegador |
| `{% block extra_styles %}` | CSS adicional específico da página |
| `{% block content %}` | Conteúdo principal da página |
| `{% block extra_scripts %}` | JavaScript adicional específico da página |

### 8.2. Arquivos CSS

Cada página possui seu próprio arquivo CSS para organização:

| Arquivo | Escopo |
|---------|--------|
| `base.css` | Variáveis CSS, reset, header, footer, navegação, tipografia, tema escuro global |
| `home.css` | Cards de funcionalidades, hero section |
| `chat.css` | Mensagens, sidebar, formulário de input, rating com estrelas, modal de fontes |
| `search.css` | Formulário de busca, cards de resultado, scores |
| `stats.css` | Grid de estatísticas, cards de métricas |
| `about.css` | Seções de conteúdo, lista de funcionalidades |
| `custom.css` | Personalizações adicionais |

**Variáveis CSS globais** são definidas em `base.css` e controlam cores, espaçamentos e bordas para ambos os temas. Os seletores `[data-theme="dark"]` ou `html[data-theme="dark"]` são usados em cada CSS para sobrescrever estilos no modo escuro.

### 8.3. JavaScript

O JavaScript está distribuído em dois locais:

1. **`static/js/app.js`** — Funções utilitárias compartilhadas:
   - Validação do formulário de busca
   - Auto-refresh da página de estatísticas
   - Botões de cópia para textos de fontes
   - Highlight de termos de busca nos resultados

2. **`templates/chat.html`** (inline no `{% block extra_scripts %}`) — Lógica completa do chat:
   - Objeto `ChatApp` com todo o estado e métodos do chat
   - Comunicação com endpoints `/api/chat/stream` via `fetch` + `ReadableStream`
   - Manipulação do DOM (criação de mensagens, indicador de digitação, scroll)
   - Gerenciamento de sessões e histórico
   - Sistema de avaliação por estrelas
   - Modal de fontes

---

## 9. Comunicação com a API Backend

A aplicação web se comunica com a API RAG backend usando `httpx.AsyncClient`. Todas as requisições incluem o header `X-API-Key` com a chave configurada em `.env`.

### Fluxo de uma pergunta no Chat:

```
1. Usuário digita mensagem → JS captura o submit
2. JS envia POST para /api/chat/stream (endpoint local)
3. main.py recebe a requisição no endpoint stream_chat_message()
4. main.py faz proxy: POST para API_BASE_URL/api/chat/stream (API backend)
5. API backend processa (busca vetorial + LLM) e retorna SSE
6. main.py retransmite o stream (StreamingResponse) para o navegador
7. JS processa cada evento SSE e atualiza a interface em tempo real
8. Ao receber evento "done", main.py salva no histórico local (JSON)
```

### Fluxo de uma busca por regulamentação:

```
1. Usuário preenche formulário e submete → POST /pesquisa
2. main.py recebe os dados do formulário (query, date, limit, score_threshold)
3. main.py faz POST para API_BASE_URL/api/vector-search (busca vetorial pura)
4. API backend consulta o Qdrant e retorna os resultados
5. main.py renderiza search.html com os resultados recebidos
```

### Timeout

O cliente HTTP (`httpx.AsyncClient`) é configurado com timeout de **180 segundos** (3 minutos), pois consultas RAG que envolvem geração com LLM podem demorar, especialmente com modelos maiores.

---

## 10. Histórico de Chat (Persistência Local)

As conversas são salvas em arquivos JSON no diretório `chat_history/`. O diretório é criado automaticamente na inicialização da aplicação.

### Estrutura de um arquivo de histórico:

```json
{
  "session_id": "uuid-da-sessao",
  "title": "Primeiras palavras da primeira mensagem...",
  "model": "nome-do-modelo-usado",
  "created_at": "2026-02-25T14:30:00.000000",
  "last_updated": "2026-02-25T14:35:00.000000",
  "messages": [
    {
      "role": "user",
      "content": "Texto da mensagem do usuário",
      "timestamp": "2026-02-25T14:30:00.000000"
    },
    {
      "role": "assistant",
      "content": "Texto da resposta do assistente",
      "timestamp": "2026-02-25T14:30:05.000000",
      "message_id": "msg_1740000000000_abc12345",
      "model": "nome-do-modelo",
      "use_rag": true,
      "sources": [
        {
          "regulation_id": "RBAC 61",
          "text": "Trecho do documento...",
          "score": 0.85
        }
      ],
      "processing_time_ms": 3500,
      "ratings": {
        "factual_accuracy": 4,
        "completeness": 5,
        "clarity": 4,
        "citation_quality": 3,
        "relevance": 5
      }
    }
  ]
}
```

### Título automático

O título da sessão é gerado automaticamente a partir da primeira mensagem do usuário, truncada em 50 caracteres. A função `_generate_title_from_messages()` é responsável por essa lógica.

### Avaliação de mensagens

O feedback explícito é capturado pela UI do chat (seção "Essa resposta foi útil?")
e persistido em **dois lugares simultaneamente**:

1. **JSON da sessão** (`web/chat_history/{session_id}.json`, campos `ratings`
   e `feedback` da mensagem) — mantém o replay da sessão funcionando e é o
   que a UI lê quando você abre uma conversa antiga.
2. **SQLite operacional** (`data/app.db`, tabela `feedback_events`) — append-only,
   consultável via Datasette em `/explore/app/`. View `feedback_current`
   expõe o último estado por mensagem.

Ver também: [`docs/migrations/2026-05-03_FEEDBACK_SQLITE.md`](../docs/migrations/2026-05-03_FEEDBACK_SQLITE.md).

**Sinais capturados (UI):**

| Sinal | Descrição |
|---|---|
| 👍 / 👎 | Primário, 1 clique. Thumbs-down abre seletor de motivo. |
| 💬 Comentar | Texto livre opcional. |
| ⭐ Detalhes | Grid de 5 estrelas por categoria (opcional). Clique na estrela preenchida mais à direita desfaz a avaliação daquela categoria. |

**Categorias de estrelas:**

| Categoria | Chave | Descrição |
|-----------|-------|-----------|
| Precisão | `factual_accuracy` | Informações corretas e sem alucinações. |
| Completude | `completeness` | A resposta cobre todos os aspectos. |
| Clareza | `clarity` | Resposta clara e bem estruturada. |
| Citações | `citation_quality` | Citou corretamente as normas (oculto quando não há fontes). |
| Relevância | `relevance` | Manteve-se no tema solicitado. |

**Taxonomia de motivos (thumbs-down):**
`hallucination`, `incomplete`, `off_topic`, `wrong_citation`, `unclear`, `other`.

**Backfill do histórico antigo:** `make feedback-backfill` importa as
avaliações existentes nos JSONs legados para `feedback_events` (idempotente).

---

## 11. Tema Claro e Escuro

O sistema de temas é implementado com CSS puro e JavaScript mínimo:

### Como funciona:

1. O `base.html` inclui um `<button id="themeToggle">` no header
2. O script no final de `base.html` gerencia a alternância:
   - Lê preferência do `localStorage`
   - Aplica atributo `data-theme` no `<html>`
   - Altera ícone (🌙 para claro, ☀️ para escuro)
3. Os arquivos CSS usam seletores `html[data-theme="dark"]` para sobrescrever estilos

### Adicionando suporte a tema escuro em novos componentes:

```css
/* Estilos padrão (tema claro) */
.meu-componente {
  background: #ffffff;
  color: #212529;
}

/* Estilos para tema escuro */
html[data-theme="dark"] .meu-componente {
  background: var(--bg-secondary);
  color: white;
}
```

---

## 12. Deploy em Produção

### 12.1. Com Reverse Proxy (Nginx)

Em produção, a aplicação tipicamente roda atrás de um Nginx como reverse proxy:

```nginx
server {
    listen 443 ssl;
    server_name seu-dominio.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # Proxy para a aplicação web
    location /ragweb/ {
        proxy_pass http://127.0.0.1:8082;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Suporte a SSE (streaming)
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }
}
```

### 12.2. Com systemd (Linux)

Para manter a aplicação rodando como serviço:

```ini
# /etc/systemd/system/ragweb.service
[Unit]
Description=AirData RAG Web Interface
After=network.target

[Service]
Type=simple
User=seu-usuario
WorkingDirectory=/caminho/para/web
Environment="PATH=/caminho/para/web/venv/bin"
ExecStart=/caminho/para/web/venv/bin/python main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Ative o serviço:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ragweb
sudo systemctl start ragweb
sudo systemctl status ragweb
```

### 12.3. Configurações recomendadas para produção

No arquivo `.env`:

```env
HOST=127.0.0.1       # Escutar apenas localmente (Nginx faz o proxy)
PORT=8082
RELOAD=False          # Desativar reload automático em produção
ROOT_PATH=/ragweb     # Ajustar conforme configuração do Nginx
```

---

## 13. Resolução de Problemas

### A aplicação não inicia

| Sintoma | Causa Provável | Solução |
|--------|----------------|---------|
| `ModuleNotFoundError` | Dependências não instaladas ou venv não ativado | `source venv/bin/activate && pip install -r requirements.txt` |
| `Address already in use` | Porta já ocupada por outro processo | Altere `PORT` no `.env` ou finalize o processo anterior: `lsof -i :8082` |
| `KeyError` ou `TypeError` em config | Variável de ambiente ausente no `.env` | Verifique se todas as variáveis do `env.example` estão definidas no `.env` |

### Páginas com erro

| Sintoma | Causa Provável | Solução |
|--------|----------------|---------|
| "API Error: 5xx" em busca ou estatísticas | A API RAG backend está com erro | Verifique os logs do servidor da API backend |
| "Connection refused" | API RAG backend não está rodando | Inicie o servidor da API ou verifique `API_BASE_URL` |
| "API Error: 401" | Chave de API incorreta | Corrija `API_KEY` no `.env` |
| Timeout no chat | Modelo LLM lento ou indisponível | Verifique se o Ollama está rodando e o modelo está carregado |

### Chat não funciona

| Sintoma | Causa Provável | Solução |
|--------|----------------|---------|
| Mensagem não envia | JS com erro no console do navegador | Abra DevTools (F12) e verifique erros no console |
| Streaming para abruptamente | Timeout na conexão | Verifique conectividade com a API e logs do servidor |
| Histórico não carrega | Arquivos JSON corrompidos em `chat_history/` | Remova arquivos corrompidos do diretório |
| Avaliação não salva | Sessão não encontrada | Verifique se o `session_id` existe em `chat_history/` |

### Verificação de conectividade

Para testar manualmente se a API backend está acessível:

```bash
# Health check da API backend
curl -H "X-API-Key: SUA_CHAVE" http://SEU_API_BASE_URL/stats

# Listar modelos disponíveis
curl -H "X-API-Key: SUA_CHAVE" http://SEU_API_BASE_URL/api/models

# Busca vetorial de teste
curl -X POST -H "X-API-Key: SUA_CHAVE" -H "Content-Type: application/json" \
  -d '{"query": "teste", "limit": 1}' \
  http://SEU_API_BASE_URL/api/vector-search
```

---

## Referências

- [START_HERE.md](START_HERE.md) — Introdução e arquitetura
- [QUICKSTART.md](QUICKSTART.md) — Guia rápido de execução
- [Documentação FastAPI](https://fastapi.tiangolo.com)
- [Documentação Jinja2](https://jinja.palletsprojects.com)
- [Portal AirData](https://www.airdata.ita.br)
- [GitHub AirData](https://github.com/ita-airdata)
