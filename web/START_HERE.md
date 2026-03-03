# Comece Aqui - AirData RAG Web Interface

> **Leia este documento primeiro.** Ele explica o que é este projeto, como ele se encaixa na arquitetura geral do sistema AirData RAG e quais são os pré-requisitos para executá-lo.

---

## O que é este projeto?

Este diretório (`web/`) contém a **interface web** do sistema AirData RAG. Trata-se de uma aplicação web construída com **FastAPI** (Python) que fornece ao usuário final uma interface gráfica para interagir com o sistema de consulta de regulamentações aeronáuticas brasileiras.

A aplicação web **não processa dados por conta própria**. Ela atua como um **frontend** que se comunica via HTTP com uma **API RAG backend** separada. Toda a lógica de busca vetorial, geração de respostas com LLM e gerenciamento de modelos é realizada pela API backend — este projeto apenas disponibiliza uma interface amigável para o usuário.

---

## Arquitetura Geral

```
┌──────────────────────────────────────────────────────────┐
│                      USUÁRIO                             │
│                   (Navegador Web)                        │
└─────────────────────┬────────────────────────────────────┘
                      │ HTTP (porta configurável)
                      ▼
┌──────────────────────────────────────────────────────────┐
│              WEB APP (Este projeto)                       │
│                                                          │
│  Framework: FastAPI + Jinja2 + Uvicorn                   │
│  Função: Interface gráfica, proxy de requisições,        │
│          persistência de histórico de chat                │
│                                                          │
│  Páginas:                                                │
│    /           → Página inicial                          │
│    /chat       → Chat com LLM (streaming)                │
│    /pesquisa   → Busca vetorial (sem LLM)                │
│    /estatisticas → Estatísticas do sistema               │
│    /sobre      → Informações sobre o projeto             │
│    /health     → Health check (JSON)                     │
└─────────────────────┬────────────────────────────────────┘
                      │ HTTP + API Key
                      ▼
┌──────────────────────────────────────────────────────────┐
│              API RAG (Projeto separado)                   │
│                                                          │
│  Endpoints consumidos:                                   │
│    POST /api/chat          → Chat com LLM                │
│    POST /api/chat/stream   → Chat com streaming (SSE)    │
│    POST /api/vector-search → Busca vetorial pura         │
│    GET  /api/models        → Lista de modelos LLM        │
│    POST /api/models/change → Trocar modelo LLM           │
│    GET  /stats             → Estatísticas do Qdrant      │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│        Qdrant (Banco Vetorial) + Ollama (LLM)            │
└──────────────────────────────────────────────────────────┘
```

### Importante

- A **API RAG backend** precisa estar rodando e acessível para que esta aplicação web funcione corretamente.
- Sem a API backend, as páginas de chat, busca e estatísticas apresentarão erros de conexão.
- A página inicial (`/`) e a página sobre (`/sobre`) funcionam de forma independente, pois são páginas estáticas.

---

## Pré-requisitos

Antes de executar esta aplicação, certifique-se de que você possui:

### Obrigatórios

| Requisito | Versão Mínima | Descrição |
|-----------|---------------|-----------|
| **Python** | 3.8+ | Interpretador Python |
| **pip** | 20.0+ | Gerenciador de pacotes Python |
| **API RAG Backend** | — | Servidor da API RAG rodando e acessível na rede |

### Recomendados

| Requisito | Descrição |
|-----------|-----------|
| **venv** | Módulo de ambientes virtuais do Python (vem incluso no Python 3.3+) |
| **Git** | Para controle de versão |

---

## Estrutura de Arquivos

```
web/
├── main.py              # Ponto de entrada da aplicação (execute este arquivo)
├── requirements.txt     # Dependências Python necessárias
├── .env                 # Variáveis de ambiente (configuração local)
├── env.example          # Exemplo de .env para referência
├── gitignore            # Arquivo gitignore do projeto
│
├── app/                 # Módulo de configuração da aplicação
│   ├── __init__.py      # Exporta o objeto 'settings'
│   └── config.py        # Classe Settings (lê variáveis do .env)
│
├── templates/           # Templates HTML (Jinja2)
│   ├── base.html        # Template base (header, footer, tema, navegação)
│   ├── index.html       # Página inicial
│   ├── chat.html        # Página de chat (inclui toda a lógica JS do chat)
│   ├── search.html      # Página de busca vetorial
│   ├── stats.html       # Página de estatísticas
│   └── about.html       # Página sobre o sistema
│
├── static/              # Arquivos estáticos
│   ├── css/             # Folhas de estilo por página
│   │   ├── base.css     # Estilos globais, tema claro/escuro, navegação
│   │   ├── home.css     # Estilos da página inicial
│   │   ├── chat.css     # Estilos do chat (mensagens, sidebar, rating, modal)
│   │   ├── search.css   # Estilos da página de busca
│   │   ├── stats.css    # Estilos da página de estatísticas
│   │   ├── about.css    # Estilos da página sobre
│   │   └── custom.css   # Estilos personalizados adicionais
│   ├── js/
│   │   └── app.js       # JavaScript utilitário (validação, highlight, etc.)
│   └── images/          # Imagens e logos
│
├── chat_history/        # Histórico de conversas (criado automaticamente)
│   └── *.json           # Arquivos JSON por sessão de chat
│
└── docs/                # Documentação (você está aqui)
    ├── START_HERE.md     # Este arquivo
    ├── QUICKSTART.md     # Guia rápido de execução
    └── README.md         # Documentação completa e detalhada
```

---

## Próximos Passos

| Objetivo | Documento |
|----------|-----------|
| Quero **rodar o projeto o mais rápido possível** | Leia o [QUICKSTART.md](QUICKSTART.md) |
| Quero **entender o projeto em profundidade** | Leia o [README.md](README.md) |

---

## Contato

Projeto AirData — Instituto Tecnológico de Aeronáutica (ITA)

- Portal: [https://www.airdata.ita.br](https://www.airdata.ita.br)
- GitHub: [https://github.com/ita-airdata](https://github.com/ita-airdata)
