# Comece Aqui - Aviation RAG System

> **Leia este documento primeiro.** Ele apresenta uma visão geral do projeto, explica como cada componente se encaixa e direciona você para a documentação específica de cada parte.

---

## O que é este projeto?

O **Aviation RAG System** é uma plataforma de inteligência artificial especializada em regulamentações da aviação civil brasileira. O sistema utiliza a técnica **RAG (Retrieval-Augmented Generation)** para permitir que usuários façam perguntas em linguagem natural sobre normas aeronáuticas e recebam respostas fundamentadas em documentos oficiais da ANAC e do DECEA.

O projeto é composto por múltiplos componentes que trabalham juntos:

- **Scrapers e Parsers** — Extraem documentos normativos de fontes oficiais (LexML, DECEA)
- **Pipeline de Ingestão** — Processa, divide em chunks e indexa os documentos no banco vetorial
- **Banco Vetorial (Qdrant)** — Armazena embeddings dos documentos para busca semântica
- **Modelo de Embeddings** — Legal-BERTimbau, modelo treinado para textos jurídicos em português
- **LLM (Ollama)** — Modelos de linguagem locais para geração de respostas
- **API RAG** — Servidor FastAPI que expõe endpoints de chat, busca e estatísticas
- **Interface Web** — Frontend FastAPI/Jinja2 para interação do usuário

---

## Arquitetura Geral

```
┌────────────────────────────────────────────────────────────────────┐
│                           FONTES DE DADOS                          │
│                                                                    │
│   LexML Brasil ─────┐        DECEA Portal ──────┐                 │
│   (Legislação)       │        (ICA, MCA, etc.)    │                │
│                      ▼                            ▼                │
│             lexml_scraper.py            decea_scraper.py           │
│                      │                            │                │
│                      ▼                            ▼                │
│               data/lexml/*.json            data/decea/*.json       │
└──────────────────────┬────────────────────────────┬────────────────┘
                       │                            │
                       ▼                            ▼
            ┌──────────────────────────────────────────────┐
            │         PIPELINE DE INGESTÃO                  │
            │                                              │
            │   1. Parsing (LexMLParser / PDFParser)        │
            │   2. Chunking (ArticleChunker / ICAChunker)   │
            │   3. Embedding (Legal-BERTimbau)              │
            │   4. Upload para Qdrant                       │
            └──────────────────┬───────────────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────────────────┐
            │              QDRANT                           │
            │         (Banco Vetorial)                      │
            │                                              │
            │   Coleção: aviation_regulations               │
            │   Vetores: 1024 dimensões (cosine)           │
            │   Índices: regulation_id, effective_date,     │
            │            expiry_date, source, doc_type      │
            └──────────────────┬───────────────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────────────────┐
            │              API RAG                          │
            │         (FastAPI + Uvicorn)                    │
            │                                              │
            │   POST /api/chat          → Chat com LLM      │
            │   POST /api/chat/stream   → Chat SSE          │
            │   POST /api/vector-search → Busca vetorial     │
            │   POST /api/search-regulations → RAG completo  │
            │   GET  /api/models        → Listar modelos     │
            │   POST /api/models/change → Trocar modelo      │
            │   GET  /stats             → Estatísticas       │
            └──────────────────┬───────────────────────────┘
                               │
                          Ollama (LLM)
                               │
                               ▼
            ┌──────────────────────────────────────────────┐
            │           INTERFACE WEB                       │
            │      (FastAPI + Jinja2 + Uvicorn)             │
            │                                              │
            │   /           → Página inicial                │
            │   /chat       → Chat com LLM                  │
            │   /pesquisa   → Busca vetorial (sem LLM)      │
            │   /estatisticas → Estatísticas                │
            │   /sobre      → Sobre o sistema               │
            └──────────────────────────────────────────────┘
```

---

## Estrutura de Diretórios

```
aviation-rag-system/
│
├── config.py                # Configuração central (lê o .env)
├── .env                     # Variáveis de ambiente (não commitado)
├── env.example              # Exemplo de .env
├── requirements.txt         # Dependências Python do sistema principal
│
├── api/                     # Servidor da API RAG
│   ├── server.py            # Aplicação FastAPI (endpoints)
│   ├── auth.py              # Autenticação por API Key
│   ├── schemas.py           # Schemas Pydantic (request/response)
│   └── session_manager.py   # Gerenciador de sessões de chat
│
├── models/                  # Wrappers de modelos de IA
│   ├── embeddings.py        # Legal-BERTimbau (sentence-transformers)
│   └── llm.py               # Ollama/Llama (geração de texto)
│
├── database/                # Banco vetorial
│   ├── qdrant_manager.py    # CRUD no Qdrant
│   └── versioning.py        # Versionamento de documentos
│
├── search/                  # Lógica de busca
│   ├── vector_search.py     # Busca vetorial (semântica + temporal)
│   └── rag.py               # Pipeline RAG (busca + geração)
│
├── parsers/                 # Extração e parsing de documentos
│   ├── decea_scraper.py     # Scraper do portal DECEA (Selenium)
│   ├── lexml_scraper.py     # Scraper do portal LexML (requests)
│   ├── lexml_parser.py      # Parser de XMLs LexML
│   ├── pdf_parser.py        # Parser de PDFs
│   ├── ocr_processor.py     # OCR para PDFs escaneados
│   ├── temporal_extractor.py # Extração de datas de vigência
│   ├── document_tracker.py  # Rastreamento de documentos (anti-duplicata)
│   └── document_counter.py  # Contagem de documentos por tipo
│
├── pipeline/                # Pipeline de processamento
│   ├── chunking.py          # Divisão de textos em chunks
│   └── ingestion.py         # Pipeline completo de ingestão
│
├── scripts/                 # Scripts utilitários
│   ├── collect.py           # Fase 1: coletar documentos (make collect)
│   ├── embed.py             # Fase 2: gerar embeddings (make embed)
│   ├── index.py             # Fase 3: indexar no Qdrant (make index)
│   ├── query.py             # Console SQL interativo (make query)
│   ├── inspect_qdrant.py    # Inspecionar dados do Qdrant
│   ├── validate_data.py     # Validar qualidade dos dados
│   ├── test_system.py       # Testar todos os componentes
│   └── test_chatbot.py      # Testar endpoints do chatbot
│
├── data/                    # Dados extraídos
│   ├── store.db             # SQLite com documentos coletados
│   ├── embeddings/          # Parquet com vetores (dense/sparse)
│   ├── originals/           # PDFs originais baixados
│   └── pdfs/                # PDFs locais para ingestão
│
├── web/                     # Interface Web (projeto separado)
│   ├── main.py              # Aplicação web FastAPI
│   ├── requirements.txt     # Dependências da web
│   ├── app/                 # Configuração
│   ├── templates/           # Templates HTML (Jinja2)
│   ├── static/              # CSS, JS, imagens
│   ├── chat_history/        # Histórico de conversas (JSON)
│   └── docs/                # Documentação da web
│
└── docs/                    # Documentação geral (você está aqui)
    ├── START_HERE.md         # Este arquivo
    ├── QUICKSTART.md         # Guia rápido
    └── README.md             # Documentação completa
```

---

## Pré-requisitos Globais

| Requisito | Versão | Descrição |
|-----------|--------|-----------|
| **Python** | 3.8+ | Linguagem principal do projeto |
| **Qdrant** | 1.x | Banco de dados vetorial (rodando como serviço ou Docker) |
| **Ollama** | 0.1+ | Servidor local de LLMs (para chat e geração de respostas) |
| **CUDA** (opcional) | 11.x+ | Para aceleração GPU no modelo de embeddings |
| **Chrome/Chromium** (opcional) | — | Necessário para o scraper DECEA (Selenium) |

---

## Próximos Passos

| Objetivo | Documento |
|----------|-----------|
| Quero **rodar o projeto o mais rápido possível** | Leia o [QUICKSTART.md](QUICKSTART.md) |
| Quero **entender tudo em profundidade** | Leia o [README.md](README.md) |
| Quero **entender apenas a interface web** | Leia o [web/docs/](../web/docs/START_HERE.md) |

---

## Contato

Projeto AirData — Instituto Tecnológico de Aeronáutica (ITA)

- Portal: [https://www.airdata.ita.br](https://www.airdata.ita.br)
- GitHub: [https://github.com/ita-airdata](https://github.com/ita-airdata)
