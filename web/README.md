# Aviation RAG Web Interface 🛩️

Interface web desenvolvida em FastAPI para interagir com o sistema RAG (Retrieval-Augmented Generation) de regulamentações aeronáuticas.

## 📋 Visão Geral

Esta aplicação web fornece uma interface amigável para buscar e consultar regulamentações aeronáuticas utilizando tecnologia RAG. O sistema se comunica com uma API backend que realiza busca vetorial e geração de respostas usando LLM.

### Funcionalidades

- 🔍 **Busca Inteligente**: Faça perguntas em linguagem natural sobre regulamentações
- 📚 **Fontes Verificáveis**: Visualize as fontes exatas com scores de relevância
- 📅 **Filtragem Temporal**: Busque regulamentações vigentes em datas específicas
- 📊 **Estatísticas em Tempo Real**: Monitore métricas do sistema
- ⚡ **Interface Responsiva**: Design moderno e responsivo usando o padrão AirData

## 🏗️ Arquitetura

```
┌─────────────┐
│   Usuário   │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   Web App       │ (FastAPI - Este projeto)
│   - Interface   │
│   - Templates   │
└────────┬────────┘
         │ HTTP + API Key
         ▼
┌─────────────────┐
│   API RAG       │ (FastAPI - Separada)
│   - /search     │
│   - /stats      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Qdrant + LLM  │
└─────────────────┘
```

## 📁 Estrutura do Projeto

```
rag-web-app/
│
├── app/
│   ├── __init__.py          # Inicialização do módulo
│   └── config.py            # Configurações da aplicação
│
├── static/                  # Arquivos estáticos
│   ├── css/                 # Estilos CSS (se necessário)
│   ├── js/                  # Scripts JavaScript (se necessário)
│   └── images/              # Imagens e logos
│
├── templates/               # Templates Jinja2
│   ├── base.html           # Template base
│   ├── index.html          # Página inicial
│   ├── search.html         # Página de busca
│   ├── stats.html          # Página de estatísticas
│   └── about.html          # Página sobre
│
├── main.py                  # Aplicação principal FastAPI
├── requirements.txt         # Dependências Python
├── .env.example            # Exemplo de variáveis de ambiente
├── .gitignore              # Arquivos ignorados pelo Git
└── README.md               # Este arquivo
```

## 🚀 Instalação e Configuração

### Pré-requisitos

- Python 3.8+
- API RAG rodando (veja repositório da API)
- Qdrant configurado

### Passo a Passo

1. **Clone o repositório**

```bash
git clone https://github.com/ita-airdata/rag-web-app.git
cd rag-web-app
```

2. **Crie um ambiente virtual**

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. **Instale as dependências**

```bash
pip install -r requirements.txt
```

4. **Configure as variáveis de ambiente**

```bash
cp .env.example .env
```

Edite o arquivo `.env` com suas configurações:

```env
# Web Server Configuration
HOST=0.0.0.0
PORT=8001
RELOAD=True

# API Configuration
API_BASE_URL=http://localhost:8000
API_KEY=sua-chave-api-aqui

# Application
APP_NAME=Aviation RAG Web Interface
APP_VERSION=1.0.0
```

5. **Inicie a aplicação**

```bash
python main.py
```

Ou usando uvicorn diretamente:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

6. **Acesse a aplicação**

Abra seu navegador em: `http://localhost:8001`

## 📖 Uso

### Página Inicial

A página inicial apresenta uma visão geral do sistema e suas funcionalidades principais.

### Buscar Regulamentações

1. Acesse `/search`
2. Digite sua pergunta em linguagem natural
3. (Opcional) Configure filtros:
   - Data específica
   - Número de resultados
   - Score mínimo de relevância
4. Clique em "Buscar"
5. Visualize a resposta e as fontes utilizadas

**Exemplo de perguntas:**

- "Quais são os requisitos para certificação de pilotos comerciais?"
- "Como deve ser feita a manutenção de aeronaves comerciais?"
- "Quais são as regras para voos noturnos?"

### Estatísticas

Acesse `/stats` para visualizar:

- Total de vetores indexados
- Total de pontos no banco
- Taxa de indexação
- Status do sistema

### Sobre

Acesse `/about` para informações sobre o projeto, tecnologias utilizadas e equipe.

## 🔧 Configuração Avançada

### Personalização de Templates

Os templates utilizam Jinja2 e herdam de `base.html`. Para personalizar:

1. Edite `templates/base.html` para mudanças globais (header, footer)
2. Edite templates individuais para mudanças específicas de página

### Adicionar Novas Páginas

1. Crie um novo template em `templates/`
2. Adicione uma rota em `main.py`:

```python
@app.get("/nova-pagina", response_class=HTMLResponse)
async def nova_pagina(request: Request):
    return templates.TemplateResponse(
        "nova-pagina.html",
        {"request": request, "current_page": "nova"}
    )
```

3. Atualize a navegação em `templates/base.html`

### Arquivos Estáticos

Para adicionar CSS, JavaScript ou imagens:

1. Coloque arquivos em `static/css/`, `static/js/` ou `static/images/`
2. Referencie nos templates:

```html
<link rel="stylesheet" href="{{ url_for('static', path='css/custom.css') }}">
<script src="{{ url_for('static', path='js/custom.js') }}"></script>
<img src="{{ url_for('static', path='images/logo.png') }}" alt="Logo">
```

## 🔒 Segurança

### API Key

A aplicação usa uma chave API para se autenticar com o backend RAG. A chave é configurada em `.env` e **nunca deve ser commitada** no Git.

### HTTPS

Para produção, configure um reverse proxy (Nginx, Caddy) com HTTPS:

```nginx
server {
    listen 443 ssl;
    server_name seu-dominio.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🐳 Docker (Opcional)

Crie um `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

E um `docker-compose.yml`:

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8001:8001"
    environment:
      - API_BASE_URL=http://api-rag:8000
      - API_KEY=${API_KEY}
    depends_on:
      - api-rag

  api-rag:
    image: seu-repo/api-rag:latest
    ports:
      - "8000:8000"
    environment:
      - QDRANT_URL=http://qdrant:6333
```

## 🧪 Testes

Para adicionar testes:

```bash
pip install pytest pytest-asyncio httpx
```

Crie `tests/test_main.py`:

```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Aviation RAG System" in response.text

def test_search_page():
    response = client.get("/search")
    assert response.status_code == 200
```

Execute:

```bash
pytest
```

## 📊 Monitoramento

Para produção, considere adicionar:

- **Logging**: Já configurado com Loguru
- **Métricas**: Prometheus + Grafana
- **APM**: New Relic, DataDog, ou similar
- **Health Checks**: Endpoint `/health` já disponível

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto é desenvolvido pelo Instituto Tecnológico de Aeronáutica (ITA) como parte do Projeto AirData.

© 2026 Projeto AirData - Todos os direitos reservados.

## 👥 Equipe

Desenvolvido pela equipe do Projeto AirData - ITA

## 🔗 Links Úteis

- [Portal AirData](https://www.airdata.ita.br)
- [GitHub - AirData](https://github.com/ita-airdata)
- [ITA](http://www.ita.br)
- [Documentação FastAPI](https://fastapi.tiangolo.com)
- [Documentação Jinja2](https://jinja.palletsprojects.com)

## 📞 Suporte

Para dúvidas ou problemas, abra uma issue no GitHub ou entre em contato com a equipe do Projeto AirData.

---

Desenvolvido com ❤️ pelo ITA
