# Estrutura de Arquivos Detalhada

## Visão Geral da Hierarquia

```
rag-web-app/
│
├── 📁 app/                          # Módulo principal da aplicação
│   ├── __init__.py                  # Inicialização do módulo
│   └── config.py                    # Configurações (Settings com Pydantic)
│
├── 📁 static/                       # Arquivos estáticos (CSS, JS, imagens)
│   ├── 📁 css/                      # Estilos CSS customizados
│   ├── 📁 js/                       # Scripts JavaScript
│   └── 📁 images/                   # Imagens e logos
│
├── 📁 templates/                    # Templates Jinja2
│   ├── base.html                    # Template base (header, footer, navegação)
│   ├── index.html                   # Página inicial
│   ├── search.html                  # Página de busca de regulamentações
│   ├── stats.html                   # Página de estatísticas
│   └── about.html                   # Página sobre o projeto
│
├── main.py                          # Aplicação FastAPI principal
├── requirements.txt                 # Dependências Python
├── .env.example                     # Exemplo de variáveis de ambiente
├── .env                            # Variáveis de ambiente (NÃO commitar)
├── .gitignore                      # Arquivos ignorados pelo Git
│
├── run.sh                          # Script de inicialização (Linux/macOS)
├── run.bat                         # Script de inicialização (Windows)
│
├── README.md                       # Documentação principal
└── QUICKSTART.md                   # Guia de início rápido
```

## Detalhamento dos Arquivos

### Diretório Raiz

#### `main.py`
**Propósito**: Ponto de entrada da aplicação FastAPI

**Contém**:
- Definição do app FastAPI
- Rotas para todas as páginas
- Configuração de templates e arquivos estáticos
- Lógica de comunicação com a API RAG
- Tratamento de erros

**Principais Rotas**:
- `GET /` - Página inicial
- `GET /search` - Formulário de busca
- `POST /search` - Processar busca
- `GET /stats` - Estatísticas do sistema
- `GET /about` - Sobre o projeto
- `GET /health` - Health check

#### `requirements.txt`
**Propósito**: Lista de dependências Python

**Principais Pacotes**:
- `fastapi` - Framework web
- `uvicorn` - Servidor ASGI
- `jinja2` - Engine de templates
- `httpx` - Cliente HTTP assíncrono
- `pydantic-settings` - Gerenciamento de configurações
- `loguru` - Logging

#### `.env.example`
**Propósito**: Exemplo de configuração

**Variáveis**:
- `HOST` - Host do servidor web
- `PORT` - Porta do servidor web
- `API_BASE_URL` - URL da API RAG
- `API_KEY` - Chave de autenticação da API

#### `.gitignore`
**Propósito**: Arquivos que não devem ser versionados

**Ignora**:
- `__pycache__/`
- `.env`
- `venv/`
- Arquivos de IDEs
- Logs

---

### Diretório `app/`

#### `__init__.py`
**Propósito**: Inicialização do módulo Python

**Contém**:
- Importação de settings
- Exports do módulo

#### `config.py`
**Propósito**: Gerenciamento de configurações

**Contém**:
- Classe `Settings` (Pydantic BaseSettings)
- Carregamento de variáveis de ambiente
- Validação de configurações
- Valores padrão

**Configurações Disponíveis**:
```python
HOST: str = "0.0.0.0"
PORT: int = 8001
RELOAD: bool = True
API_BASE_URL: str = "http://localhost:8000"
API_KEY: str = "your-api-key-here"
APP_NAME: str = "Aviation RAG Web Interface"
APP_VERSION: str = "1.0.0"
```

---

### Diretório `static/`

#### `css/`
**Propósito**: Estilos CSS customizados (opcional)

**Uso**: Adicione arquivos CSS para estilos que não estão inline nos templates

**Exemplo**:
```
css/
├── custom.css          # Estilos customizados
└── components.css      # Estilos de componentes
```

#### `js/`
**Propósito**: Scripts JavaScript (opcional)

**Uso**: Adicione scripts para funcionalidades interativas

**Exemplo**:
```
js/
├── search.js           # Lógica da página de busca
└── stats.js            # Lógica da página de estatísticas
```

#### `images/`
**Propósito**: Imagens e logos

**Uso**: Armazene logos, ícones e imagens

**Exemplo**:
```
images/
├── logo.png
├── favicon.ico
└── icons/
    ├── search.svg
    └── stats.svg
```

---

### Diretório `templates/`

#### `base.html`
**Propósito**: Template base que todos outros herdam

**Contém**:
- Estrutura HTML básica
- Header com navegação
- Footer
- Blocos para personalização:
  - `{% block title %}`
  - `{% block extra_styles %}`
  - `{% block content %}`
  - `{% block extra_scripts %}`

**Características**:
- Navegação responsiva
- Estilo consistente (padrão AirData)
- Menu ativo baseado em `current_page`

#### `index.html`
**Propósito**: Página inicial

**Contém**:
- Hero section com descrição do sistema
- Grid de features/funcionalidades
- CTA (Call to Action) para começar a buscar
- Cards informativos

**Herda**: `base.html`

**Blocos Personalizados**:
- `title` - "Início - Aviation RAG System"
- `extra_styles` - Estilos específicos da página
- `content` - Conteúdo principal

#### `search.html`
**Propósito**: Interface de busca de regulamentações

**Contém**:
- Formulário de busca com campos:
  - Query (pergunta)
  - Data (opcional)
  - Limite de resultados
  - Score threshold
- Exibição de resultados:
  - Resposta gerada
  - Métricas de performance
  - Lista de fontes com scores
  - Metadados de cada fonte

**Herda**: `base.html`

**Funcionalidades**:
- Submissão via POST
- Preservação de valores após busca
- Exibição de erros
- Cards expansíveis para metadados

#### `stats.html`
**Propósito**: Exibir estatísticas do sistema

**Contém**:
- Cards com métricas principais:
  - Total de vetores
  - Total de pontos
  - Taxa de indexação
- Indicador de status do sistema
- Tabela de informações detalhadas
- Seção explicativa sobre métricas
- Botão de refresh

**Herda**: `base.html`

**Características**:
- Atualização manual (botão)
- Timestamp da última atualização
- Cards com gradiente visual
- Status indicator animado

#### `about.html`
**Propósito**: Informações sobre o projeto

**Contém**:
- Descrição do sistema
- Lista de recursos
- Stack tecnológico (grid de cards)
- Explicação de como funciona
- Informações sobre o Projeto AirData/ITA
- Links úteis
- Informações de licença

**Herda**: `base.html`

**Seções**:
1. O que é o Aviation RAG System
2. Recursos Principais
3. Stack Tecnológico
4. Como Funciona
5. Projeto AirData - ITA
6. Licença e Uso

---

## Scripts de Inicialização

### `run.sh` (Linux/macOS)
**Propósito**: Automatizar inicialização no Unix

**Faz**:
1. Cria ambiente virtual se não existir
2. Ativa ambiente virtual
3. Instala/atualiza dependências
4. Cria .env se não existir
5. Inicia aplicação

**Uso**:
```bash
chmod +x run.sh
./run.sh
```

### `run.bat` (Windows)
**Propósito**: Automatizar inicialização no Windows

**Faz**: Mesmas etapas que `run.sh`, adaptado para Windows

**Uso**:
```bat
run.bat
```

---

## Documentação

### `README.md`
**Propósito**: Documentação principal do projeto

**Seções**:
- Visão geral
- Arquitetura
- Estrutura do projeto
- Instalação e configuração
- Uso
- Configuração avançada
- Segurança
- Docker
- Testes
- Contribuição
- Links úteis

### `QUICKSTART.md`
**Propósito**: Guia rápido para iniciantes

**Seções**:
- Pré-requisitos
- Passos rápidos
- Configuração manual
- Verificação
- Troubleshooting
- Próximos passos

---

## Fluxo de Dados

```
1. Usuário acessa página
   ↓
2. FastAPI renderiza template Jinja2
   ↓
3. Template é preenchido com dados
   ↓
4. HTML é retornado ao navegador
   ↓
5. Usuário submete formulário
   ↓
6. FastAPI processa POST
   ↓
7. Faz requisição HTTP para API RAG
   ↓
8. Recebe resposta da API
   ↓
9. Renderiza template com resultados
   ↓
10. Exibe resultados ao usuário
```

---

## Convenções de Código

### Python
- PEP 8 compliant
- Type hints em funções
- Docstrings em funções públicas
- Async/await para operações I/O

### Templates
- Nomes descritivos para variáveis
- Blocos bem identificados
- Comentários quando necessário
- Indentação consistente

### CSS
- Classes BEM-like quando necessário
- Cores da paleta AirData
- Responsivo por padrão
- Variáveis CSS para cores principais

---

## Manutenção

### Adicionar Nova Página

1. Criar template em `templates/nova_pagina.html`
2. Adicionar rota em `main.py`
3. Adicionar link de navegação em `base.html`
4. Testar em localhost

### Atualizar Estilos

1. Editar `base.html` para mudanças globais
2. Editar template específico para mudanças locais
3. Ou criar arquivo CSS em `static/css/`

### Adicionar Funcionalidade

1. Adicionar rota em `main.py`
2. Adicionar lógica de comunicação com API
3. Criar/atualizar template
4. Atualizar navegação se necessário

---

Desenvolvido pelo ITA - Projeto AirData
