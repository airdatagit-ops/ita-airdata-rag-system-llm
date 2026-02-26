# Hierarquia Completa do Projeto

```
rag-web-app/
│
├── 📄 main.py                      # Aplicação FastAPI principal (rotas, lógica)
├── 📄 requirements.txt             # Dependências Python
├── 📄 .env.example                 # Exemplo de configuração
├── 📄 .env                         # Configuração real (NÃO commitar)
├── 📄 .gitignore                   # Arquivos ignorados pelo Git
├── 📄 .dockerignore                # Arquivos ignorados pelo Docker
│
├── 📄 Dockerfile                   # Configuração Docker
├── 📄 docker-compose.yml           # Orquestração de containers
│
├── 🔧 run.sh                       # Script de inicialização (Unix)
├── 🔧 run.bat                      # Script de inicialização (Windows)
│
├── 📚 README.md                    # Documentação principal
├── 📚 QUICKSTART.md                # Guia de início rápido
├── 📚 STRUCTURE.md                 # Estrutura detalhada dos arquivos
├── 📚 DEPLOY.md                    # Guia de deploy em produção
│
├── 📁 app/                         # Módulo da aplicação
│   ├── 📄 __init__.py              # Inicialização do módulo
│   └── 📄 config.py                # Configurações (Pydantic Settings)
│
├── 📁 static/                      # Arquivos estáticos
│   ├── 📁 css/                     # Estilos CSS
│   │   └── 📄 custom.css           # CSS customizado (opcional)
│   │
│   ├── 📁 js/                      # Scripts JavaScript
│   │   └── 📄 app.js               # JavaScript customizado (opcional)
│   │
│   └── 📁 images/                  # Imagens, logos, ícones
│       └── (seus arquivos aqui)
│
└── 📁 templates/                   # Templates Jinja2
    ├── 📄 base.html                # Template base (herança)
    ├── 📄 index.html               # Página inicial
    ├── 📄 search.html              # Busca de regulamentações
    ├── 📄 stats.html               # Estatísticas do sistema
    └── 📄 about.html               # Sobre o projeto
```

## Contagem de Arquivos

### Arquivos Principais
- **Python**: 3 arquivos (main.py, config.py, __init__.py)
- **Templates HTML**: 5 arquivos
- **Configuração**: 7 arquivos (.env.example, requirements.txt, Dockerfile, etc.)
- **Documentação**: 4 arquivos markdown
- **Scripts**: 2 arquivos (run.sh, run.bat)
- **Static**: 2 arquivos base (custom.css, app.js)

**Total**: ~23 arquivos principais

## Tamanho Aproximado

```
app/                    ~2 KB
templates/             ~40 KB
static/                 ~5 KB
main.py                ~10 KB
documentation/         ~50 KB
config files/           ~3 KB
─────────────────────────────
Total:                ~110 KB
```

## Estrutura Minimalista (Essencial)

Se você quiser apenas o essencial para rodar:

```
rag-web-app/
├── main.py                  ✅ Essencial
├── requirements.txt         ✅ Essencial
├── .env                     ✅ Essencial
├── app/
│   ├── __init__.py          ✅ Essencial
│   └── config.py            ✅ Essencial
├── templates/
│   ├── base.html            ✅ Essencial
│   ├── index.html           ✅ Essencial
│   ├── search.html          ✅ Essencial
│   ├── stats.html           ✅ Essencial
│   └── about.html           ✅ Essencial
└── static/
    ├── css/                 ⚪ Opcional
    ├── js/                  ⚪ Opcional
    └── images/              ⚪ Opcional
```

**Mínimo para rodar**: 10 arquivos essenciais

## Ordem de Criação Recomendada

Se você estiver construindo do zero:

1. **Fase 1 - Setup Básico**
   ```
   ├── requirements.txt
   ├── .env
   └── app/config.py
   ```

2. **Fase 2 - Estrutura Base**
   ```
   ├── main.py (com rota / apenas)
   └── templates/base.html
   ```

3. **Fase 3 - Páginas Core**
   ```
   ├── templates/index.html
   └── templates/search.html
   ```

4. **Fase 4 - Funcionalidades Adicionais**
   ```
   ├── templates/stats.html
   └── templates/about.html
   ```

5. **Fase 5 - Melhorias** (opcional)
   ```
   ├── static/css/custom.css
   ├── static/js/app.js
   └── Dockerfile
   ```

6. **Fase 6 - Documentação**
   ```
   ├── README.md
   ├── QUICKSTART.md
   └── DEPLOY.md
   ```

## Arquivos por Responsabilidade

### Backend (Python)
```
main.py           → Rotas, lógica, comunicação com API
app/config.py     → Configurações, variáveis de ambiente
app/__init__.py   → Inicialização do módulo
```

### Frontend (Templates)
```
templates/base.html    → Estrutura base, header, footer
templates/index.html   → Página inicial
templates/search.html  → Interface de busca
templates/stats.html   → Estatísticas
templates/about.html   → Informações do projeto
```

### Configuração
```
requirements.txt  → Dependências
.env             → Variáveis de ambiente
.gitignore       → Controle de versão
Dockerfile       → Containerização
docker-compose   → Orquestração
```

### Documentação
```
README.md        → Visão geral e instalação
QUICKSTART.md    → Início rápido
STRUCTURE.md     → Estrutura detalhada
DEPLOY.md        → Deploy em produção
```

### Scripts
```
run.sh           → Inicialização (Unix)
run.bat          → Inicialização (Windows)
```

### Assets (Opcional)
```
static/css/      → Estilos customizados
static/js/       → JavaScript customizado
static/images/   → Imagens e logos
```

## Dependências Entre Arquivos

```
main.py
  ├─→ app/config.py (Settings)
  ├─→ templates/*.html (Jinja2)
  └─→ static/* (arquivos estáticos)

templates/base.html
  └─→ herdado por todos outros templates

run.sh / run.bat
  ├─→ requirements.txt
  └─→ main.py

Dockerfile
  ├─→ requirements.txt
  └─→ main.py

docker-compose.yml
  └─→ Dockerfile
```

## Checklist de Arquivos

Para verificar se você tem tudo:

### Essenciais
- [ ] main.py
- [ ] app/config.py
- [ ] app/__init__.py
- [ ] requirements.txt
- [ ] .env (ou .env.example)
- [ ] templates/base.html
- [ ] templates/index.html
- [ ] templates/search.html
- [ ] templates/stats.html
- [ ] templates/about.html

### Recomendados
- [ ] README.md
- [ ] QUICKSTART.md
- [ ] .gitignore
- [ ] run.sh ou run.bat
- [ ] static/css/ (diretório)
- [ ] static/js/ (diretório)
- [ ] static/images/ (diretório)

### Opcionais
- [ ] Dockerfile
- [ ] docker-compose.yml
- [ ] .dockerignore
- [ ] STRUCTURE.md
- [ ] DEPLOY.md
- [ ] static/css/custom.css
- [ ] static/js/app.js

---

**Nota**: Esta estrutura é escalável. Comece com o essencial e adicione conforme necessário!

Desenvolvido pelo ITA - Projeto AirData
