# 🛩️ Aviation RAG Web Interface - COMECE AQUI

## ⚡ Início Rápido (5 minutos)

### 1. Configure a API Key

Edite o arquivo `.env`:
```bash
nano .env  # ou use seu editor preferido
```

Altere esta linha:
```env
API_KEY=sua-chave-api-aqui
```

### 2. Execute

**Linux/macOS:**
```bash
chmod +x run.sh
./run.sh
```

**Windows:**
```bat
run.bat
```

### 3. Acesse

Abra seu navegador em: **http://localhost:8001**

---

## 📚 Documentação

Este projeto contém documentação completa em vários arquivos:

| Arquivo | Descrição |
|---------|-----------|
| **[README.md](README.md)** | 📖 Documentação principal completa |
| **[QUICKSTART.md](QUICKSTART.md)** | ⚡ Guia de início rápido detalhado |
| **[STRUCTURE.md](STRUCTURE.md)** | 🏗️ Estrutura de arquivos explicada |
| **[HIERARCHY.md](HIERARCHY.md)** | 📁 Hierarquia visual do projeto |
| **[DEPLOY.md](DEPLOY.md)** | 🚀 Guia de deploy em produção |

**Recomendação**: Leia nesta ordem se for sua primeira vez:
1. Este arquivo (START_HERE.md) ✅
2. QUICKSTART.md
3. README.md (para detalhes completos)

---

## 🎯 O que este projeto faz?

Este é um **interface web** para interagir com um sistema RAG (Retrieval-Augmented Generation) de regulamentações aeronáuticas.

### Funcionalidades:
- 🔍 Buscar regulamentações em linguagem natural
- 📊 Ver estatísticas do sistema
- 📚 Visualizar fontes e metadados
- ⚡ Interface moderna e responsiva

---

## 📂 Estrutura do Projeto

```
rag-web-app/
├── 📄 main.py              # Aplicação FastAPI
├── 📁 app/                 # Configurações
├── 📁 templates/           # Páginas HTML
├── 📁 static/              # CSS, JS, imagens
└── 📚 *.md                 # Documentação
```

Para ver a estrutura completa: **[HIERARCHY.md](HIERARCHY.md)**

---

## ⚙️ Pré-requisitos

Você precisa ter:

1. **Python 3.8+** instalado
2. **API RAG** rodando (backend)
3. **Chave de API** válida

Não tem a API? Você precisa do backend rodando primeiro!

---

## 🔧 Configuração

O arquivo `.env` controla todas as configurações:

```env
# Servidor Web
HOST=0.0.0.0
PORT=8001

# API Backend
API_BASE_URL=http://localhost:8000
API_KEY=sua-chave-aqui         # ← IMPORTANTE: Configure isso!
```

---

## 🚀 Próximos Passos

Depois de rodar pela primeira vez:

1. ✅ Teste a busca em `/search`
2. ✅ Veja as estatísticas em `/stats`
3. ✅ Leia o [README.md](README.md) completo
4. ✅ Para produção, veja [DEPLOY.md](DEPLOY.md)

---

## 🆘 Problemas Comuns

### "Connection refused" ao buscar

**Causa**: API backend não está rodando

**Solução**: Verifique se a API está rodando em `http://localhost:8000`
```bash
curl http://localhost:8000/health
```

### "Invalid API Key"

**Causa**: Chave incorreta no `.env`

**Solução**: Verifique o `.env` e configure a chave correta

### Porta 8001 já em uso

**Solução**: Mude a porta no `.env`:
```env
PORT=8002
```

---

## 📞 Suporte

- 📖 Leia a documentação em `README.md`
- 🐛 Problemas? Veja `QUICKSTART.md` > Troubleshooting
- 💬 Issues: GitHub do projeto

---

## 🎓 Projeto AirData - ITA

Desenvolvido pelo Instituto Tecnológico de Aeronáutica

© 2026 Projeto AirData - Todos os direitos reservados

---

**Pronto para começar? Execute `./run.sh` (Linux/Mac) ou `run.bat` (Windows)!**
