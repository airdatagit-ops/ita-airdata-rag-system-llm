# Guia de Início Rápido 🚀

Este guia ajudará você a configurar e executar a aplicação web em poucos minutos.

## Pré-requisitos

Antes de começar, certifique-se de ter:

- [ ] Python 3.8 ou superior instalado
- [ ] API RAG rodando em `http://localhost:8000` (ou outra URL)
- [ ] Chave de API válida da API RAG

## Passos Rápidos

### 1. Clone e Entre no Diretório

```bash
cd web
```

### 2. Execute o Script de Inicialização

**Linux/macOS:**
```bash
./run.sh
```

**Windows:**
```bat
run.bat
```

### 3. Configure a API Key

Edite o arquivo `.env` e configure sua chave de API:

```env
API_KEY=sua-chave-api-aqui
API_BASE_URL=http://localhost:8000
```

### 4. Acesse a Aplicação

Abra seu navegador em: **http://localhost:8001**

## Configuração Manual (Alternativa)

Se preferir configurar manualmente:

```bash
# 1. Criar ambiente virtual
python -m venv venv

# 2. Ativar ambiente virtual
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Copiar e editar .env
cp .env.example .env
# Edite .env com suas configurações

# 5. Executar aplicação
python main.py
```

## Verificação Rápida

Teste se tudo está funcionando:

1. **Página Inicial**: http://localhost:8001
   - Deve exibir a página inicial do sistema

2. **Busca**: http://localhost:8001/search
   - Deve exibir o formulário de busca

3. **Estatísticas**: http://localhost:8001/stats
   - Deve exibir estatísticas do sistema (requer API rodando)

4. **Health Check**: http://localhost:8001/health
   - Deve retornar: `{"status": "healthy", "timestamp": "..."}`

## Primeiros Passos

### Fazer uma Busca

1. Acesse: http://localhost:8001/search
2. Digite uma pergunta, exemplo:
   ```
   Quais são os requisitos para certificação de pilotos comerciais?
   ```
3. Clique em "Buscar"
4. Visualize a resposta e as fontes

### Ver Estatísticas

1. Acesse: http://localhost:8001/stats
2. Visualize:
   - Total de vetores indexados
   - Total de pontos no banco
   - Status do sistema

## Troubleshooting

### Erro: "Connection refused" ao acessar /stats ou /search

**Causa**: A API RAG não está rodando ou URL incorreta

**Solução**:
1. Verifique se a API RAG está rodando: `curl http://localhost:8000/health`
2. Confirme a URL no arquivo `.env`
3. Verifique se a porta está correta

### Erro: "Invalid API Key"

**Causa**: Chave de API incorreta ou não configurada

**Solução**:
1. Verifique o arquivo `.env`
2. Confirme que `API_KEY` está correta
3. Verifique com o administrador da API RAG

### Erro: "Port already in use"

**Causa**: Porta 8001 já está sendo usada

**Solução**:
1. Pare o processo que está usando a porta
2. Ou altere a porta no `.env`:
   ```env
   PORT=8002
   ```

### Página não carrega estilos CSS

**Causa**: Problema com arquivos estáticos

**Solução**:
1. Verifique se o diretório `static/` existe
2. Reinicie a aplicação com `--reload`

## Próximos Passos

Depois de configurar e testar:

- [ ] Leia o [README.md](README.md) completo para mais detalhes
- [ ] Explore a [documentação da API](http://localhost:8000/docs)
- [ ] Personalize os templates em `templates/`
- [ ] Configure HTTPS para produção
- [ ] Configure monitoring e logging

## Comandos Úteis

```bash
# Parar a aplicação
Ctrl + C

# Reinstalar dependências
pip install -r requirements.txt --upgrade

# Limpar cache Python
find . -type d -name __pycache__ -exec rm -r {} +

# Ver logs em tempo real
tail -f logs/app.log  # se logging em arquivo estiver configurado
```

## Suporte

Se encontrar problemas:

1. Verifique os logs no terminal
2. Consulte a seção de Troubleshooting acima
3. Abra uma issue no GitHub
4. Entre em contato com a equipe AirData

---

Desenvolvido pelo ITA - Projeto AirData
