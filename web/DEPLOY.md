# Guia de Deploy - Production 🚀

Este guia detalha como fazer deploy da aplicação web em ambiente de produção.

## Índice

1. [Pré-requisitos](#pré-requisitos)
2. [Deploy com Docker](#deploy-com-docker)
3. [Deploy Tradicional](#deploy-tradicional)
4. [Configuração com Nginx](#configuração-com-nginx)
5. [SSL/HTTPS](#sslhttps)
6. [Monitoramento](#monitoramento)
7. [Backup](#backup)

---

## Pré-requisitos

- [ ] Servidor Linux (Ubuntu 20.04+ recomendado)
- [ ] Python 3.8+
- [ ] Nginx ou outro reverse proxy
- [ ] Certificado SSL (Let's Encrypt recomendado)
- [ ] API RAG configurada e rodando
- [ ] Domínio configurado (DNS)

---

## Deploy com Docker

### 1. Preparar ambiente

```bash
# Instalar Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Instalar Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. Configurar variáveis de ambiente

```bash
# Criar arquivo .env
cat > .env << EOF
HOST=0.0.0.0
PORT=8001
RELOAD=False
API_BASE_URL=https://api.seu-dominio.com
API_KEY=sua-chave-api-super-secreta
APP_NAME=Aviation RAG Web Interface
APP_VERSION=1.0.0
EOF

# Proteger arquivo .env
chmod 600 .env
```

### 3. Build e Deploy

```bash
# Build da imagem
docker-compose build

# Iniciar serviços
docker-compose up -d

# Verificar logs
docker-compose logs -f web

# Verificar status
docker-compose ps
```

### 4. Gerenciamento

```bash
# Parar serviços
docker-compose stop

# Reiniciar serviços
docker-compose restart

# Atualizar aplicação
git pull
docker-compose build
docker-compose up -d

# Ver logs
docker-compose logs -f web
```

---

## Deploy Tradicional

### 1. Preparar servidor

```bash
# Atualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependências
sudo apt install -y python3 python3-pip python3-venv nginx
```

### 2. Configurar aplicação

```bash
# Criar diretório
sudo mkdir -p /opt/aviation-rag-web
cd /opt/aviation-rag-web

# Clonar repositório
git clone https://github.com/ita-airdata/rag-web-app.git .

# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Configurar .env
cp .env.example .env
nano .env  # Editar com configurações de produção
```

### 3. Configurar Systemd Service

```bash
# Criar arquivo de serviço
sudo nano /etc/systemd/system/aviation-rag-web.service
```

Conteúdo:

```ini
[Unit]
Description=Aviation RAG Web Interface
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/aviation-rag-web
Environment="PATH=/opt/aviation-rag-web/venv/bin"
ExecStart=/opt/aviation-rag-web/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001 --workers 4

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Ativar e iniciar:

```bash
# Recarregar systemd
sudo systemctl daemon-reload

# Habilitar serviço
sudo systemctl enable aviation-rag-web

# Iniciar serviço
sudo systemctl start aviation-rag-web

# Verificar status
sudo systemctl status aviation-rag-web

# Ver logs
sudo journalctl -u aviation-rag-web -f
```

---

## Configuração com Nginx

### 1. Criar configuração

```bash
sudo nano /etc/nginx/sites-available/aviation-rag-web
```

Conteúdo:

```nginx
# Upstream para a aplicação FastAPI
upstream aviation_rag_web {
    server 127.0.0.1:8001;
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name seu-dominio.com www.seu-dominio.com;
    
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name seu-dominio.com www.seu-dominio.com;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/seu-dominio.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/seu-dominio.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Logging
    access_log /var/log/nginx/aviation-rag-web-access.log;
    error_log /var/log/nginx/aviation-rag-web-error.log;
    
    # Client body size limit
    client_max_body_size 10M;
    
    # Static files
    location /static {
        alias /opt/aviation-rag-web/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    # Proxy to FastAPI
    location / {
        proxy_pass http://aviation_rag_web;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Health check
    location /health {
        proxy_pass http://aviation_rag_web;
        access_log off;
    }
}
```

### 2. Ativar configuração

```bash
# Criar symlink
sudo ln -s /etc/nginx/sites-available/aviation-rag-web /etc/nginx/sites-enabled/

# Testar configuração
sudo nginx -t

# Recarregar Nginx
sudo systemctl reload nginx
```

---

## SSL/HTTPS

### Usando Let's Encrypt

```bash
# Instalar Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obter certificado
sudo certbot --nginx -d seu-dominio.com -d www.seu-dominio.com

# Renovação automática (já configurada)
sudo certbot renew --dry-run

# Verificar timer de renovação
sudo systemctl status certbot.timer
```

---

## Monitoramento

### 1. Logs

```bash
# Logs da aplicação
sudo journalctl -u aviation-rag-web -f

# Logs do Nginx
sudo tail -f /var/log/nginx/aviation-rag-web-access.log
sudo tail -f /var/log/nginx/aviation-rag-web-error.log

# Logs do Docker (se usando)
docker-compose logs -f web
```

### 2. Health Checks

Configurar monitoramento externo:

```bash
# Usando curl
curl https://seu-dominio.com/health

# Usando systemd timer
sudo nano /etc/systemd/system/aviation-rag-healthcheck.service
```

Conteúdo:

```ini
[Unit]
Description=Aviation RAG Health Check

[Service]
Type=oneshot
ExecStart=/usr/bin/curl -f https://seu-dominio.com/health || /usr/bin/systemctl restart aviation-rag-web
```

Timer:

```bash
sudo nano /etc/systemd/system/aviation-rag-healthcheck.timer
```

```ini
[Unit]
Description=Aviation RAG Health Check Timer

[Timer]
OnBootSec=5min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
```

Ativar:

```bash
sudo systemctl enable aviation-rag-healthcheck.timer
sudo systemctl start aviation-rag-healthcheck.timer
```

### 3. Prometheus + Grafana (Opcional)

Para métricas avançadas, instale:

```bash
# Adicionar instrumentação no código
pip install prometheus-fastapi-instrumentator

# No main.py, adicionar:
# from prometheus_fastapi_instrumentator import Instrumentator
# Instrumentator().instrument(app).expose(app)
```

---

## Backup

### Script de backup

```bash
sudo nano /usr/local/bin/backup-aviation-rag.sh
```

Conteúdo:

```bash
#!/bin/bash

BACKUP_DIR="/backup/aviation-rag"
DATE=$(date +%Y%m%d_%H%M%S)
APP_DIR="/opt/aviation-rag-web"

# Criar diretório de backup
mkdir -p $BACKUP_DIR

# Backup do código
tar -czf $BACKUP_DIR/code_$DATE.tar.gz $APP_DIR

# Backup do .env
cp $APP_DIR/.env $BACKUP_DIR/env_$DATE

# Manter apenas últimos 7 dias
find $BACKUP_DIR -type f -mtime +7 -delete

echo "Backup completed: $DATE"
```

Tornar executável e agendar:

```bash
sudo chmod +x /usr/local/bin/backup-aviation-rag.sh

# Adicionar ao cron (diariamente às 2am)
sudo crontab -e
# Adicionar: 0 2 * * * /usr/local/bin/backup-aviation-rag.sh
```

---

## Checklist de Produção

Antes de ir para produção, verifique:

- [ ] `.env` com configurações de produção
- [ ] `RELOAD=False` no `.env`
- [ ] API_KEY segura e única
- [ ] SSL/HTTPS configurado
- [ ] Firewall configurado (apenas 80, 443)
- [ ] Nginx como reverse proxy
- [ ] Systemd service configurado
- [ ] Logs configurados
- [ ] Health checks ativos
- [ ] Backup automatizado
- [ ] Monitoramento configurado
- [ ] DNS configurado corretamente
- [ ] Testes end-to-end executados

---

## Comandos Úteis

```bash
# Reiniciar aplicação
sudo systemctl restart aviation-rag-web

# Ver logs em tempo real
sudo journalctl -u aviation-rag-web -f

# Verificar status
sudo systemctl status aviation-rag-web

# Recarregar Nginx
sudo systemctl reload nginx

# Verificar SSL
sudo certbot certificates

# Testar configuração Nginx
sudo nginx -t

# Ver processos
ps aux | grep uvicorn

# Ver uso de recursos
htop
```

---

## Troubleshooting

### Aplicação não inicia

```bash
# Verificar logs
sudo journalctl -u aviation-rag-web -n 50

# Verificar permissões
ls -la /opt/aviation-rag-web

# Verificar porta
sudo netstat -tulpn | grep 8001
```

### Erros de conexão com API

```bash
# Testar conexão
curl -H "X-API-Key: sua-chave" http://api-url/health

# Verificar .env
cat /opt/aviation-rag-web/.env
```

### SSL não funciona

```bash
# Verificar certificado
sudo certbot certificates

# Renovar manualmente
sudo certbot renew

# Verificar Nginx
sudo nginx -t
```

---

Desenvolvido pelo ITA - Projeto AirData
