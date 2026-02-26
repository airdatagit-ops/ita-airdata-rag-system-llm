#!/bin/bash
# ========================================
# Aviation RAG System - Docker Control Script
# ========================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

COMPOSE_FILE="docker-compose.unified.yml"

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Aviation RAG System - Docker${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_usage() {
    echo -e "${YELLOW}Uso:${NC} $0 {start|stop|restart|build|logs|status|clean|init-data}"
    echo ""
    echo "Comandos:"
    echo "  start      - Inicia todos os containers"
    echo "  stop       - Para todos os containers"
    echo "  restart    - Reinicia todos os containers"
    echo "  build      - Reconstrói as imagens"
    echo "  logs       - Mostra logs em tempo real"
    echo "  status     - Mostra status dos containers"
    echo "  clean      - Remove containers, imagens e volumes"
    echo "  init-data  - Copia dados locais para os volumes Docker"
    echo ""
}

check_env() {
    if [ ! -f ".env" ]; then
        echo -e "${YELLOW}Aviso: Arquivo .env não encontrado.${NC}"
        echo -e "${YELLOW}Copiando .env.docker para .env...${NC}"
        cp .env.docker .env
        echo -e "${RED}IMPORTANTE: Edite o arquivo .env e configure sua API_KEY!${NC}"
    fi
}

start() {
    print_header
    check_env
    echo -e "${GREEN}Iniciando containers...${NC}"
    docker-compose -f $COMPOSE_FILE up -d
    echo -e "${GREEN}Containers iniciados!${NC}"
    echo ""
    echo -e "Acesse:"
    echo -e "  - Web App: ${BLUE}http://localhost:8001${NC}"
    echo -e "  - API RAG: ${BLUE}http://localhost:8000${NC}"
    echo -e "  - Qdrant:  ${BLUE}http://localhost:6333${NC}"
}

stop() {
    print_header
    echo -e "${YELLOW}Parando containers...${NC}"
    docker-compose -f $COMPOSE_FILE down
    echo -e "${GREEN}Containers parados!${NC}"
}

restart() {
    stop
    start
}

build() {
    print_header
    check_env
    echo -e "${GREEN}Reconstruindo imagens...${NC}"
    docker-compose -f $COMPOSE_FILE build --no-cache
    echo -e "${GREEN}Imagens reconstruídas!${NC}"
}

logs() {
    print_header
    echo -e "${GREEN}Mostrando logs (Ctrl+C para sair)...${NC}"
    docker-compose -f $COMPOSE_FILE logs -f
}

status() {
    print_header
    echo -e "${GREEN}Status dos containers:${NC}"
    echo ""
    docker-compose -f $COMPOSE_FILE ps
    echo ""
    echo -e "${GREEN}Uso de recursos:${NC}"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" \
        aviation-qdrant aviation-api-rag aviation-web-app 2>/dev/null || true
}

clean() {
    print_header
    echo -e "${RED}ATENÇÃO: Isso removerá todos os containers, imagens e volumes!${NC}"
    read -p "Tem certeza? (y/N): " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        echo -e "${YELLOW}Removendo containers...${NC}"
        docker-compose -f $COMPOSE_FILE down -v --rmi all
        echo -e "${GREEN}Limpeza concluída!${NC}"
    else
        echo -e "${YELLOW}Operação cancelada.${NC}"
    fi
}

init_data() {
    print_header
    echo -e "${GREEN}Inicializando volumes com dados locais...${NC}"
    
    # Verifica se os containers estão rodando
    if ! docker-compose -f $COMPOSE_FILE ps | grep -q "Up"; then
        echo -e "${YELLOW}Iniciando containers primeiro...${NC}"
        docker-compose -f $COMPOSE_FILE up -d
        sleep 5
    fi
    
    # Copia dados para o volume de documentos
    if [ -d "data" ]; then
        echo -e "${BLUE}Copiando documentos normativos...${NC}"
        docker cp data/. aviation-api-rag:/app/data/
        echo -e "${GREEN}Documentos copiados!${NC}"
    fi
    
    # Copia histórico de chat se existir
    if [ -d "web/chat_history" ]; then
        echo -e "${BLUE}Copiando histórico de chats...${NC}"
        docker cp web/chat_history/. aviation-web-app:/app/chat_history/
        echo -e "${GREEN}Histórico copiado!${NC}"
    fi
    
    echo -e "${GREEN}Inicialização concluída!${NC}"
}

# Main
case "${1:-}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    build)
        build
        ;;
    logs)
        logs
        ;;
    status)
        status
        ;;
    clean)
        clean
        ;;
    init-data)
        init_data
        ;;
    *)
        print_header
        print_usage
        exit 1
        ;;
esac
