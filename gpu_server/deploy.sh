#!/usr/bin/env bash
# ============================================================================
# GPU Inference Server — Deployment Script
# Run on the GPU server (machine with NVIDIA GPU)
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh                      # Setup venv + dependencies only
#   sudo ./deploy.sh --install       # Also install systemd service
#   sudo ./deploy.sh --with-nginx    # Also install nginx snippet
#
# Flags:
#   --install      Install systemd service (requires root)
#   --with-nginx   Install nginx location snippet (requires root)
#
# Configuration via environment variables:
#   INSTALL_DIR    — where server files live    (default: script directory)
#   MODEL_CACHE    — HuggingFace model cache    (default: /dados/airdata/models_cache)
#   OLLAMA_DATA    — Ollama models directory    (default: /dados/airdata/ollama_data)
#   GPU_SERVER_PORT — server port               (default: 8090)
# ============================================================================

set -euo pipefail

# ── Resolve paths ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$SCRIPT_DIR}"
VENV_DIR="${INSTALL_DIR}/venv"
MODEL_CACHE="${MODEL_CACHE:-/dados/airdata/models_cache}"
OLLAMA_DATA="${OLLAMA_DATA:-/dados/airdata/ollama_data}"
PORT="${GPU_SERVER_PORT:-8090}"

DEPLOY_USER="${SUDO_USER:-$USER}"
DEPLOY_GROUP="$(id -gn "$DEPLOY_USER" 2>/dev/null || echo "$DEPLOY_USER")"

# Flags
INSTALL_SERVICE=false
INSTALL_NGINX=false

for arg in "$@"; do
    case "$arg" in
        --install)     INSTALL_SERVICE=true ;;
        --with-nginx)  INSTALL_NGINX=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $*"; }

echo "=== GPU Inference Server — Setup ==="
echo "  Install dir : ${INSTALL_DIR}"
echo "  Model cache : ${MODEL_CACHE}"
echo "  Ollama data : ${OLLAMA_DATA}"
echo "  Port        : ${PORT}"
echo "  User        : ${DEPLOY_USER}:${DEPLOY_GROUP}"
echo ""

# ── Step 1: Create directories ───────────────────────────────
info "[1/6] Creating directories…"
mkdir -p "${MODEL_CACHE}" "${OLLAMA_DATA}"
success "Directories ready."

# ── Step 2: Setup .env ───────────────────────────────────────
info "[2/6] Checking .env…"
if [ ! -f "${INSTALL_DIR}/.env" ]; then
    cp "${INSTALL_DIR}/.env.example" "${INSTALL_DIR}/.env"
    success "Created .env from .env.example — edit if needed."
else
    success ".env already exists."
fi

# ── Step 3: Setup Python venv ────────────────────────────────
info "[3/6] Setting up Python virtual environment…"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    success "Created venv."
else
    success "Venv exists."
fi
source "${VENV_DIR}/bin/activate"

info "[4/6] Installing dependencies…"
pip install --upgrade pip -q
pip install -r "${INSTALL_DIR}/requirements.txt" -q
success "Dependencies installed."

# ── Step 4: Check GPU ────────────────────────────────────────
info "[5/6] Checking GPU availability…"
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'  CUDA: {torch.version.cuda}')
else:
    print('  WARNING: No CUDA GPU detected — models will run on CPU')
"

# ── Step 5: Check Ollama ─────────────────────────────────────
info "[6/6] Checking Ollama…"
if command -v ollama &>/dev/null; then
    success "Ollama found: $(ollama --version 2>/dev/null || echo 'version unknown')"
else
    warn "Ollama not installed — install with: curl -fsSL https://ollama.com/install.sh | sh"
fi

echo ""
success "=== Base setup complete ==="

# ── Optional: Install systemd service ────────────────────────
if $INSTALL_SERVICE; then
    echo ""
    info "Installing systemd service…"

    if [ "$(id -u)" -ne 0 ]; then
        fail "--install requires root. Run with sudo."
        exit 1
    fi

    SERVICE_FILE="/etc/systemd/system/gpu-server.service"
    TEMPLATE="${INSTALL_DIR}/gpu_server.service"

    if [ ! -f "$TEMPLATE" ]; then
        fail "Service template not found: $TEMPLATE"
        exit 1
    fi

    sed \
        -e "s|__INSTALL_DIR__|${INSTALL_DIR}|g" \
        -e "s|__USER__|${DEPLOY_USER}|g" \
        -e "s|__GROUP__|${DEPLOY_GROUP}|g" \
        -e "s|__MODEL_CACHE__|${MODEL_CACHE}|g" \
        -e "s|__OLLAMA_DATA__|${OLLAMA_DATA}|g" \
        "$TEMPLATE" > "$SERVICE_FILE"

    systemctl daemon-reload
    systemctl enable gpu-server
    success "Systemd service installed and enabled."
    info "Start with: sudo systemctl start gpu-server"
    info "Logs:       sudo journalctl -u gpu-server -f"
fi

# ── Optional: Install nginx snippet ──────────────────────────
if $INSTALL_NGINX; then
    echo ""
    info "Installing nginx location snippet…"

    if [ "$(id -u)" -ne 0 ]; then
        fail "--with-nginx requires root. Run with sudo."
        exit 1
    fi

    NGINX_SNIPPET="${INSTALL_DIR}/nginx-gpu-api.conf"
    NGINX_TARGET="/etc/nginx/sites-available/gpu-api"

    if [ ! -f "$NGINX_SNIPPET" ]; then
        fail "Nginx snippet not found: $NGINX_SNIPPET"
        exit 1
    fi

    cp "$NGINX_SNIPPET" "$NGINX_TARGET"
    success "Copied snippet to $NGINX_TARGET"

    # Check if already included in the default site
    DEFAULT_SITE="/etc/nginx/sites-enabled/default"
    if [ -f "$DEFAULT_SITE" ]; then
        if grep -q 'include /etc/nginx/sites-available/gpu-api' "$DEFAULT_SITE"; then
            success "Include directive already present in default site."
        else
            warn "Add the following line inside your server {} block:"
            warn "    include /etc/nginx/sites-available/gpu-api;"
            warn ""
            warn "Example:"
            warn "    sudo sed -i '/server_name/a\\\\    include /etc/nginx/sites-available/gpu-api;' $DEFAULT_SITE"
        fi
    fi

    if nginx -t 2>/dev/null; then
        systemctl reload nginx
        success "Nginx configuration valid and reloaded."
    else
        warn "nginx -t failed. Check configuration manually."
    fi
fi

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "=== Next steps ==="
echo ""
echo "  Start manually:"
echo "    cd ${INSTALL_DIR}"
echo "    source venv/bin/activate"
echo "    python server.py"
echo ""
if ! $INSTALL_SERVICE; then
    echo "  Install as systemd service:"
    echo "    sudo ./deploy.sh --install"
    echo ""
fi
if ! $INSTALL_NGINX; then
    echo "  Install nginx reverse proxy:"
    echo "    sudo ./deploy.sh --with-nginx"
    echo ""
fi
echo "  Test:"
echo "    curl http://localhost:${PORT}/health"
echo ""
echo "  Pull Ollama models:"
echo "    ollama pull llama3.1:8b"
echo "    ollama pull llama3.1:70b"
