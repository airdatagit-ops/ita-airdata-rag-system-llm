#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Aviation RAG System — Deploy Script
# ============================================================
# Usage:
#   make deploy              Regular deploy (git pull + deps + restart)
#   make deploy-first        First-time setup (creates .env from example)
#   make check               Run pre-flight checks only
#
# Flags:
#   --first-run    Create .env files from examples if missing
#   --skip-pull    Skip git pull
#   --with-nginx   Update nginx snippet (skipped by default)
#   --check-only   Run verification checks without deploying
#
# nginx is skipped by default to avoid conflicts on shared servers.
# Use --with-nginx only to update the /ragapi/ and /explore/ snippet.
# Virtual host configuration is managed manually (see README §14.3).
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_DIR="$PROJECT_DIR/deploy"

DEPLOY_USER="${SUDO_USER:-$USER}"
DEPLOY_GROUP="$(id -gn "$DEPLOY_USER")"

FIRST_RUN=false
SKIP_PULL=false
SKIP_NGINX=true
CHECK_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --first-run)   FIRST_RUN=true ;;
        --skip-pull)   SKIP_PULL=true ;;
        --with-nginx)  SKIP_NGINX=false ;;
        --check-only)  CHECK_ONLY=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# ── Colored output helpers ───────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[FAIL]${NC}  $*"; }

# ── Pre-flight checks ───────────────────────────────────────

preflight_check() {
    info "Running pre-flight checks..."
    local failed=0

    # Python
    if command -v python3 &>/dev/null; then
        local pyver
        pyver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        success "Python $pyver"
    else
        error "python3 not found — sudo apt install python3 python3-venv python3-pip"
        failed=1
    fi

    # python3-venv
    if python3 -m venv --help &>/dev/null; then
        success "python3-venv available"
    else
        error "python3-venv not found — sudo apt install python3-venv"
        failed=1
    fi

    # Docker
    if command -v docker &>/dev/null && docker info &>/dev/null; then
        success "Docker running"
    else
        warn "Docker not available — needed for Qdrant"
    fi

    # Qdrant
    if curl -sf http://localhost:6333/healthz &>/dev/null; then
        success "Qdrant healthy (port 6333)"
    else
        warn "Qdrant not responding on port 6333"
    fi

    # GPU / NVIDIA
    if command -v nvidia-smi &>/dev/null; then
        local gpu_name gpu_mem
        gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
        gpu_mem="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
        success "GPU: $gpu_name ($gpu_mem)"
    else
        warn "nvidia-smi not found — embeddings will run on CPU (slow)"
    fi

    # Ollama
    if systemctl is-active --quiet ollama 2>/dev/null; then
        success "Ollama service active"
        if command -v ollama &>/dev/null; then
            local models
            models="$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | paste -sd', ')"
            info "  Ollama models: ${models:-none}"
        fi
    else
        warn "Ollama service not active — sudo systemctl start ollama"
    fi

    # nginx
    if command -v nginx &>/dev/null; then
        success "nginx installed ($(nginx -v 2>&1 | awk -F/ '{print $2}'))"
    else
        error "nginx not found — sudo apt install nginx"
        failed=1
    fi

    # .env files
    if [[ -f "$PROJECT_DIR/.env" ]]; then
        success ".env exists"
    else
        warn ".env missing — use --first-run to create from env.example"
    fi

    if [[ -f "$PROJECT_DIR/web/.env" ]]; then
        success "web/.env exists"
    else
        warn "web/.env missing — use --first-run to create from web/env.example"
    fi

    echo ""
    if [[ $failed -ne 0 ]]; then
        error "Pre-flight checks failed. Fix the issues above and retry."
        exit 1
    fi
    success "Pre-flight checks passed."
}

# ── Health checks (post-deploy) ─────────────────────────────

health_check() {
    info "Running post-deploy health checks..."
    local all_ok=true

    # Wait for services to start
    sleep 3

    # systemd services
    for svc in ragapi ragweb ragexplore; do
        if systemctl is-active --quiet "$svc" 2>/dev/null; then
            success "systemd: $svc active"
        else
            error "systemd: $svc not active"
            all_ok=false
        fi
    done

    # HTTP health checks with retry
    local endpoints=(
        "http://127.0.0.1:8083/health|API"
        "http://127.0.0.1:8082/health|Web"
        "http://127.0.0.1:8001/explore/|Datasette"
    )

    for entry in "${endpoints[@]}"; do
        local url="${entry%%|*}"
        local name="${entry##*|}"
        local ok=false
        for attempt in 1 2 3 4 5; do
            if curl -sf --max-time 5 "$url" &>/dev/null; then
                ok=true
                break
            fi
            sleep 2
        done
        if $ok; then
            success "HTTP: $name ($url)"
        else
            error "HTTP: $name not responding ($url)"
            all_ok=false
        fi
    done

    # PyTorch GPU check
    if [[ -f "$PROJECT_DIR/venv/bin/python" ]]; then
        local cuda_status
        cuda_status="$("$PROJECT_DIR/venv/bin/python" -c "import torch; print('GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')" 2>/dev/null || echo "torch not available")"
        info "PyTorch: $cuda_status"
    fi

    echo ""
    if $all_ok; then
        success "All health checks passed!"
        return 0
    else
        error "Some checks failed. Check logs: journalctl -u ragapi -n 30"
        return 1
    fi
}

# ── Check-only mode ─────────────────────────────────────────

if $CHECK_ONLY; then
    preflight_check
    echo ""
    if systemctl is-active --quiet ragapi 2>/dev/null; then
        health_check || exit 1
    else
        info "Services not running — skipping health checks."
    fi
    exit 0
fi

# ── Deploy starts here ──────────────────────────────────────

echo ""
echo "========================================"
echo "  Aviation RAG System — Deploy"
echo "========================================"
echo "  Project: $PROJECT_DIR"
echo "  User:    $DEPLOY_USER:$DEPLOY_GROUP"
echo "  Mode:    $(if $FIRST_RUN; then echo 'first-run'; else echo 'update'; fi)"
echo "========================================"
echo ""

preflight_check

# ── Step 1: Git pull ─────────────────────────────────────────

if ! $SKIP_PULL; then
    info "Pulling latest changes..."
    cd "$PROJECT_DIR"
    sudo -u "$DEPLOY_USER" git stash --quiet 2>/dev/null || true
    sudo -u "$DEPLOY_USER" git pull
    sudo -u "$DEPLOY_USER" git stash pop --quiet 2>/dev/null || true
    success "Git pull complete."
else
    info "Skipping git pull (--skip-pull)."
fi

# ── Step 2: First-run .env setup ─────────────────────────────

if $FIRST_RUN; then
    if [[ ! -f "$PROJECT_DIR/.env" ]]; then
        info "Creating .env from env.example..."
        cp "$PROJECT_DIR/env.example" "$PROJECT_DIR/.env"
        chown "$DEPLOY_USER:$DEPLOY_GROUP" "$PROJECT_DIR/.env"
        warn "Edit .env and set API_KEY, OLLAMA_MODEL, and other values."
    fi
    if [[ ! -f "$PROJECT_DIR/web/.env" ]]; then
        info "Creating web/.env from web/env.example..."
        cp "$PROJECT_DIR/web/env.example" "$PROJECT_DIR/web/.env"
        chown "$DEPLOY_USER:$DEPLOY_GROUP" "$PROJECT_DIR/web/.env"
        warn "Edit web/.env and set API_KEY (must match root .env)."
    fi
fi

# Ensure web/.env has production values (idempotent)
if [[ -f "$PROJECT_DIR/web/.env" ]]; then
    _web_env="$PROJECT_DIR/web/.env"

    if ! grep -q '^ENVIRONMENT=' "$_web_env"; then
        echo 'ENVIRONMENT=production' >> "$_web_env"
        info "Added ENVIRONMENT=production to web/.env"
    else
        sed -i 's/^ENVIRONMENT=.*/ENVIRONMENT=production/' "$_web_env"
    fi

    sed -i 's/^RELOAD=True/RELOAD=False/' "$_web_env"

    if grep -q 'API_BASE_URL=http://161' "$_web_env"; then
        sed -i 's|^API_BASE_URL=http://161.*|API_BASE_URL=http://127.0.0.1:8083|' "$_web_env"
        info "Fixed API_BASE_URL to use direct local connection"
    fi
fi

# ── Step 3: Backend venv + dependencies ──────────────────────

info "Setting up backend virtual environment..."
cd "$PROJECT_DIR"

if [[ ! -d venv ]]; then
    info "Creating venv..."
    sudo -u "$DEPLOY_USER" python3 -m venv venv
fi

REQ_HASH="$(md5sum requirements.txt | awk '{print $1}')"
REQ_STAMP="venv/.requirements.md5"

if [[ ! -f "$REQ_STAMP" ]] || [[ "$(cat "$REQ_STAMP" 2>/dev/null)" != "$REQ_HASH" ]]; then
    info "Installing backend dependencies (requirements changed)..."
    sudo -u "$DEPLOY_USER" venv/bin/pip install --upgrade pip -q
    sudo -u "$DEPLOY_USER" venv/bin/pip install -r requirements.txt -q
    echo "$REQ_HASH" > "$REQ_STAMP"
    chown "$DEPLOY_USER:$DEPLOY_GROUP" "$REQ_STAMP"
    success "Backend dependencies installed."
else
    success "Backend dependencies up to date (skipped)."
fi

# ── Step 4: Web venv + dependencies ──────────────────────────

info "Setting up web virtual environment..."
cd "$PROJECT_DIR/web"

if [[ ! -d venv ]]; then
    info "Creating web/venv..."
    sudo -u "$DEPLOY_USER" python3 -m venv venv
fi

WEB_REQ_HASH="$(md5sum requirements.txt | awk '{print $1}')"
WEB_REQ_STAMP="venv/.requirements.md5"

if [[ ! -f "$WEB_REQ_STAMP" ]] || [[ "$(cat "$WEB_REQ_STAMP" 2>/dev/null)" != "$WEB_REQ_HASH" ]]; then
    info "Installing web dependencies (requirements changed)..."
    sudo -u "$DEPLOY_USER" venv/bin/pip install --upgrade pip -q
    sudo -u "$DEPLOY_USER" venv/bin/pip install -r requirements.txt -q
    echo "$WEB_REQ_HASH" > "$WEB_REQ_STAMP"
    chown "$DEPLOY_USER:$DEPLOY_GROUP" "$WEB_REQ_STAMP"
    success "Web dependencies installed."
else
    success "Web dependencies up to date (skipped)."
fi

cd "$PROJECT_DIR"

# ── Step 4b: Pre-download ML models ──────────────────────────

info "Pre-downloading ML models..."
sudo -u "$DEPLOY_USER" venv/bin/python -m scripts.download_models
success "ML models ready."

# ── Step 5: Install systemd services ─────────────────────────

SERVICES_CHANGED=false

for service_template in ragapi.service ragweb.service ragexplore.service; do
    local_file="$DEPLOY_DIR/$service_template"
    target_file="/etc/systemd/system/$service_template"

    rendered="$(sed \
        -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
        -e "s|__USER__|$DEPLOY_USER|g" \
        -e "s|__GROUP__|$DEPLOY_GROUP|g" \
        "$local_file")"

    if [[ ! -f "$target_file" ]] || [[ "$rendered" != "$(cat "$target_file")" ]]; then
        echo "$rendered" > "$target_file"
        info "Updated $service_template"
        SERVICES_CHANGED=true
    fi
done

if $SERVICES_CHANGED; then
    systemctl daemon-reload
    success "Service files updated, daemon reloaded."
else
    success "Service files unchanged (skipped)."
fi

# Enable services to start on boot
systemctl enable ragapi ragweb ragexplore --quiet 2>/dev/null || true

# ── Step 6: Install nginx configuration ──────────────────────

if ! $SKIP_NGINX; then
    if [[ ! -f /etc/nginx/sites-available/rag ]] || ! diff -q "$DEPLOY_DIR/nginx-rag.conf" /etc/nginx/sites-available/rag &>/dev/null; then
        info "Installing nginx RAG locations snippet..."
        cp "$DEPLOY_DIR/nginx-rag.conf" /etc/nginx/sites-available/rag

        # Remove legacy standalone site symlink (replaced by include approach)
        if [[ -L /etc/nginx/sites-enabled/rag ]]; then
            rm -f /etc/nginx/sites-enabled/rag
            info "Removed legacy sites-enabled/rag symlink."
        fi

        # Inject include directive into default site if not already present
        local _default="/etc/nginx/sites-available/default"
        if [[ -f "$_default" ]] && ! grep -q 'include /etc/nginx/sites-available/rag' "$_default"; then
            sed -i '/server_name/a\\n    include /etc/nginx/sites-available/rag;' "$_default"
            ln -sf "$_default" /etc/nginx/sites-enabled/default
            info "Added 'include rag' to default site."
        fi

        if nginx -t 2>/dev/null; then
            systemctl reload nginx
            success "nginx configured and reloaded."
        else
            error "nginx config test failed — check: nginx -t"
            error "You may need to manually add 'include /etc/nginx/sites-available/rag;' to your server block."
            exit 1
        fi
    else
        success "nginx config unchanged (skipped)."
    fi
else
    info "Skipping nginx (--skip-nginx)."
fi

# ── Step 7: Restart services ─────────────────────────────────

info "Restarting services..."

systemctl restart ragapi
systemctl restart ragweb
systemctl restart ragexplore

success "All services restarted."

# ── Step 8: Health checks ────────────────────────────────────

if ! health_check; then
    error "Deploy finished but post-deploy health checks failed — failing the run so the workflow surfaces it."
    exit 1
fi

echo ""
echo "========================================"
echo "  Deploy complete!"
echo "========================================"
echo "  Web:       http://<IP>/ragweb/"
echo "  API:       http://<IP>/ragapi/"
echo "  Explore:   http://<IP>/explore/"
echo "========================================"
