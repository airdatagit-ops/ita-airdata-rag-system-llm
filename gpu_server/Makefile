.PHONY: help setup install start stop restart status logs health models uninstall deploy deploy-nginx clean

PYTHON   ?= python3
PORT     ?= 8090
WORKERS  ?= 1

help:
	@echo "GPU Inference Server — Management"
	@echo ""
	@echo "  ── setup ──────────────────────────────────────────────────────────────"
	@echo "  make setup                  Create venv and install dependencies"
	@echo "  make setup-cuda             Install PyTorch with CUDA 12.1 support"
	@echo "  make models                 Pull Ollama models (llama3.1:8b, qwen2.5:7b)"
	@echo ""
	@echo "  ── run (foreground) ─────────────────────────────────────────────────"
	@echo "  make start                  Start server in foreground (Ctrl+C to stop)"
	@echo "  make start PORT=8090        Start on a custom port"
	@echo ""
	@echo "  ── systemd service ─────────────────────────────────────────────────"
	@echo "  make install                Install and enable systemd service (sudo)"
	@echo "  make uninstall              Stop, disable, and remove service (sudo)"
	@echo "  make up                     Start the systemd service"
	@echo "  make down                   Stop the systemd service"
	@echo "  make restart                Restart the systemd service"
	@echo "  make status                 Show service status"
	@echo "  make logs                   Follow service logs (Ctrl+C to stop)"
	@echo "  make logs-tail              Show last 100 lines of logs"
	@echo ""
	@echo "  ── deploy ─────────────────────────────────────────────────────────"
	@echo "  make deploy                 Full deploy: setup + install service"
	@echo "  make deploy-nginx           Deploy + install nginx reverse proxy"
	@echo ""
	@echo "  ── diagnostics ────────────────────────────────────────────────────"
	@echo "  make health                 Check server health endpoint"
	@echo "  make gpu                    Show GPU info (nvidia-smi)"
	@echo "  make clean                  Remove venv and __pycache__"

# ── Setup ──────────────────────────────────────────────────────────────────

setup:
	@echo "=== Setting up GPU server ==="
	@test -d venv || $(PYTHON) -m venv venv
	@. venv/bin/activate && pip install --upgrade pip -q && pip install -r requirements.txt -q
	@test -f .env || cp .env.example .env
	@echo "=== Setup complete ==="

setup-cuda: setup
	@echo "=== Installing PyTorch with CUDA 12.1 ==="
	@. venv/bin/activate && pip install torch --index-url https://download.pytorch.org/whl/cu121
	@echo "=== CUDA setup complete ==="

# ── Run (foreground) ───────────────────────────────────────────────────────

start: setup
	@echo "Starting GPU server on port $(PORT)…"
	. venv/bin/activate && GPU_SERVER_PORT=$(PORT) $(PYTHON) server.py

# ── Systemd service ───────────────────────────────────────────────────────

install:
	@sudo bash deploy.sh --install

uninstall:
	@echo "=== Removing systemd service ==="
	sudo systemctl stop gpu-server 2>/dev/null || true
	sudo systemctl disable gpu-server 2>/dev/null || true
	sudo rm -f /etc/systemd/system/gpu-server.service
	sudo systemctl daemon-reload
	@echo "=== Service removed ==="

up:
	sudo systemctl start gpu-server

down:
	sudo systemctl stop gpu-server

restart:
	sudo systemctl restart gpu-server

status:
	@systemctl status gpu-server --no-pager -l

logs:
	sudo journalctl -u gpu-server -f

logs-tail:
	sudo journalctl -u gpu-server -n 100 --no-pager

# ── Deploy ─────────────────────────────────────────────────────────────────

deploy: setup
	sudo bash deploy.sh --install

deploy-nginx: setup
	sudo bash deploy.sh --install --with-nginx

# ── Models ─────────────────────────────────────────────────────────────────

models:
	@echo "=== Pulling Ollama models ==="
	ollama pull llama3.1:8b
	ollama pull qwen2.5:7b
	@echo "=== Models ready ==="

# ── Diagnostics ────────────────────────────────────────────────────────────

health:
	@curl -sf http://localhost:$(PORT)/health | python3 -m json.tool 2>/dev/null \
		|| echo "Server not responding on port $(PORT)"

gpu:
	@nvidia-smi 2>/dev/null || echo "nvidia-smi not found — no GPU detected"

# ── Cleanup ────────────────────────────────────────────────────────────────

clean:
	rm -rf venv __pycache__ *.log
