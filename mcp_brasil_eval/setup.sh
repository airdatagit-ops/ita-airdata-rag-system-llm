#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -U pip
pip install -q -r requirements.txt

if ! command -v uvx >/dev/null 2>&1; then
  echo "uvx nao encontrado no PATH. Instalando 'uv' no venv local (provera 'uvx')..."
  pip install -q uv
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo ""
  echo "Arquivo .env criado a partir de .env.example."
  echo "Edite mcp_brasil_eval/.env e preencha ANTHROPIC_API_KEY antes de rodar."
fi

echo ""
echo "Setup OK."
echo "Para rodar:"
echo "  source mcp_brasil_eval/.venv/bin/activate"
echo "  python mcp_brasil_eval/ask.py \"sua pergunta\""
