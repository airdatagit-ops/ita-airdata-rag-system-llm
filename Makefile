.PHONY: install install-backend install-web test lint lint-fix eval eval-retrieval eval-generation extract-pipeline validate-data validate-lexml clean help collect collect-sislaer collect-legacy collect-anac embed index pipeline query explore migrate deploy deploy-first deploy-nginx check start start-api start-web download-models backup restore

PYTHON ?= python
PYTHON3 ?= python3
VENV_DIR ?= venv
WEB_VENV_DIR ?= web/venv
K ?= 5
WORKERS ?= 4
SAMPLE ?=
LIMIT ?= 0
FILE ?=
SEARCH_MODE ?= auto
DOC_TYPES ?=
KEYWORDS ?=
CONCURRENCY ?= 10
SOURCES ?= sislaer,lexml
PDF_DIR ?= ./data/pdfs
MODE ?=
BATCH_SIZE ?=
STORE_DB ?= data/store.db
SQL ?=
INPUT ?=
OUTPUT ?=
QDRANT_HOST ?= localhost
QDRANT_PORT ?= 6333
QDRANT_COLLECTION ?= aviation_regulations
BACKUP_DIR ?= data/backups

help:
	@echo "Usage:"
	@echo ""
	@echo "  ── 3-phase pipeline ──────────────────────────────────────────────────"
	@echo "  make collect                                      Phase 1: collect new documents into SQLite"
	@echo "  make collect CHECK=1                              Re-download all and verify content hashes"
	@echo "  make collect FORCE=1                              Wipe source docs and re-collect from scratch"
	@echo "  make collect SOURCES=sislaer                       Collect only SISLAER (primary source)"
	@echo "  make collect SOURCES=lexml                        Collect only LexML"
	@echo "  make collect SOURCES=decea LIMIT=50               Collect only DECEA, limit to 50 docs"
	@echo "  make collect SOURCES=anac_rbac                    Collect only ANAC RBACs"
	@echo "  make collect-sislaer                              Shortcut: SISLAER only"
	@echo "  make collect-legacy                               Shortcut: DECEA + LexML (fallback)"
	@echo "  make collect-anac                                 Shortcut: ANAC RBACs only"
	@echo "  make collect-anac CHECK=1                         Re-download all ANAC RBACs and verify hashes"
	@echo "  make collect-anac FORCE=1                         Wipe ANAC docs and re-collect from scratch"
	@echo "  make collect-anac LIMIT=5                         Collect only the first N RBACs (useful for tests)"
	@echo "  make collect SOURCES=pdf PDF_DIR=./data/pdfs      Collect local PDFs from directory"
	@echo "  make collect ALL_LOCALITIES=1                     Include state/municipal docs (default: federal only)"
	@echo "  make embed                                        Phase 2: generate embeddings (incremental)"
	@echo "  make embed MODE=dense                             Dense embeddings only"
	@echo "  make embed MODE=sparse                            Sparse embeddings only"
	@echo "  make embed MODE=hybrid                            Both dense + sparse"
	@echo "  make embed FORCE=1                                Re-embed everything"
	@echo "  make index                                        Phase 3: push embeddings to Qdrant"
	@echo "  make index RECREATE=1                             Drop + recreate Qdrant collection"
	@echo "  make pipeline                                     Run all 3 phases in sequence"
	@echo ""
	@echo "  ── validation & evaluation ───────────────────────────────────────────"
	@echo "  make validate-data                                Quality report for DECEA (no changes)"
	@echo "  make validate-data CLEAN=1                        DECEA report + save cleaned snapshot"
	@echo "  make validate-lexml                               Quality report for LexML (no changes)"
	@echo "  make validate-lexml CLEAN=1                       LexML report + save cleaned snapshot"
	@echo "  make eval                                         Run both evaluations"
	@echo "  make eval-retrieval                               Run retrieval evaluation"
	@echo "  make eval-retrieval K=10                          Override K for retrieval"
	@echo "  make eval-retrieval SEARCH_MODE=hybrid            Evaluate with hybrid search"
	@echo "  make eval-generation                              Run generation evaluation"
	@echo "  make eval-generation SAMPLE=10                    Limit generation to 10 queries"
	@echo "  make extract-pipeline INPUT=path.csv              Extract per-stage RAG data into a single .xlsx"
	@echo "  make extract-pipeline INPUT=p.xlsx OUTPUT=out.xlsx K=3 SAMPLE=5"
	@echo "  make extract-pipeline INPUT=p.csv NO_GENERATE=1   Skip the generator stage (much faster)"
	@echo ""
	@echo "  ── analytics ─────────────────────────────────────────────────────────"
	@echo "  make query                                        Open interactive SQL console"
	@echo "  make query SQL='SELECT source, COUNT(*) ...'      Run a one-shot SQL query"
	@echo "  make explore                                      Open datasette web UI for the store"
	@echo ""
	@echo "  ── development ──────────────────────────────────────────────────────"
	@echo "  make install                                      Create venvs + install backend & web deps"
	@echo "  make install-backend                              Only backend (root) venv + requirements.txt"
	@echo "  make install-web                                  Only web/ venv + web/requirements.txt"
	@echo "  make install FORCE=1                              Re-create venvs from scratch"
	@echo "  make download-models                              Pre-download all ML models"
	@echo "  make download-models SKIP_OLLAMA=1                Skip Ollama pulls"
	@echo "  make start                                        Start API + Web (Ctrl+C to stop)"
	@echo "  make start-api                                    Start only the API server"
	@echo "  make start-web                                    Start only the Web server"
	@echo ""
	@echo "  ── backup & restore ──────────────────────────────────────────────────"
	@echo "  make backup                                       Snapshot Qdrant collection to data/backups/"
	@echo "  make restore FILE=data/backups/<snapshot>.snapshot Restore collection from snapshot file"
	@echo ""
	@echo "  ── utilities ─────────────────────────────────────────────────────────"
	@echo "  make migrate                                      Run database migrations"
	@echo "  make test                                         Run all unit tests"
	@echo "  make test FILE=tests/evaluation                   Run tests in a specific dir or file"
	@echo "  make lint                                         Run linter (ruff) — unused imports, etc."
	@echo "  make lint-fix                                     Auto-fix lint errors"
	@echo "  make clean                                        Remove evaluation result files"
	@echo ""
	@echo "  ── deploy ────────────────────────────────────────────────────────────"
	@echo "  make check                                        Verify server prerequisites"
	@echo "  make deploy                                       Deploy (git pull + deps + restart)"
	@echo "  make deploy-first                                 First-time setup + deploy"
	@echo "  make deploy-nginx                                 Deploy + update nginx snippet"

install: install-backend install-web

install-backend:
ifdef FORCE
	@echo "→ Removing existing $(VENV_DIR)/ ..."
	@rm -rf $(VENV_DIR)
endif
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "→ Creating backend venv at $(VENV_DIR)/ ..."; \
		$(PYTHON3) -m venv $(VENV_DIR); \
	else \
		echo "→ Reusing existing $(VENV_DIR)/ (use FORCE=1 to recreate)"; \
	fi
	@echo "→ Upgrading pip ..."
	@$(VENV_DIR)/bin/pip install --upgrade pip -q
	@echo "→ Installing backend requirements ..."
	@$(VENV_DIR)/bin/pip install -r requirements.txt
	@echo "✓ Backend ready. Activate with: source $(VENV_DIR)/bin/activate"

install-web:
ifdef FORCE
	@echo "→ Removing existing $(WEB_VENV_DIR)/ ..."
	@rm -rf $(WEB_VENV_DIR)
endif
	@if [ ! -f web/requirements.txt ]; then \
		echo "× web/requirements.txt not found — skipping web install"; \
		exit 0; \
	fi
	@if [ ! -d "$(WEB_VENV_DIR)" ]; then \
		echo "→ Creating web venv at $(WEB_VENV_DIR)/ ..."; \
		$(PYTHON3) -m venv $(WEB_VENV_DIR); \
	else \
		echo "→ Reusing existing $(WEB_VENV_DIR)/ (use FORCE=1 to recreate)"; \
	fi
	@echo "→ Upgrading pip ..."
	@$(WEB_VENV_DIR)/bin/pip install --upgrade pip -q
	@echo "→ Installing web requirements ..."
	@$(WEB_VENV_DIR)/bin/pip install -r web/requirements.txt
	@echo "✓ Web ready. Activate with: source $(WEB_VENV_DIR)/bin/activate"

test:
	$(PYTHON) -m pytest $(or $(FILE),tests/) -v --tb=short

lint:
	$(PYTHON) -m ruff check .

lint-fix:
	$(PYTHON) -m ruff check --fix .

eval: eval-retrieval eval-generation

eval-retrieval:
	$(PYTHON) -m evaluation.evaluate_retrieval --k $(K) --workers $(WORKERS) --search-mode $(SEARCH_MODE)

eval-generation:
	$(PYTHON) -m evaluation.evaluate_generation --k $(K) $(if $(SAMPLE),--sample $(SAMPLE),)

extract-pipeline:
	@test -n "$(INPUT)" || (echo "Usage: make extract-pipeline INPUT=<path.csv|.xlsx> [OUTPUT=<path.xlsx>] [K=5] [SAMPLE=N] [NO_GENERATE=1]" && exit 1)
	$(PYTHON) -m scripts.extract_pipeline_data --input "$(INPUT)" $(if $(OUTPUT),--output "$(OUTPUT)",) --k $(K) $(if $(SAMPLE),--sample $(SAMPLE),) $(if $(NO_GENERATE),--no-generate,)

validate-data:
ifdef CLEAN
	$(PYTHON) -m scripts.validate_data --data-dir data/decea --clean
else
	$(PYTHON) -m scripts.validate_data --data-dir data/decea --report-only
endif

validate-lexml:
ifdef CLEAN
	$(PYTHON) -m scripts.validate_data --data-dir data/lexml --clean
else
	$(PYTHON) -m scripts.validate_data --data-dir data/lexml --report-only
endif

collect:
	$(PYTHON) -m scripts.collect --sources $(SOURCES) --limit $(LIMIT) --concurrency $(CONCURRENCY) $(if $(DOC_TYPES),--doc-types $(DOC_TYPES),) --pdf-dir $(PDF_DIR) $(if $(KEYWORDS),--keywords $(KEYWORDS),) $(if $(CHECK),--check,) $(if $(FORCE),--force,) $(if $(ALL_LOCALITIES),--no-federal-only,)

collect-sislaer:
	$(PYTHON) -m scripts.collect --sources sislaer --limit $(LIMIT) --concurrency $(CONCURRENCY) $(if $(DOC_TYPES),--doc-types $(DOC_TYPES),) $(if $(CHECK),--check,) $(if $(FORCE),--force,)

collect-legacy:
	$(PYTHON) -m scripts.collect --sources decea,lexml --limit $(LIMIT) --concurrency $(CONCURRENCY) $(if $(DOC_TYPES),--doc-types $(DOC_TYPES),) $(if $(KEYWORDS),--keywords $(KEYWORDS),) $(if $(CHECK),--check,) $(if $(FORCE),--force,) $(if $(ALL_LOCALITIES),--no-federal-only,)

collect-anac:
	$(PYTHON) -m scripts.collect --sources anac_rbac --limit $(LIMIT) --concurrency $(CONCURRENCY) $(if $(CHECK),--check,) $(if $(FORCE),--force,)

embed:
	$(PYTHON) -m scripts.embed $(if $(MODE),--mode $(MODE),) $(if $(FORCE),--force,) $(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) $(if $(EMBED_BATCH),--embed-batch $(EMBED_BATCH),)

index:
	$(PYTHON) -m scripts.index --workers $(WORKERS) $(if $(RECREATE),--recreate,) $(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),)

pipeline: collect embed index

query:
	$(PYTHON) -m scripts.query $(if $(SQL),--sql "$(SQL)",)

explore:
	$(PYTHON) -m datasette serve --immutable $(STORE_DB) --metadata metadata.yml --open --setting sql_time_limit_ms 30000

migrate:
	@$(PYTHON) -c "from pipeline.document_store import DocumentStore; store = DocumentStore(); store.close()"

clean:
	rm -f evaluation/results/*.csv evaluation/results/*.json

download-models:
	$(PYTHON) -m scripts.download_models $(if $(SKIP_OLLAMA),--skip-ollama,) $(if $(SKIP_EMBEDDINGS),--skip-embeddings,) $(if $(SKIP_CROSS_ENCODER),--skip-cross-encoder,)

start-api:
	$(PYTHON) -m uvicorn api.server:app --host 127.0.0.1 --port 8083 --reload

start-web:
	cd web && API_BASE_URL=http://127.0.0.1:8083 ROOT_PATH= $(PYTHON) -m uvicorn main:app --host 127.0.0.1 --port 8082 --reload

start:
	@echo "API  →  http://127.0.0.1:8083"
	@echo "Web  →  http://127.0.0.1:8082"
	@echo "Ctrl+C to stop both"
	@echo ""
	@trap 'kill 0' EXIT; \
	$(PYTHON) -m uvicorn api.server:app --host 127.0.0.1 --port 8083 --reload & \
	cd web && API_BASE_URL=http://127.0.0.1:8083 ROOT_PATH= $(PYTHON) -m uvicorn main:app --host 127.0.0.1 --port 8082 --reload & \
	wait

deploy:
	@sudo bash deploy/deploy.sh

deploy-first:
	@sudo bash deploy/deploy.sh --first-run

deploy-nginx:
	@sudo bash deploy/deploy.sh --with-nginx

check:
	@bash deploy/deploy.sh --check-only

# ── Qdrant Backup & Restore ──────────────────────────────────────────────────

backup:
	@curl -sf "http://$(QDRANT_HOST):$(QDRANT_PORT)/healthz" > /dev/null \
	  || (echo "Error: Qdrant not reachable at $(QDRANT_HOST):$(QDRANT_PORT)" && exit 1)
	@mkdir -p $(BACKUP_DIR)
	@echo "Creating Qdrant snapshot for '$(QDRANT_COLLECTION)' …"
	@SNAP_NAME=$$(curl -sf -X POST \
	  "http://$(QDRANT_HOST):$(QDRANT_PORT)/collections/$(QDRANT_COLLECTION)/snapshots" \
	  | $(PYTHON) -c "import sys,json; print(json.load(sys.stdin)['result']['name'])") && \
	echo "Snapshot created: $$SNAP_NAME" && \
	echo "Downloading to $(BACKUP_DIR)/$$SNAP_NAME …" && \
	curl -sf -o "$(BACKUP_DIR)/$$SNAP_NAME" \
	  "http://$(QDRANT_HOST):$(QDRANT_PORT)/collections/$(QDRANT_COLLECTION)/snapshots/$$SNAP_NAME" && \
	FILE_SIZE=$$(du -h "$(BACKUP_DIR)/$$SNAP_NAME" | cut -f1) && \
	echo "Backup saved: $(BACKUP_DIR)/$$SNAP_NAME ($$FILE_SIZE)" && \
	echo "Restore with: make restore FILE=$(BACKUP_DIR)/$$SNAP_NAME"

restore:
ifndef FILE
	@echo "Usage: make restore FILE=data/backups/<snapshot-name>.snapshot"
	@echo ""
	@echo "Available snapshots:"
	@ls -lh $(BACKUP_DIR)/*.snapshot 2>/dev/null || echo "  (none found in $(BACKUP_DIR)/)"
	@exit 1
endif
	@test -f "$(FILE)" || (echo "File not found: $(FILE)" && exit 1)
	@FILE_SIZE=$$(du -h "$(FILE)" | cut -f1) && \
	echo "Restoring '$(QDRANT_COLLECTION)' from $(FILE) ($$FILE_SIZE) …"
	@curl -sf -X POST \
	  "http://$(QDRANT_HOST):$(QDRANT_PORT)/collections/$(QDRANT_COLLECTION)/snapshots/upload?priority=snapshot" \
	  -H "Content-Type: multipart/form-data" \
	  -F "snapshot=@$(FILE)" && \
	echo "" && echo "Restore complete." && \
	echo "Verifying …" && \
	curl -sf "http://$(QDRANT_HOST):$(QDRANT_PORT)/collections/$(QDRANT_COLLECTION)" \
	  | $(PYTHON) -c "import sys,json; i=json.load(sys.stdin)['result']; print(f\"  Points: {i['points_count']}, Status: {i['status']}\")"
