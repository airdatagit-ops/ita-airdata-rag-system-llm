.PHONY: test eval eval-retrieval eval-generation validate-data validate-lexml clean help collect collect-sislaer collect-legacy embed index pipeline query explore migrate deploy deploy-first deploy-nginx check start start-api start-web

PYTHON ?= python
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
	@echo "  make collect-sislaer                              Shortcut: SISLAER only"
	@echo "  make collect-legacy                               Shortcut: DECEA + LexML (fallback)"
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
	@echo ""
	@echo "  ── analytics ─────────────────────────────────────────────────────────"
	@echo "  make query                                        Open interactive SQL console"
	@echo "  make query SQL='SELECT source, COUNT(*) ...'      Run a one-shot SQL query"
	@echo "  make explore                                      Open datasette web UI for the store"
	@echo ""
	@echo "  ── development ──────────────────────────────────────────────────────"
	@echo "  make start                                        Start API + Web (Ctrl+C to stop)"
	@echo "  make start-api                                    Start only the API server"
	@echo "  make start-web                                    Start only the Web server"
	@echo ""
	@echo "  ── utilities ─────────────────────────────────────────────────────────"
	@echo "  make migrate                                      Run database migrations"
	@echo "  make test                                         Run all unit tests"
	@echo "  make test FILE=tests/evaluation                   Run tests in a specific dir or file"
	@echo "  make clean                                        Remove evaluation result files"
	@echo ""
	@echo "  ── deploy ────────────────────────────────────────────────────────────"
	@echo "  make check                                        Verify server prerequisites"
	@echo "  make deploy                                       Deploy (git pull + deps + restart)"
	@echo "  make deploy-first                                 First-time setup + deploy"
	@echo "  make deploy-nginx                                 Deploy + update nginx snippet"

test:
	$(PYTHON) -m pytest $(or $(FILE),tests/) -v --tb=short

eval: eval-retrieval eval-generation

eval-retrieval:
	$(PYTHON) -m evaluation.evaluate_retrieval --k $(K) --workers $(WORKERS) --search-mode $(SEARCH_MODE)

eval-generation:
	$(PYTHON) -m evaluation.evaluate_generation --k $(K) $(if $(SAMPLE),--sample $(SAMPLE),)

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

start-api:
	$(PYTHON) -m uvicorn api.server:app --host 127.0.0.1 --port 8083 --reload

start-web:
	cd web && $(PYTHON) -m uvicorn main:app --host 127.0.0.1 --port 8082 --reload

start:
	@echo "API  →  http://127.0.0.1:8083"
	@echo "Web  →  http://127.0.0.1:8082"
	@echo "Ctrl+C to stop both"
	@echo ""
	@trap 'kill 0' EXIT; \
	$(PYTHON) -m uvicorn api.server:app --host 127.0.0.1 --port 8083 --reload & \
	cd web && $(PYTHON) -m uvicorn main:app --host 127.0.0.1 --port 8082 --reload & \
	wait

deploy:
	@sudo bash deploy/deploy.sh

deploy-first:
	@sudo bash deploy/deploy.sh --first-run

deploy-nginx:
	@sudo bash deploy/deploy.sh --with-nginx

check:
	@bash deploy/deploy.sh --check-only
