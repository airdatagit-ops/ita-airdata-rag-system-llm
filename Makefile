.PHONY: test eval eval-retrieval eval-generation collect-decea collect-lexml benchmark-lexml validate-data validate-lexml clean help

PYTHON ?= python
K ?= 5
WORKERS ?= 4
SAMPLE ?=
LIMIT ?= 100
FILE ?=
SEARCH_MODE ?= auto
DOC_TYPES ?= ICA
DATA_DIR ?= data/decea
KEYWORDS ?=
CONCURRENCY ?= 5

help:
	@echo "Usage:"
	@echo "  make test                                         Run all unit tests"
	@echo "  make test FILE=tests/evaluation                   Run tests in a specific dir or file"
	@echo "  make collect-decea                                Collect DECEA documents (100 ICAs, 4 workers)"
	@echo "  make collect-decea LIMIT=50 WORKERS=8             Custom DECEA collection"
	@echo "  make collect-decea SKIP_DOWNLOAD=1                Ingest existing JSONs only"
	@echo "  make collect-lexml                                Collect LexML documents (100 docs, 5 parallel)"
	@echo "  make collect-lexml LIMIT=50 CONCURRENCY=3         Custom LexML collection"
	@echo "  make collect-lexml KEYWORDS='ANAC,portaria'       Custom keywords"
	@echo "  make benchmark-lexml                              Benchmark async LexML scraper (seq vs parallel)"
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
	@echo "  make clean                                        Remove evaluation result files"

test:
	$(PYTHON) -m pytest $(or $(FILE),tests/) -v --tb=short

collect-decea:
	$(PYTHON) -m scripts.ingest_decea --doc-types $(DOC_TYPES) --limit $(LIMIT) --workers $(WORKERS) $(if $(SKIP_DOWNLOAD),--skip-download,) --download-dir $(DATA_DIR)

collect-lexml:
	$(PYTHON) -m scripts.ingest_lexml --limit $(LIMIT) --concurrency $(CONCURRENCY) $(if $(KEYWORDS),--keywords $(KEYWORDS),) $(if $(SKIP_DOWNLOAD),--skip-download,) $(if $(FORCE_DOWNLOAD),--force-download,)

benchmark-lexml:
	$(PYTHON) -m scripts.benchmark_lexml --limit $(LIMIT) --download 10 $(if $(KEYWORDS),--keywords $(KEYWORDS),)
  
eval: eval-retrieval eval-generation

eval-retrieval:
	$(PYTHON) -m evaluation.evaluate_retrieval --k $(K) --workers $(WORKERS) --search-mode $(SEARCH_MODE)

eval-generation:
	$(PYTHON) -m evaluation.evaluate_generation --k $(K) $(if $(SAMPLE),--sample $(SAMPLE),)

validate-data:
ifdef CLEAN
	$(PYTHON) -m scripts.validate_data --data-dir $(DATA_DIR) --clean
else
	$(PYTHON) -m scripts.validate_data --data-dir $(DATA_DIR) --report-only
endif

validate-lexml:
ifdef CLEAN
	$(PYTHON) -m scripts.validate_data --data-dir data/lexml --clean
else
	$(PYTHON) -m scripts.validate_data --data-dir data/lexml --report-only
endif

clean:
	rm -f evaluation/results/*.csv evaluation/results/*.json
