.PHONY: test eval eval-retrieval eval-generation collect-decea clean help

PYTHON ?= python
K ?= 5
WORKERS ?= 4
SAMPLE ?=
LIMIT ?= 100
FILE ?=
SEARCH_MODE ?= auto
DOC_TYPES ?= ICA

help:
	@echo "Usage:"
	@echo "  make test                               Run all unit tests"
	@echo "  make test FILE=tests/evaluation         Run tests in a specific dir or file"
  @echo "  make collect-decea                      Collect DECEA documents"
	@echo "  make collect-decea LIMIT=50 WORKERS=8   Custom collection"
	@echo "  make eval                               Run both evaluations"
	@echo "  make eval-retrieval                     Run retrieval evaluation"
	@echo "  make eval-retrieval K=10                Override K for retrieval"
	@echo "  make eval-retrieval SEARCH_MODE=hybrid  Evaluate with hybrid search"
	@echo "  make eval-generation                    Run generation evaluation"
	@echo "  make eval-generation SAMPLE=10          Limit generation to 10 queries"
	@echo "  make clean                              Remove evaluation result files"

test:
	$(PYTHON) -m pytest $(or $(FILE),tests/) -v --tb=short

collect-decea:
	$(PYTHON) -m scripts.ingest_decea --doc-types $(DOC_TYPES) --limit $(LIMIT) --workers $(WORKERS)
  
eval: eval-retrieval eval-generation

eval-retrieval:
	$(PYTHON) -m evaluation.evaluate_retrieval --k $(K) --workers $(WORKERS) --search-mode $(SEARCH_MODE)

eval-generation:
	$(PYTHON) -m evaluation.evaluate_generation --k $(K) $(if $(SAMPLE),--sample $(SAMPLE),)

clean:
	rm -f evaluation/results/*.csv evaluation/results/*.json
