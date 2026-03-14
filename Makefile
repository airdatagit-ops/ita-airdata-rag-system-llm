.PHONY: test eval eval-retrieval eval-generation clean help

PYTHON ?= python
K ?= 5
WORKERS ?= 4
SAMPLE ?=
FILE ?=
SEARCH_MODE ?= auto

help:
	@echo "Usage:"
	@echo "  make test                              Run all unit tests"
	@echo "  make test FILE=tests/evaluation         Run tests in a specific dir or file"
	@echo "  make eval                               Run both evaluations"
	@echo "  make eval-retrieval                     Run retrieval evaluation"
	@echo "  make eval-retrieval K=10                Override K for retrieval"
	@echo "  make eval-retrieval SEARCH_MODE=hybrid  Evaluate with hybrid search"
	@echo "  make eval-generation                    Run generation evaluation"
	@echo "  make eval-generation SAMPLE=10          Limit generation to 10 queries"
	@echo "  make clean                              Remove evaluation result files"

test:
	$(PYTHON) -m pytest $(or $(FILE),tests/) -v --tb=short

eval: eval-retrieval eval-generation

eval-retrieval:
	$(PYTHON) -m evaluation.evaluate_retrieval --k $(K) --workers $(WORKERS) --search-mode $(SEARCH_MODE)

eval-generation:
	$(PYTHON) -m evaluation.evaluate_generation --k $(K) $(if $(SAMPLE),--sample $(SAMPLE),)

clean:
	rm -f evaluation/results/*.csv evaluation/results/*.json
