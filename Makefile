.PHONY: test test-retrieval test-generation eval eval-retrieval eval-generation lint clean help

PYTHON ?= python
K ?= 5
WORKERS ?= 4
SAMPLE ?=

help:
	@echo "Usage:"
	@echo "  make test                 Run all unit tests"
	@echo "  make test-retrieval       Run retrieval evaluation tests"
	@echo "  make test-generation      Run generation evaluation tests"
	@echo "  make eval                 Run both evaluations"
	@echo "  make eval-retrieval       Run retrieval evaluation (K=$(K), WORKERS=$(WORKERS))"
	@echo "  make eval-generation      Run generation evaluation (K=$(K), SAMPLE=$(SAMPLE))"
	@echo "  make lint                 Check code with linter"
	@echo "  make clean                Remove evaluation result files"
	@echo ""
	@echo "Variables:"
	@echo "  K=10                      Number of results to retrieve (default: 5)"
	@echo "  WORKERS=8                 Parallel workers (default: 4)"
	@echo "  SAMPLE=10                 Limit generation eval queries (default: all)"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-retrieval:
	$(PYTHON) -m pytest tests/test_evaluate_retrieval.py -v --tb=short

test-generation:
	$(PYTHON) -m pytest tests/test_evaluate_generation.py -v --tb=short

eval: eval-retrieval eval-generation

eval-retrieval:
	$(PYTHON) -m evaluation.evaluate_retrieval --k $(K) --workers $(WORKERS)

eval-generation:
	$(PYTHON) -m evaluation.evaluate_generation --k $(K) --workers 1 $(if $(SAMPLE),--sample $(SAMPLE),)

lint:
	$(PYTHON) -m py_compile evaluation/evaluate_retrieval.py
	$(PYTHON) -m py_compile evaluation/evaluate_generation.py

clean:
	rm -f evaluation/results/*.csv evaluation/results/*.json
