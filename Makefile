.PHONY: test eval eval-retrieval eval-generation clean help

PYTHON ?= python
K ?= 5
WORKERS ?= 4
SAMPLE ?=
FILE ?=

help:
	@echo "Usage:"
	@echo "  make test                   Run all unit tests"
	@echo "  make test FILE=path         Run specific test file or directory"
	@echo "  make eval                   Run both evaluations"
	@echo "  make eval-retrieval         Run retrieval evaluation"
	@echo "  make eval-generation        Run generation evaluation"
	@echo "  make clean                  Remove evaluation result files"
	@echo ""
	@echo "Variables:"
	@echo "  K=10                        Number of results to retrieve (default: 5)"
	@echo "  WORKERS=8                   Parallel search workers (default: 4)"
	@echo "  SAMPLE=10                   Limit generation eval queries (default: all)"
	@echo "  FILE=tests/test_foo.py      Target for make test (default: tests/)"

test:
	$(PYTHON) -m pytest $(or $(FILE),tests/) -v --tb=short

eval: eval-retrieval eval-generation

eval-retrieval:
	$(PYTHON) -m evaluation.evaluate_retrieval --k $(K) --workers $(WORKERS)

eval-generation:
	$(PYTHON) -m evaluation.evaluate_generation --k $(K) $(if $(SAMPLE),--sample $(SAMPLE),)

clean:
	rm -f evaluation/results/*.csv evaluation/results/*.json
