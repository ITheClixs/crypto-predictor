.PHONY: help setup data backtest report tables paper test lint format typecheck check app clean

PY := ./venv/bin/python
PIP := ./venv/bin/pip

help:
	@echo "Targets:"
	@echo "  setup      Install the package + dev extras into ./venv"
	@echo "  data       Download & cache market data"
	@echo "  backtest   Run the full walk-forward study (writes reports/)"
	@echo "  report     Rebuild reports/results.md + figures from cached runs"
	@echo "  tables     Emit the manuscript's LaTeX tables from reports/results.csv"
	@echo "  paper      Typeset paper/paper.pdf (needs tectonic)"
	@echo "  test       Run the test suite with coverage"
	@echo "  lint       Ruff lint"
	@echo "  format     Black + ruff --fix"
	@echo "  typecheck  Mypy"
	@echo "  check      lint + typecheck + test (CI gate)"
	@echo "  app        Launch the Flask demo on :8000"

setup:
	$(PIP) install -e ".[dev,demo]"

data:
	$(PY) -m cryptoforecast.cli data

backtest:
	$(PY) -m cryptoforecast.cli backtest

report:
	$(PY) -m cryptoforecast.cli report

tables:
	$(PY) -m cryptoforecast.cli tables --out paper/tables

# Tables come from the committed results, so the manuscript cannot quote a number
# the study does not produce.
paper: tables
	cd paper && tectonic -X compile paper.tex

test:
	$(PY) -m pytest --cov=cryptoforecast --cov-report=term-missing

lint:
	$(PY) -m ruff check src tests app

format:
	$(PY) -m black src tests app
	$(PY) -m ruff check --fix src tests app

typecheck:
	$(PY) -m mypy

check: lint typecheck test

app:
	$(PY) -m app.server

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path './venv/*' -exec rm -rf {} +
