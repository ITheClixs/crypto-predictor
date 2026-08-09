.PHONY: help setup data backtest report tables paper arxiv certificate test lint format typecheck check app clean

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
	@echo "  arxiv      Build the flat, self-contained arXiv tarball"
	@echo "  certificate  Rerun the certificate study, calibration and controls"
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

arxiv: paper
	./paper/build_arxiv.sh

# Everything the manuscript's certificate sections quote. The joint null is excluded
# deliberately: it is hours of refitting and is run on its own.
certificate:
	$(PY) audit/scripts/gen_forecasts.py
	$(PY) audit/scripts/certificate_study.py
	$(PY) audit/scripts/certificate_study.py --payoff sign
	$(PY) audit/scripts/positive_control.py
	$(PY) audit/scripts/power_head_to_head.py 400
	$(PY) audit/scripts/certificate_calibration.py 400
	$(PY) audit/scripts/garch_null.py 400
	$(PY) audit/scripts/execution_contrast.py
	$(PY) audit/scripts/bootstrap_stability.py
	$(PY) audit/scripts/plot_certificate.py

test:
	$(PY) -m pytest --cov=cryptoforecast --cov=alphacert --cov-report=term-missing

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
