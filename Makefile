# Common developer tasks. Requires GNU Make.
# Launch needs two terminals: `make backend` and `make frontend`.

PYTHON ?= ./venv/bin/python
PIP    ?= ./venv/bin/pip
NPM    ?= npm

.PHONY: help install backend frontend notebook clean

help:
	@echo "Targets:"
	@echo "  make install    Create venv (if needed), pip + npm install"
	@echo "  make backend    API on http://127.0.0.1:8000  (reload)"
	@echo "  make frontend   UI on  http://localhost:3000   (proxies API)"
	@echo "  make notebook   Jupyter for notebook-backtest.ipynb"
	@echo "  make clean      Remove caches / build artifacts"
	@echo ""
	@echo "Day-to-day: run backend and frontend in two terminals, then open"
	@echo "http://localhost:3000"

install:
	@test -d venv || python3 -m venv venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	cd frontend && $(NPM) install

backend:
	$(PYTHON) run.py

frontend:
	cd frontend && $(NPM) run dev

notebook:
	$(PIP) show jupyter >/dev/null 2>&1 || $(PIP) install jupyter
	$(PYTHON) -m jupyter notebook notebook-backtest.ipynb

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	rm -rf frontend/dist frontend/.vite *.egg-info
	rm -f detailed_strategy_log.txt
