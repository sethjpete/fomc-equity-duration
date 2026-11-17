PY ?= python

.PHONY: install data data-free data-wrds report test lint

install:            ## editable install with WRDS + dev extras
	$(PY) -m pip install -e ".[wrds,dev]"

data: data-free data-wrds

data-free:          ## FRED, Ken French, SF Fed surprises, Chen estimates (no credentials)
	$(PY) scripts/01_pull_data.py fred french sffed chen

data-wrds:          ## CRSP daily (legacy to 2024, CIZ after) + 2024 CIZ overlap + Compustat
	$(PY) scripts/01_pull_data.py crsp crsp_overlap compustat

report:             ## data report and event calendar -> results/tables/
	$(PY) scripts/02_data_report.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
