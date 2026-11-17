PY ?= python

.PHONY: install data data-free data-wrds test lint

install:            ## editable install with WRDS + dev extras
	$(PY) -m pip install -e ".[wrds,dev]"

data: data-free data-wrds

data-free:          ## FRED, Ken French, SF Fed surprises (no credentials)
	$(PY) scripts/01_pull_data.py fred french sffed

data-wrds:          ## CRSP daily + Compustat annual (needs WRDS_USERNAME / ~/.pgpass)
	$(PY) scripts/01_pull_data.py crsp compustat

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
