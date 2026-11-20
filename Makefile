PY ?= python

.PHONY: install data data-free data-wrds report replicate data-taq intraday test lint

install:            ## editable install (includes wrds) + dev tools
	$(PY) -m pip install -e ".[dev]"

data: data-free data-wrds

data-free:          ## FRED, Ken French, SF Fed surprises, Chen estimates (no credentials)
	$(PY) scripts/01_pull_data.py fred french sffed chen

data-wrds:          ## CRSP daily (legacy to 2024, CIZ after) + 2024 CIZ overlap + Compustat
	$(PY) scripts/01_pull_data.py crsp crsp_overlap compustat

report:             ## data report and event calendar -> results/tables/
	$(PY) scripts/02_data_report.py

replicate:          ## M2 Chen (2022) replication -> results/tables/m2_*, results/figures/m2_*
	$(PY) scripts/03_replicate_chen.py

data-taq:           ## intraday NBBO midpoints at FOMC meetings from WRDS Millisecond TAQ (2003-09 on)
	$(PY) scripts/01_pull_data.py taq

intraday:           ## intraday vs daily vs Chen day-0 durations -> results/tables/m2_intraday.md
	$(PY) scripts/06_intraday_durations.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
