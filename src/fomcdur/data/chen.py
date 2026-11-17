"""Chen (2022, JFQA) posted effective equity duration estimates.

Zhanhui Chen, "Inferring Stock Duration Around FOMC Surprises: Estimates and
Implications", JFQA 57(2). Posted on https://chenzhanhui.people.ust.hk/publications.html
as three SAS datasets (checked 2026-11-17):

- duration_table1.sas7bdat: effective duration (DURATION) vs Dechow-Sloan-Soliman
  duration (DSSDURATION), Table 1. Common stocks, NYSE/Amex/Nasdaq, both measures
  available, ex financials (SIC 6000-6999) and utilities (4900-4999).
- duration_size_return_table2.sas7bdat: estimates used in the main tables (Table 2),
  same filters plus ex NYSE-20th-percentile microcaps; also SIZE at the end of the
  event month and next-month RET / RETURNDATE.
- duration_all_estimates.sas7bdat: all estimates, no filters.

These are stock-level estimates, not portfolio durations. For M2 we compare our
day-0 durations against portfolio aggregates of these (e.g. size/BM deciles).
Column names are lower-cased on load; run `python scripts/02_data_report.py` after
download to print their schema.
"""

from __future__ import annotations

import pandas as pd

from fomcdur.config import raw_dir, settings
from fomcdur.data import http

FILES = {
    "table1": "duration_table1.sas7bdat",
    "table2": "duration_size_return_table2.sas7bdat",
    "all": "duration_all_estimates.sas7bdat",
}


def download() -> None:
    base = settings()["sources"]["chen_base"]
    for fname in FILES.values():
        r = http.get(base + fname)
        (raw_dir("chen") / fname).write_bytes(r.content)


def load(which: str = "table2") -> pd.DataFrame:
    path = raw_dir("chen") / FILES[which]
    if not path.exists():
        raise FileNotFoundError(f"Run `python scripts/01_pull_data.py chen` first ({path})")
    df = pd.read_sas(path, format="sas7bdat", encoding="latin-1")
    df.columns = [c.lower() for c in df.columns]
    return df
