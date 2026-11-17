"""CRSP daily and Compustat annual via WRDS.

Credentials: set WRDS_USERNAME in .env. On first run the `wrds` package asks for
your password once and offers to save it to ~/.pgpass (outside the repo).

Output is written year by year as parquet to data/raw/crsp/ and data/raw/compustat/.

NOTE(M1): WRDS froze the legacy CRSP SIZ tables (crsp.dsf / dsenames) at the end
of 2024. Data through 2025 needs the CIZ tables (crsp.dsf_v2 and friends). The
queries below use the legacy layout so the 1994-2024 pipeline can be built and
checked first; switching to CIZ is a tracked roadmap item.
"""

from __future__ import annotations

import os

import pandas as pd

from fomcdur.config import raw_dir, settings


def connect():
    import wrds  # optional dependency: pip install -e ".[wrds]"

    user = os.environ.get("WRDS_USERNAME")
    if not user:
        raise EnvironmentError("Set WRDS_USERNAME in .env (see .env.example)")
    return wrds.Connection(wrds_username=user)


CRSP_DAILY_SQL = """
select a.permno, a.date, a.ret, a.retx, a.prc, a.shrout, a.vol,
       b.shrcd, b.exchcd, b.siccd
from crsp.dsf as a
join crsp.dsenames as b
  on a.permno = b.permno
 and a.date between b.namedt and coalesce(b.nameendt, '9999-12-31')
where a.date between '{start}' and '{end}'
  and b.shrcd in ({shrcd})
  and b.exchcd in ({exchcd})
"""

CRSP_DELIST_SQL = """
select permno, dlstdt as date, dlret, dlstcd
from crsp.dsedelist
where dlstdt between '{start}' and '{end}'
"""

COMPUSTAT_SQL = """
select gvkey, datadate, fyear, seq, ceq, at, lt, txditc, pstkrv, pstkl, pstk,
       revt, cogs, xsga, xint, dltt, dlc, sich
from comp.funda
where indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C'
  and datadate between '{start}' and '{end}'
"""

CCM_LINK_SQL = """
select gvkey, lpermno as permno, linkdt, linkenddt, linktype, linkprim
from crsp.ccmxpf_lnkhist
where linktype in ('LU', 'LC') and linkprim in ('P', 'C')
"""


def pull_crsp(db=None, start_year: int | None = None, end_year: int | None = None) -> None:
    s = settings()
    start_year = start_year or int(s["sample"]["start"][:4])
    end_year = end_year or int(s["sample"]["end"][:4])
    db = db or connect()
    out = raw_dir("crsp")
    shrcd = ",".join(map(str, s["universe"]["share_codes"]))
    exchcd = ",".join(map(str, s["universe"]["exchange_codes"]))
    for y in range(start_year, end_year + 1):
        path = out / f"dsf_{y}.parquet"
        if path.exists():
            continue
        q = CRSP_DAILY_SQL.format(start=f"{y}-01-01", end=f"{y}-12-31", shrcd=shrcd, exchcd=exchcd)
        db.raw_sql(q, date_cols=["date"]).to_parquet(path, index=False)
    db.raw_sql(
        CRSP_DELIST_SQL.format(start=f"{start_year}-01-01", end=f"{end_year}-12-31"),
        date_cols=["date"],
    ).to_parquet(out / "dsedelist.parquet", index=False)


def pull_compustat(db=None) -> None:
    s = settings()
    db = db or connect()
    out = raw_dir("compustat")
    # Start a few years early so lagged characteristics exist in 1994.
    start = f"{int(s['sample']['start'][:4]) - 3}-01-01"
    db.raw_sql(COMPUSTAT_SQL.format(start=start, end=s["sample"]["end"]),
               date_cols=["datadate"]).to_parquet(out / "funda.parquet", index=False)
    db.raw_sql(CCM_LINK_SQL, date_cols=["linkdt", "linkenddt"]).to_parquet(
        out / "ccm_link.parquet", index=False
    )


def load_crsp() -> pd.DataFrame:
    files = sorted(raw_dir("crsp").glob("dsf_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
