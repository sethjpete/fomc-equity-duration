"""CRSP daily and Compustat annual via WRDS.

Credentials: set WRDS_USERNAME in .env. On first run the `wrds` package asks for
your password once and offers to save it to ~/.pgpass (outside the repo).

Output is written year by year as parquet to data/raw/crsp/ and data/raw/compustat/.

CRSP format split: WRDS froze the legacy SIZ tables (crsp.dsf / dsenames /
dsedelist) at the end of 2024. Years up to `wrds.legacy_end_year` come from the
legacy tables; later years come from the CIZ flat file crsp.dsf_v2, renamed to the
legacy schema by `normalize_ciz` so everything downstream sees one format.

CIZ -> legacy mapping (WRDS's published crosswalk):
- shrcd in (10, 11)  ~ sharetype='NS', securitytype='EQTY', securitysubtype='COM',
                       usincflg='Y', issuertype in ('ACOR','CORP')
- exchcd in (1, 2, 3) ~ primaryexch in ('N','A','Q'), conditionaltype='RW',
                       tradingstatusflg='A'
- CIZ dlyret already includes the delisting return on the last trading day, so
  CIZ years need no separate delisting merge.

VERIFY ON FIRST RUN (cannot be checked without WRDS access): the dsf_v2 column
names below, and that `python scripts/01_pull_data.py crsp_overlap` followed by
`python scripts/02_data_report.py` shows the 2024 overlap agreeing.
"""

from __future__ import annotations

import os

import pandas as pd

from fomcdur.config import raw_dir, settings


def connect():
    try:
        import wrds
    except ImportError as e:
        raise ImportError("The wrds package is missing; reinstall with `make install` "
                          "(pip install -e \".[dev]\")") from e

    user = os.environ.get("WRDS_USERNAME")
    if not user:
        raise OSError("Set WRDS_USERNAME in .env (see .env.example)")
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


CRSP_DSI_SQL = """
select date, vwretd, vwretx
from crsp.dsi
where date between '{start}' and '{end}'
"""


CIZ_DAILY_SQL = """
select permno, dlycaldt, dlyret, dlyretx, dlyprc, shrout, dlyvol, siccd, primaryexch
from crsp.dsf_v2
where dlycaldt between '{start}' and '{end}'
  and sharetype = 'NS' and securitytype = 'EQTY' and securitysubtype = 'COM'
  and usincflg = 'Y' and issuertype in ('ACOR', 'CORP')
  and primaryexch in ('N', 'A', 'Q')
  and conditionaltype = 'RW' and tradingstatusflg = 'A'
"""

CIZ_RENAME = {
    "dlycaldt": "date", "dlyret": "ret", "dlyretx": "retx", "dlyprc": "prc", "dlyvol": "vol",
}
PRIMARYEXCH_TO_EXCHCD = {"N": 1, "A": 2, "Q": 3}


def normalize_ciz(df: pd.DataFrame) -> pd.DataFrame:
    """Rename a dsf_v2 extract to the legacy dsf/dsenames schema used downstream.

    shrcd is set to 11 for every row: the CIZ filters above select ordinary common
    shares of US-incorporated firms, which is the union of legacy codes 10 and 11.
    """
    out = df.rename(columns=CIZ_RENAME).copy()
    out["date"] = pd.to_datetime(out["date"])
    out["exchcd"] = out.pop("primaryexch").map(PRIMARYEXCH_TO_EXCHCD).astype("Int64")
    out["shrcd"] = 11
    out["source"] = "ciz"
    cols = ["permno", "date", "ret", "retx", "prc", "shrout", "vol", "shrcd", "exchcd", "siccd",
            "source"]
    return out[cols]


def _pull_year(db, y: int, legacy: bool, shrcd: str, exchcd: str) -> pd.DataFrame:
    if legacy:
        q = CRSP_DAILY_SQL.format(start=f"{y}-01-01", end=f"{y}-12-31", shrcd=shrcd, exchcd=exchcd)
        df = db.raw_sql(q, date_cols=["date"])
        df["source"] = "legacy"
        return df
    q = CIZ_DAILY_SQL.format(start=f"{y}-01-01", end=f"{y}-12-31")
    return normalize_ciz(db.raw_sql(q, date_cols=["dlycaldt"]))


def pull_crsp(db=None, start_year: int | None = None, end_year: int | None = None) -> None:
    """Daily stock file, one parquet per year; legacy SIZ up to legacy_end_year, CIZ after."""
    s = settings()
    start_year = start_year or int(s["sample"]["start"][:4])
    end_year = end_year or int(s["sample"]["data_end"][:4])
    legacy_end = int(s["wrds"]["legacy_end_year"])
    db = db or connect()
    out = raw_dir("crsp")
    shrcd = ",".join(map(str, s["universe"]["share_codes"]))
    exchcd = ",".join(map(str, s["universe"]["exchange_codes"]))
    for y in range(start_year, end_year + 1):
        path = out / f"dsf_{y}.parquet"
        if path.exists():
            continue
        print(f"[crsp] {y} ({'legacy' if y <= legacy_end else 'CIZ'})")
        _pull_year(db, y, y <= legacy_end, shrcd, exchcd).to_parquet(path, index=False)
    db.raw_sql(
        CRSP_DELIST_SQL.format(start=f"{start_year}-01-01", end=f"{min(end_year, legacy_end)}-12-31"),
        date_cols=["date"],
    ).to_parquet(out / "dsedelist.parquet", index=False)


def pull_crsp_overlap(db=None, year: int | None = None) -> None:
    """Pull one legacy year again from CIZ, for the format check in the data report."""
    s = settings()
    year = year or int(s["wrds"]["legacy_end_year"])
    db = db or connect()
    out = raw_dir("crsp") / "overlap"
    out.mkdir(exist_ok=True)
    _pull_year(db, year, False, "", "").to_parquet(out / f"dsf_ciz_{year}.parquet", index=False)


def compare_overlap(legacy: pd.DataFrame, ciz: pd.DataFrame) -> dict:
    """Agreement between a legacy and a CIZ extract of the same period.

    Legacy returns are compared after the delisting adjustment would apply, so a
    handful of last-day differences are expected; everything else should match.
    """
    key = ["permno", "date"]
    m = legacy[key + ["ret", "prc"]].merge(ciz[key + ["ret", "prc"]], on=key, how="outer",
                                           suffixes=("_leg", "_ciz"), indicator=True)
    both = m[m["_merge"] == "both"]
    d = (both["ret_leg"] - both["ret_ciz"]).abs()
    return {
        "rows_legacy": len(legacy), "rows_ciz": len(ciz), "rows_both": len(both),
        "only_legacy": int((m["_merge"] == "left_only").sum()),
        "only_ciz": int((m["_merge"] == "right_only").sum()),
        "permnos_legacy": legacy["permno"].nunique(), "permnos_ciz": ciz["permno"].nunique(),
        "ret_corr": float(both["ret_leg"].corr(both["ret_ciz"])),
        "share_ret_diff_gt_1e-6": float((d > 1e-6).mean()),
        "max_abs_ret_diff": float(d.max()),
        "share_abs_prc_diff_gt_1c": float(((both["prc_leg"].abs() - both["prc_ciz"].abs()).abs()
                                           > 0.01).mean()),
    }


def pull_dsi(db=None, start: str = "1990-01-01", end: str = "2016-12-31") -> None:
    """CRSP daily value-weighted index returns with and without dividends (Chen's VAR: market
    return and dividend yield, 1990-2016)."""
    db = db or connect()
    db.raw_sql(CRSP_DSI_SQL.format(start=start, end=end), date_cols=["date"]).to_parquet(
        raw_dir("crsp") / "dsi.parquet", index=False)


def pull_compustat(db=None) -> None:
    s = settings()
    db = db or connect()
    out = raw_dir("compustat")
    # Start a few years early so lagged characteristics exist in 1994.
    start = f"{int(s['sample']['start'][:4]) - 3}-01-01"
    db.raw_sql(COMPUSTAT_SQL.format(start=start, end=s["sample"]["data_end"]),
               date_cols=["datadate"]).to_parquet(out / "funda.parquet", index=False)
    db.raw_sql(CCM_LINK_SQL, date_cols=["linkdt", "linkenddt"]).to_parquet(
        out / "ccm_link.parquet", index=False
    )


def load_crsp(years: list[int] | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    files = sorted(raw_dir("crsp").glob("dsf_*.parquet"))
    if years is not None:
        files = [f for f in files if int(f.stem.split("_")[1]) in years]
    if not files:
        raise FileNotFoundError("No CRSP files; run `make data-wrds` on a machine with WRDS access")
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
