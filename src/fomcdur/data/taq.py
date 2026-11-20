"""Intraday NBBO midpoints around FOMC announcements from WRDS Millisecond TAQ.

Access (checked 2026-11-19 on Seth's WRDS account):
- Millisecond TAQ is available from 2003-09-10 on: yearly schemas `taqm_YYYY` with daily
  tables `complete_nbbo_YYYYMMDD` (time_m, sym_root, sym_suffix, best_bid, best_ask, ...).
- Monthly TAQ (`taq`, 1993-2014) is NOT licensed for the account, so 1994-2003 meetings have
  no intraday data for now.
- CRSP link: `wrdsapps_link_crsp_taqm.taqmclink` (date, sym_root, sym_suffix, permno, ...).

Chen (2022) measures each stock's return as the change in the NBBO midpoint over
[s - 30, s + 10] minutes around the statement time s. For each meeting we store, per
symbol, the last valid NBBO midpoint at or before s - 30 and at or before s + 10 (looking
back at most `LOOKBACK_MIN` minutes, so stale quotes do not count).

Output: data/raw/taq/nbbo_mid_YYYYMMDD.parquet and data/raw/taq/taqmclink.parquet.
"""

from __future__ import annotations

import time as _time
from datetime import timedelta

import pandas as pd

from fomcdur.config import raw_dir

FIRST_DATE = pd.Timestamp("2003-09-10")
PRE_MIN, POST_MIN = 30, 10
LOOKBACK_MIN = 15
MAX_REL_SPREAD = 0.10  # drop quotes with (ask - bid) / mid above 10%

NBBO_SQL = """
select sym_root, sym_suffix,
       (array_agg((best_bid + best_ask) / 2 order by time_m desc)
          filter (where time_m <= '{t0}'))[1] as mid_pre,
       (array_agg((best_bid + best_ask) / 2 order by time_m desc)
          filter (where time_m <= '{t1}'))[1] as mid_post,
       max(time_m) filter (where time_m <= '{t0}') as t_pre,
       max(time_m) filter (where time_m <= '{t1}') as t_post,
       count(*) as n_quotes
from taqm_{year}.complete_nbbo_{ymd}
where time_m between '{lo}' and '{t1}'
  and best_bid > 0 and best_ask >= best_bid
  and (best_ask - best_bid) <= {max_spread} * (best_bid + best_ask) / 2
group by sym_root, sym_suffix
"""

LINK_SQL = "select * from wrdsapps_link_crsp_taqm.taqmclink where date in ({dates})"


def window(day: pd.Timestamp, hour: float) -> dict:
    """Clock times for the window around an announcement at `hour` (e.g. 14.25 = 2:15pm)."""
    s = pd.Timestamp(day).normalize() + timedelta(hours=float(hour))
    fmt = "%H:%M:%S"
    return {
        "t0": (s - timedelta(minutes=PRE_MIN)).strftime(fmt),
        "t1": (s + timedelta(minutes=POST_MIN)).strftime(fmt),
        "lo": (s - timedelta(minutes=PRE_MIN + LOOKBACK_MIN)).strftime(fmt),
    }


def nbbo_sql(day: pd.Timestamp, hour: float) -> str:
    day = pd.Timestamp(day)
    return NBBO_SQL.format(year=day.year, ymd=day.strftime("%Y%m%d"),
                           max_spread=MAX_REL_SPREAD, **window(day, hour))


def path_for(day: pd.Timestamp):
    return raw_dir("taq") / f"nbbo_mid_{pd.Timestamp(day):%Y%m%d}.parquet"


def pull_nbbo(meetings: pd.DataFrame, db=None, limit: int | None = None) -> pd.DataFrame:
    """meetings: columns day0, hour. Skips meetings already on disk; returns a timing log."""
    from fomcdur.data.wrds_pull import connect

    m = meetings[(meetings["day0"] >= FIRST_DATE) & meetings["hour"].notna()]
    m = m[[not path_for(d).exists() for d in m["day0"]]]
    if limit:
        m = m.head(limit)
    db = db or connect()
    log = []
    for r in m.itertuples():
        t = _time.time()
        df = db.raw_sql(nbbo_sql(r.day0, r.hour))
        df["date"] = pd.Timestamp(r.day0)
        df.to_parquet(path_for(r.day0), index=False)
        log.append({"day0": r.day0, "symbols": len(df), "seconds": round(_time.time() - t, 1)})
        print(f"[taq] {r.day0:%Y-%m-%d}: {len(df):,} symbols in {log[-1]['seconds']}s")
    return pd.DataFrame(log)


def pull_link(dates: list[pd.Timestamp], db=None) -> None:
    from fomcdur.data.wrds_pull import connect

    db = db or connect()
    q = LINK_SQL.format(dates=",".join(f"'{pd.Timestamp(d):%Y-%m-%d}'" for d in dates))
    db.raw_sql(q, date_cols=["date"]).to_parquet(raw_dir("taq") / "taqmclink.parquet",
                                                  index=False)


def load_nbbo() -> pd.DataFrame:
    files = sorted(raw_dir("taq").glob("nbbo_mid_*.parquet"))
    if not files:
        raise FileNotFoundError("No TAQ files; run `python scripts/01_pull_data.py taq`")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_link() -> pd.DataFrame:
    p = raw_dir("taq") / "taqmclink.parquet"
    if not p.exists():
        raise FileNotFoundError("No TAQ link file; run `python scripts/01_pull_data.py taq`")
    return pd.read_parquet(p)


def fomc_meetings() -> pd.DataFrame:
    """Scheduled main-sample meetings with their day 0 and announcement hour."""
    from fomcdur.clean.events import build_calendar
    from fomcdur.config import settings
    from fomcdur.data import french, sffed

    end = pd.Timestamp(settings()["sample"]["end"])
    s = sffed.load(scheduled_only=True)
    cal = build_calendar(s[s["date"] <= end], french.trading_days())
    return cal[["date", "day0", "hour", "mps"]].reset_index(drop=True)


def pull(limit: int | None = None) -> pd.DataFrame:
    """Link table, then NBBO midpoints for every meeting since 2003-09-10 not yet on disk.

    Set TAQ_LIMIT=<n> in the environment to pull only the first n missing meetings (timing).
    """
    import os

    limit = limit or (int(os.environ["TAQ_LIMIT"]) if os.environ.get("TAQ_LIMIT") else None)
    from fomcdur.data.wrds_pull import connect

    m = fomc_meetings()
    m = m[m["day0"] >= FIRST_DATE]
    db = connect()
    if not (raw_dir("taq") / "taqmclink.parquet").exists():
        pull_link(list(m["day0"]), db)
    return pull_nbbo(m, db, limit)


# ---------------------------------------------------------------- Chen-exact windows
# Chen (2022, p. 678) uses "simple average stock prices before and after time s": the mean
# NBBO midpoint over [s-30, s) and over (s, s+10]. s is taken from his Table A.1
# (config/chen_table_a1.csv). Stored separately in data/raw/taq_chen/.

NBBO_AVG_SQL = """
select sym_root, sym_suffix,
       avg((best_bid + best_ask) / 2) filter (where time_m >= '{t0}' and time_m < '{s}')
         as avg_pre,
       avg((best_bid + best_ask) / 2) filter (where time_m > '{s}' and time_m <= '{t1}')
         as avg_post,
       count(*) filter (where time_m >= '{t0}' and time_m < '{s}') as n_pre,
       count(*) filter (where time_m > '{s}' and time_m <= '{t1}') as n_post
from taqm_{year}.complete_nbbo_{ymd}
where time_m between '{t0}' and '{t1}'
  and best_bid > 0 and best_ask >= best_bid
  and (best_ask - best_bid) <= {max_spread} * (best_bid + best_ask) / 2
group by sym_root, sym_suffix
"""


def avg_window(day: pd.Timestamp, s_time: str) -> dict:
    """s_time 'HH:MM:SS' -> clock strings for s, s-30 and s+10 minutes."""
    s = pd.Timestamp(f"{pd.Timestamp(day):%Y-%m-%d} {s_time}")
    fmt = "%H:%M:%S"
    return {"s": s.strftime(fmt),
            "t0": (s - timedelta(minutes=PRE_MIN)).strftime(fmt),
            "t1": (s + timedelta(minutes=POST_MIN)).strftime(fmt)}


def nbbo_avg_sql(day: pd.Timestamp, s_time: str) -> str:
    day = pd.Timestamp(day)
    return NBBO_AVG_SQL.format(year=day.year, ymd=day.strftime("%Y%m%d"),
                               max_spread=MAX_REL_SPREAD, **avg_window(day, s_time))


def chen_path_for(day: pd.Timestamp):
    return raw_dir("taq_chen") / f"nbbo_avg_{pd.Timestamp(day):%Y%m%d}.parquet"


def pull_chen() -> pd.DataFrame:
    """Window-average NBBO midpoints at Chen's announcements since 2003-09 (resumable).

    Also extends the CRSP-TAQ link table to these dates if needed."""
    from fomcdur.data.wrds_pull import connect
    from fomcdur.replication.chen_exact import load_events

    ev = load_events()
    ev = ev[ev["date"] >= FIRST_DATE]
    db = connect()
    lp = raw_dir("taq") / "taqmclink.parquet"
    have = set(pd.read_parquet(lp, columns=["date"])["date"]) if lp.exists() else set()
    missing = [d for d in ev["date"] if d not in have]
    if missing:
        q = LINK_SQL.format(dates=",".join(f"'{d:%Y-%m-%d}'" for d in missing))
        new = db.raw_sql(q, date_cols=["date"])
        old = pd.read_parquet(lp) if lp.exists() else None
        (pd.concat([old, new]) if old is not None else new).to_parquet(lp, index=False)
    log = []
    for r in ev.itertuples():
        if chen_path_for(r.date).exists():
            continue
        t = _time.time()
        df = db.raw_sql(nbbo_avg_sql(r.date, r.time))
        df["date"] = r.date
        df.to_parquet(chen_path_for(r.date), index=False)
        log.append({"date": r.date, "symbols": len(df), "seconds": round(_time.time() - t, 1)})
        print(f"[taq_chen] {r.date:%Y-%m-%d}: {len(df):,} symbols in {log[-1]['seconds']}s",
              flush=True)
    return pd.DataFrame(log)


def load_chen_nbbo() -> pd.DataFrame:
    files = sorted(raw_dir("taq_chen").glob("nbbo_avg_*.parquet"))
    if not files:
        raise FileNotFoundError("No Chen TAQ files; run `python scripts/01_pull_data.py taq_chen`")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
