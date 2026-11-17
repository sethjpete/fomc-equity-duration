"""SF Fed monetary policy surprises (Bauer-Swanson 2023 update), raw and orthogonalized.

Source page: https://www.frbsf.org/research-and-insights/data-and-indicators/monetary-policy-surprises/
The Excel file has five sheets; we use "FOMC (update 2023)", one row per FOMC
announcement since 1988 with columns Date, Time, Unscheduled, FF1..TBOND, MPS,
MPS_ORTH and the six orthogonalization regressors.

Facts checked against the file downloaded 2026-11-17:
- Units: MPS and MPS_ORTH are in percentage points (Dec 2023 MPS = -0.082, i.e. -8bp).
- Coverage: 1988-02-04 to 2023-12-13. 239 scheduled announcements 1994-2023
  (8 a year, 7 in 2020); 17 unscheduled since 1994. Nothing after 2023 yet.
- MPS is missing for 2001-09-17 (post-9/11; Bauer-Swanson exclude it).
- MPS_ORTH is NA for Mar-Dec 2020 by design (COVID); use MPS there.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fomcdur.config import raw_dir, settings
from fomcdur.data import http

FILENAME = "monetary_policy_surprises.xlsx"
SHEET = "FOMC (update 2023)"

COLUMN_MAP = {
    # file column -> our name
    "Date": "date",
    "Time": "time",
    "Unscheduled": "unscheduled",
    "MPS": "mps",            # raw surprise (pp), 30-minute window, first PC of ED1-ED4
    "MPS_ORTH": "mps_orth",  # orthogonalized to pre-meeting public news (pp)
    "TNOTE02": "tnote02",    # 30-minute 2y Treasury futures yield change (pp)
    "TNOTE10": "tnote10",
}


def download() -> Path:
    url = settings()["sources"]["sffed_url"]
    dest = raw_dir("sffed") / FILENAME
    if not url:
        if not dest.exists():
            raise FileNotFoundError(
                f"Set sources.sffed_url in config/settings.toml or save the SF Fed file to {dest}"
            )
        return dest
    r = http.get(url)
    dest.write_bytes(r.content)
    return dest


def _to_date(col: pd.Series) -> pd.Series:
    """Dates may arrive as datetimes or as Excel serial numbers depending on cell format."""
    if pd.api.types.is_datetime64_any_dtype(col):  # date-formatted cells (the real file)
        return col.dt.normalize()
    num = pd.to_numeric(col, errors="coerce")
    # Excel serials for 1900-2173 lie in (0, 100000); anything else is not a serial.
    if num.notna().mean() > 0.9 and num.dropna().between(1, 100_000).all():
        return pd.to_datetime(num, unit="D", origin="1899-12-30").dt.normalize()
    return pd.to_datetime(col).dt.normalize()


def parse(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw FOMC sheet: one row per announcement."""
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    missing = {"date", "mps", "unscheduled"} - set(df.columns)
    if missing:
        raise KeyError(f"Update COLUMN_MAP; sheet has {list(df.columns)}")
    df = df.dropna(subset=["date"]).copy()
    df["date"] = _to_date(df["date"])
    for c in ["mps", "mps_orth", "tnote02", "tnote10"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")  # "NA" strings -> NaN
    df["scheduled"] = pd.to_numeric(df["unscheduled"], errors="coerce").fillna(1).astype(int) == 0
    if "time" in df.columns:
        df["time"] = df["time"].astype(str).str.strip()
    keep = [c for c in ["date", "time", "scheduled", "mps", "mps_orth", "tnote02", "tnote10"]
            if c in df.columns]
    return df[keep].sort_values(["date", *(["time"] if "time" in keep else [])]).reset_index(
        drop=True
    )


def load(scheduled_only: bool = True, start: str | None = None,
         drop_missing: bool = True) -> pd.DataFrame:
    """One row per FOMC announcement: date, time, scheduled, mps, mps_orth (pp), ...

    By default: scheduled meetings only, from settings sample.start, rows with
    missing MPS dropped (only 2001-09-17).
    """
    path = raw_dir("sffed") / FILENAME
    df = parse(pd.read_excel(path, sheet_name=SHEET))
    start = start or settings()["sample"]["start"]
    df = df[df["date"] >= pd.Timestamp(start)]
    if scheduled_only:
        df = df[df["scheduled"]]
    if drop_missing:
        df = df[df["mps"].notna()]
    return df.reset_index(drop=True)


def coverage_end() -> pd.Timestamp:
    return load(scheduled_only=False, start="1900-01-01")["date"].max()
