"""Ken French Data Library: daily FF5 + momentum, NYSE ME breakpoints, 49 industries."""

from __future__ import annotations

import io
import re
import zipfile

import pandas as pd
import requests

from fomcdur.config import raw_dir, settings

MISSING_CODES = (-99.99, -999.0)

FILES = {
    "ff5_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom_daily": "F-F_Momentum_Factor_daily_CSV.zip",
    "me_breakpoints": "ME_Breakpoints_CSV.zip",
    "siccodes49": "Siccodes49.zip",
}


def download() -> None:
    base = settings()["sources"]["french_base"]
    for name, fname in FILES.items():
        r = requests.get(base + fname, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            member = z.namelist()[0]
            (raw_dir("french") / f"{name}.txt").write_bytes(z.read(member))


def parse_daily_csv(text: str) -> pd.DataFrame:
    """Parse a French daily CSV: keep the first block of YYYYMMDD rows.

    Returns decimals (French files are in percent). Missing-value codes -99.99
    and -999 become NaN. Checked on the 2026-09 vintage of the FF5 and momentum
    daily files (header lines ",Mkt-RF,SMB,HML,RMW,CMA,RF" and ",Mom")."""
    lines = text.splitlines()
    header, rows = None, []
    for line in lines:
        cells = [c.strip() for c in line.split(",")]
        if re.fullmatch(r"\d{8}", cells[0] or ""):
            rows.append(cells)
        elif rows:
            break  # end of first (daily) block
        elif len(cells) > 1 and cells[0] == "" and any(cells[1:]):
            header = cells[1:]
    if header is None or not rows:
        raise ValueError("Could not find a daily data block")
    df = pd.DataFrame([r[1:] for r in rows], columns=header, dtype=float)
    df.index = pd.to_datetime([r[0] for r in rows], format="%Y%m%d")
    df.index.name = "date"
    df.columns = [c.strip().replace("-", "_") for c in df.columns]
    df = df.mask(df.isin(MISSING_CODES))
    return df / 100.0


def trading_days() -> pd.DatetimeIndex:
    """NYSE trading days, taken from the daily FF5 file (no CRSP needed)."""
    return load_factors().index


def load_factors() -> pd.DataFrame:
    """Daily Mkt_RF, SMB, HML, RMW, CMA, RF, Mom (decimals)."""
    ff5 = parse_daily_csv((raw_dir("french") / "ff5_daily.txt").read_text(errors="ignore"))
    mom = parse_daily_csv((raw_dir("french") / "mom_daily.txt").read_text(errors="ignore"))
    mom.columns = ["Mom"]
    return ff5.join(mom, how="inner")


def load_me_breakpoints() -> pd.DataFrame:
    """Monthly NYSE ME percentiles (5,10,...,100), in $ millions, indexed by month end."""
    text = (raw_dir("french") / "me_breakpoints.txt").read_text(errors="ignore")
    rows = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if re.fullmatch(r"\d{6}", cells[0] or ""):
            rows.append(cells)
    # Columns: yyyymm, number of NYSE firms, then 20 percentiles in $ millions.
    df = pd.DataFrame([r[2:22] for r in rows], dtype=float)
    df.columns = [f"p{q}" for q in range(5, 105, 5)]
    df.index = pd.to_datetime([r[0] for r in rows], format="%Y%m") + pd.offsets.MonthEnd(0)
    df.index.name = "month"
    return df


def load_siccodes49() -> pd.DataFrame:
    """Table of (ind49, ind_name, sic_lo, sic_hi)."""
    text = (raw_dir("french") / "siccodes49.txt").read_text(errors="ignore")
    out, cur = [], None
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\s+(\w+)\s+(.*)$", line)
        if m:
            cur = (int(m.group(1)), m.group(2))
            continue
        m = re.match(r"^\s*(\d{4})-(\d{4})", line)
        if m and cur:
            out.append((*cur, int(m.group(1)), int(m.group(2))))
    return pd.DataFrame(out, columns=["ind49", "ind_name", "sic_lo", "sic_hi"])


def assign_ff49(sic: pd.Series, table: pd.DataFrame | None = None) -> pd.Series:
    """Map SIC codes to FF49 industry numbers (49 = 'Other' if unmatched)."""
    table = load_siccodes49() if table is None else table
    out = pd.Series(49, index=sic.index, dtype="Int64")
    for row in table.itertuples():
        out[(sic >= row.sic_lo) & (sic <= row.sic_hi)] = row.ind49
    out[sic.isna()] = pd.NA
    return out
