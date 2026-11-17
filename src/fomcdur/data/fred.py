"""Daily Treasury constant-maturity yields DGS2 and DGS10. No API key needed.

Primary source: FRED's fredgraph CSV endpoint, which can be slow to respond.
Fallback: the U.S. Treasury's daily par yield curve CSVs (one file per year),
which are the source FRED copies DGS2/DGS10 from ("2 Yr" and "10 Yr" columns).
Both are written to data/raw/fred/<SERIES>.csv in FRED's format, so `load` is
the same either way. A file you save there by hand is also picked up.
"""

from __future__ import annotations

import io

import pandas as pd

from fomcdur.config import raw_dir, settings
from fomcdur.data import http

SERIES = ("DGS2", "DGS10")
TREASURY_COLUMNS = {"DGS2": "2 Yr", "DGS10": "10 Yr"}
TREASURY_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                "daily-treasury-rates.csv/{year}/all")
DOWNLOAD_START = "1990-01-01"  # enough history before the sample; keeps the response small


def _from_fred(series: str) -> str:
    url = settings()["sources"]["fred_url"].format(series=series)
    r = http.get(url, params={"cosd": DOWNLOAD_START}, read_timeout=180, retries=3)
    text = r.text
    if not text.startswith("observation_date") and not text.upper().startswith("DATE"):
        raise ValueError(f"FRED returned something other than a CSV for {series}: {text[:80]!r}")
    return text


def treasury_to_fred(frames: list[pd.DataFrame], series: str) -> str:
    """Convert Treasury par-curve CSVs (Date = MM/DD/YYYY) to FRED's CSV format."""
    col = TREASURY_COLUMNS[series]
    df = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({"observation_date": pd.to_datetime(df["Date"], format="%m/%d/%Y"),
                        series: pd.to_numeric(df[col], errors="coerce")})
    out = out.drop_duplicates("observation_date").sort_values("observation_date")
    out["observation_date"] = out["observation_date"].dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False)


def _treasury_year(year: int) -> pd.DataFrame:
    r = http.get(TREASURY_URL.format(year=year),
                 params={"type": "daily_treasury_yield_curve",
                         "field_tdr_date_value": year, "page": "", "_format": "csv"},
                 read_timeout=60, retries=2)
    return pd.read_csv(io.StringIO(r.text))


def _from_treasury(series_list: tuple[str, ...]) -> dict[str, str]:
    start = int(DOWNLOAD_START[:4])
    end = pd.Timestamp.today().year
    frames = []
    for y in range(start, end + 1):
        print(f"  [treasury] {y}")
        frames.append(_treasury_year(y))
    return {s: treasury_to_fred(frames, s) for s in series_list}


def download(series: tuple[str, ...] = SERIES, force: bool = False) -> None:
    out = raw_dir("fred")
    todo = [s for s in series if force or not (out / f"{s}.csv").exists()]
    failed = []
    for s in todo:
        try:
            (out / f"{s}.csv").write_text(_from_fred(s))
            print(f"  [fred] {s} ok")
        except Exception as e:  # noqa: BLE001 - fall back below
            print(f"  [fred] {s} failed ({type(e).__name__}: {e}); will try Treasury.gov")
            failed.append(s)
    if not failed:
        return
    try:
        for s, text in _from_treasury(tuple(failed)).items():
            (out / f"{s}.csv").write_text(text)
            print(f"  [treasury] {s} ok")
    except Exception as e:
        raise RuntimeError(
            f"Could not download {failed} from FRED or Treasury.gov ({e}). Save them by hand: "
            f"open https://fred.stlouisfed.org/series/<SERIES>, Download -> CSV, and save as "
            f"{out}/<SERIES>.csv"
        ) from e


def parse(text: str) -> pd.Series:
    # Header is "observation_date,<SERIES>"; holidays are blank (older files used ".").
    df = pd.read_csv(io.StringIO(text), na_values=".")
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].astype(float)


def load(series: tuple[str, ...] = SERIES) -> pd.DataFrame:
    """Daily yields in percent, one column per series, NaNs on holidays dropped."""
    out = {s: parse((raw_dir("fred") / f"{s}.csv").read_text()) for s in series}
    return pd.DataFrame(out).dropna(how="all")
