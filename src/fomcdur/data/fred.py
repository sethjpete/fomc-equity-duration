"""Daily Treasury constant-maturity yields DGS2 and DGS10.

Primary source: the official FRED API (api.stlouisfed.org), keyed by FRED_API_KEY
in the git-ignored .env. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html.
The fredgraph.csv endpoint on fred.stlouisfed.org is not used: it sits behind
Akamai bot protection that drops Python HTTP clients.

Fallback (no key, or the API fails): the U.S. Treasury's daily par yield curve
CSVs, the source FRED copies DGS2/DGS10 from ("2 Yr" and "10 Yr" columns).

Both are written to data/raw/fred/<SERIES>.csv in FRED's CSV format, so `load` is
the same either way. A file you save there by hand is also picked up.
"""

from __future__ import annotations

import io
import os

import pandas as pd

from fomcdur.config import raw_dir, settings
from fomcdur.data import http

SERIES = ("DGS2", "DGS10")
TREASURY_COLUMNS = {"DGS2": "2 Yr", "DGS10": "10 Yr"}
TREASURY_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                "daily-treasury-rates.csv/{year}/all")
DOWNLOAD_START = "1990-01-01"  # enough history before the sample


def api_key() -> str | None:
    return os.environ.get("FRED_API_KEY") or None


def api_json_to_fred(payload: dict, series: str) -> str:
    """FRED API observations JSON -> FRED CSV text ('.' marks a missing value)."""
    obs = payload.get("observations")
    if obs is None:
        raise ValueError(f"FRED API error for {series}: {payload.get('error_message', payload)}")
    df = pd.DataFrame(obs)[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
    df.columns = ["observation_date", series]
    return df.to_csv(index=False)


def _from_api(series: str, key: str) -> str:
    try:
        r = http.get(settings()["sources"]["fred_api_url"],
                     params={"series_id": series, "api_key": key, "file_type": "json",
                             "observation_start": DOWNLOAD_START},
                     read_timeout=60, retries=3)
    except http.requests.HTTPError as e:
        # Re-raise without the request URL, which contains the key.
        body = e.response.json() if e.response is not None and e.response.content else {}
        raise ValueError(f"FRED API HTTP {getattr(e.response, 'status_code', '?')} for "
                         f"{series}: {body.get('error_message', '')}") from None
    return api_json_to_fred(r.json(), series)


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
    if not todo:
        print("  [fred] already on disk (use --force to refresh)")
        return
    key = api_key()
    failed = []
    if key is None:
        print("  [fred] FRED_API_KEY not set in .env; using Treasury.gov "
              "(free key: https://fred.stlouisfed.org/docs/api/api_key.html)")
        failed = todo
    else:
        for s in todo:
            try:
                (out / f"{s}.csv").write_text(_from_api(s, key))
                print(f"  [fred api] {s} ok")
            except Exception as e:  # noqa: BLE001 - fall back below
                msg = str(e) if isinstance(e, ValueError) else type(e).__name__
                print(f"  [fred api] {s} failed ({msg}); will try Treasury.gov")
                failed.append(s)
    if not failed:
        return
    try:
        for s, text in _from_treasury(tuple(failed)).items():
            (out / f"{s}.csv").write_text(text)
            print(f"  [treasury] {s} ok")
    except Exception as e:
        raise RuntimeError(
            f"Could not download {failed} from the FRED API or Treasury.gov ({e}). Save them by "
            f"hand: open https://fred.stlouisfed.org/series/<SERIES>, Download -> CSV, and save "
            f"as {out}/<SERIES>.csv"
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
