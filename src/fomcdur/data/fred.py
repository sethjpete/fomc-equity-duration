"""FRED daily Treasury yields (DGS2, DGS10). No API key needed."""

from __future__ import annotations

import io

import pandas as pd
import requests

from fomcdur.config import raw_dir, settings

SERIES = ("DGS2", "DGS10")


def download(series: tuple[str, ...] = SERIES) -> None:
    url = settings()["sources"]["fred_url"]
    for s in series:
        r = requests.get(url.format(series=s), timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        (raw_dir("fred") / f"{s}.csv").write_text(r.text)


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
