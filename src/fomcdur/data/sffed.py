"""SF Fed monetary policy surprises (Bauer-Swanson), raw and orthogonalized.

TODO(M1): confirm the file layout and that coverage starts in 1994. Column names
below are the expected ones; adjust COLUMN_MAP after looking at the real file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from fomcdur.config import raw_dir, settings

COLUMN_MAP = {
    # file column      -> our name
    "Date": "date",
    "MPS": "mps",            # raw surprise (pp), 30-minute window
    "MPS_ORTH": "mps_orth",  # orthogonalized to pre-meeting public news
    "Unscheduled": "unscheduled",
}


def download() -> Path:
    url = settings()["sources"]["sffed_url"]
    dest = raw_dir("sffed") / "monetary_policy_surprises.xlsx"
    if not url:
        if not dest.exists():
            raise FileNotFoundError(
                f"Set sources.sffed_url in config/settings.toml or save the SF Fed file to {dest}"
            )
        return dest
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def load(scheduled_only: bool = True) -> pd.DataFrame:
    """One row per FOMC announcement: date, mps, mps_orth (pp), scheduled flag."""
    path = raw_dir("sffed") / "monetary_policy_surprises.xlsx"
    df = pd.read_excel(path)
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    missing = {"date", "mps"} - set(df.columns)
    if missing:
        raise KeyError(f"Update COLUMN_MAP; file has {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if "unscheduled" in df.columns:
        df["scheduled"] = df["unscheduled"].fillna(0).astype(int) == 0
    else:
        df["scheduled"] = True
    if scheduled_only:
        df = df[df["scheduled"]]
    keep = [c for c in ["date", "mps", "mps_orth", "scheduled"] if c in df.columns]
    return df[keep].sort_values("date").reset_index(drop=True)
