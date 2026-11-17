"""Chen (2022) posted duration estimates, for the replication check.

TODO(M2): confirm which file on Chen's website holds the portfolio durations,
save it to data/raw/chen/, and write the parser here.
"""

from __future__ import annotations

import pandas as pd

from fomcdur.config import raw_dir


def load() -> pd.DataFrame:
    files = sorted(raw_dir("chen").glob("*"))
    if not files:
        raise FileNotFoundError("Save Chen's posted estimates to data/raw/chen/ (see TODO)")
    raise NotImplementedError(f"Write a parser for {files[0].name}")
