"""Settings and canonical paths. Everything else imports paths from here."""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

DATA = Path(os.environ.get("FOMCDUR_DATA_DIR", ROOT / "data"))
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"


@lru_cache
def settings() -> dict:
    with open(ROOT / "config" / "settings.toml", "rb") as f:
        return tomllib.load(f)


def raw_dir(source: str) -> Path:
    p = RAW / source
    p.mkdir(parents=True, exist_ok=True)
    return p
