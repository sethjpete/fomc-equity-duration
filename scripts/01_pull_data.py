"""Download raw data.

Usage: python scripts/01_pull_data.py [fred french sffed chen crsp crsp_overlap compustat]
With no arguments, pulls the free sources only (WRDS steps must be named).
"""

import sys

from fomcdur.data import chen, fred, french, sffed, wrds_pull

FREE = {
    "fred": fred.download,
    "french": french.download,
    "sffed": sffed.download,
    "chen": chen.download,
}
WRDS = {
    "crsp": wrds_pull.pull_crsp,
    "crsp_overlap": wrds_pull.pull_crsp_overlap,
    "compustat": wrds_pull.pull_compustat,
}
STEPS = FREE | WRDS

if __name__ == "__main__":
    names = sys.argv[1:] or list(FREE)
    unknown = [n for n in names if n not in STEPS]
    if unknown:
        sys.exit(f"Unknown step(s) {unknown}; choose from {list(STEPS)}")
    for name in names:
        print(f"[pull] {name}")
        STEPS[name]()
