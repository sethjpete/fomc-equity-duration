"""Download raw data. Usage: python scripts/01_pull_data.py [fred french sffed crsp compustat]"""

import sys

from fomcdur.data import fred, french, sffed, wrds_pull

STEPS = {
    "fred": fred.download,
    "french": french.download,
    "sffed": sffed.download,
    "crsp": wrds_pull.pull_crsp,
    "compustat": wrds_pull.pull_compustat,
}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(STEPS):
        print(f"[pull] {name}")
        STEPS[name]()
