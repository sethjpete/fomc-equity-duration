"""Download raw data.

Usage: python scripts/01_pull_data.py [--force] [fred french sffed chen crsp crsp_overlap compustat]
With no step names, pulls the free sources only (WRDS steps must be named).
A failing step does not stop the others; failures are listed at the end.
FRED skips series already on disk unless --force is given.
"""

import sys
import traceback

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
    args = sys.argv[1:]
    force = "--force" in args
    names = [a for a in args if a != "--force"] or list(FREE)
    unknown = [n for n in names if n not in STEPS]
    if unknown:
        sys.exit(f"Unknown step(s) {unknown}; choose from {list(STEPS)}")
    failed = {}
    for name in names:
        print(f"[pull] {name}")
        try:
            STEPS[name](force=True) if (force and name == "fred") else STEPS[name]()
        except Exception as e:  # noqa: BLE001 - report all failures at the end
            traceback.print_exc()
            failed[name] = f"{type(e).__name__}: {e}"
    if failed:
        print("\nFailed steps:")
        for k, v in failed.items():
            print(f"  {k}: {v}")
        sys.exit(1)
    print("\nAll steps done.")
