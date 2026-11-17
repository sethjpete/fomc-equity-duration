# Effective Equity Duration and Post-FOMC Drift

Seth Peterson, Kenzo Ota von Steun, Grigorios Chatziandreou

We replicate Chen (2022, JFQA) FOMC-day effective equity duration with daily CRSP
returns and the SF Fed monetary policy surprises (239 scheduled meetings,
1994-2023), then extend the event window to 20 trading days to ask whether
rate-sensitive stocks keep repricing after the announcement. The headline number
is the **same-day share** `D(0) / D(20)` for a long-short portfolio sorted on
fitted stock-level duration, estimated with post-announcement yield controls.

The writeup is in [`paper/`](paper/). The plan of work is in [ROADMAP.md](ROADMAP.md).

## Layout

```
config/settings.toml      sample dates, universe filters, surprise scaling, source URLs
src/fomcdur/
  config.py               paths (data/, results/) and settings loader
  data/                   one module per source: download() -> data/raw/<source>/, load()
    wrds_pull.py          CRSP daily + delistings, Compustat annual, CCM link (WRDS)
    sffed.py              SF Fed surprises, raw and orthogonalized
    fred.py               DGS2, DGS10
    french.py             FF5 + momentum daily, NYSE ME breakpoints, 49 industries
    chen.py               Chen's posted estimates (replication check)
  clean/
    events.py             event-day alignment, cumulative returns t..t+h, post-event yield changes
    universe.py           delisting returns, market equity, microcap flag
    characteristics.py    book equity, profitability, leverage, monthly z-scores
  portfolios.py           quantile sorts, industry neutralization, value-weighted returns
  stats/ols.py            OLS with HC1 standard errors
  replication/duration.py r = a - D s on announcement days
  extension/
    pooled_model.py       expanding-window pooled characteristic model -> D_hat_{i,m}
    horizon.py            D(h) curves with yield/factor controls, same-day share + bootstrap
  strategy/backtest.py    event positions, costs, Sharpe, FF5+Mom alpha, by decade
scripts/                  numbered pipeline entry points
tests/                    synthetic-data tests for every estimator
data/{raw,interim,processed}/   git-ignored
results/{tables,figures}/       small outputs, committed
paper/                    writeup
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # includes wrds
cp .env.example .env        # set WRDS_USERNAME and FRED_API_KEY
pytest -q
```

## Data

| Data | Module | Access |
|---|---|---|
| CRSP daily stock file + delistings | `data/wrds_pull.py` | WRDS |
| Compustat annual + CCM link | `data/wrds_pull.py` | WRDS |
| SF Fed monetary policy surprises | `data/sffed.py` | Free (Excel) |
| DGS2, DGS10 | `data/fred.py` | FRED API, free key ([get one](https://fred.stlouisfed.org/docs/api/api_key.html)); Treasury.gov fallback |
| Chen's posted durations (stock-level SAS files) | `data/chen.py` | Free |
| FF5, momentum, RF (daily) | `data/french.py` | Free |
| NYSE ME breakpoints | `data/french.py` | Free |
| 49-industry SIC definitions | `data/french.py` | Free |

```bash
make data-free     # FRED, French, SF Fed, Chen
make data-wrds     # CRSP (legacy to 2024, CIZ for 2025, plus a 2024 CIZ overlap) + Compustat
make report        # results/tables/data_report.md and fomc_calendar.csv
```

The SF Fed surprise file currently ends at 2023-12-13, so the main sample is the
239 scheduled meetings from 1994 to 2023 (`sample.end`). CRSP and Compustat are
pulled through `sample.data_end` (2025) so the sample can be extended later. Legacy CRSP tables stop at 2024,
so later years come from the CIZ table `crsp.dsf_v2`, renamed to the legacy schema.

**Sources use official APIs with keys where one exists** (FRED API, WRDS). Ken
French, the SF Fed and Chen publish static files only, so those are plain downloads.

**Credentials never go in the repo.** `WRDS_USERNAME` and `FRED_API_KEY` live in `.env`
(git-ignored). The `wrds` package stores the password in `~/.pgpass` in your home
directory. CRSP and Compustat extracts are licensed and are git-ignored under
`data/`; share them through a shared drive and point `FOMCDUR_DATA_DIR` at it.

## Conventions

- Surprises are in percentage points in the raw file; `duration.surprise_scale = 0.01`
  converts to decimals so duration is in years (10bp tightening x D=10 = -1%).
- Day 0 is the announcement day; windows are close-to-close; `h` counts trading days.
  Announcements at or after the 4pm close, or on non-trading days, move to the next
  trading day (only 2020-03-15 so far). Trading days come from the French daily file.
- Characteristics become usable 6 months after fiscal year end and are z-scored within month.
- Every estimate used for a portfolio date uses only meetings strictly before that date.
