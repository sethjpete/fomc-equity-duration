# Effective Equity Duration and Post-FOMC Drift

Seth Peterson, Kenzo Ota von Steun, Grigorios Chatziandreou

We replicate Chen (2022, JFQA) FOMC-day effective equity duration with daily CRSP
returns and the SF Fed monetary policy surprises (~250 scheduled meetings,
1994-2025), then extend the event window to 20 trading days to ask whether
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
pip install -e ".[wrds,dev]"
cp .env.example .env        # set WRDS_USERNAME
pytest -q
```

## Data

| Data | Module | Access |
|---|---|---|
| CRSP daily stock file + delistings | `data/wrds_pull.py` | WRDS |
| Compustat annual + CCM link | `data/wrds_pull.py` | WRDS |
| SF Fed monetary policy surprises | `data/sffed.py` | Free (Excel) |
| DGS2, DGS10 | `data/fred.py` | Free |
| Chen's posted durations | `data/chen.py` | Free |
| FF5, momentum, RF (daily) | `data/french.py` | Free |
| NYSE ME breakpoints | `data/french.py` | Free |
| 49-industry SIC definitions | `data/french.py` | Free |

```bash
make data-free     # FRED, French, SF Fed
make data-wrds     # CRSP + Compustat (prompts for your WRDS password once)
```

**Credentials never go in the repo.** Only `WRDS_USERNAME` lives in `.env`
(git-ignored). The `wrds` package stores the password in `~/.pgpass` in your home
directory. CRSP and Compustat extracts are licensed and are git-ignored under
`data/`; share them through a shared drive and point `FOMCDUR_DATA_DIR` at it.

## Conventions

- Surprises are in percentage points in the raw file; `duration.surprise_scale = 0.01`
  converts to decimals so duration is in years (10bp tightening x D=10 = -1%).
- Day 0 is the announcement day; windows are close-to-close; `h` counts trading days.
- Characteristics become usable 6 months after fiscal year end and are z-scored within month.
- Every estimate used for a portfolio date uses only meetings strictly before that date.
