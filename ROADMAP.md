# Roadmap

Milestones are in dependency order. Each ends with a concrete check so we know it is done.
Tick boxes as work lands; open a GitHub issue per bullet when someone picks it up.

## M0. Infrastructure (done in the initial commit)
- [x] Package layout, settings, `.gitignore` for data and secrets, tests (`make test`,
      run locally before each PR; no CI)
- [x] Estimators with synthetic-data tests: event duration, D(h) curve, same-day share
      bootstrap, pooled model, quintile/VW portfolios, event strategy
- [ ] Everyone runs `pip install -e ".[wrds,dev]"` and `pytest` locally
- [ ] Decide where shared data lives (shared drive path for `FOMCDUR_DATA_DIR`)

## M1. Data pipeline
- [x] SF Fed surprises: download URL and column names confirmed (sheet "FOMC (update 2023)",
      MPS / MPS_ORTH in pp). Coverage 1988 to 2023-12-13: **239 scheduled meetings 1994-2023**
- [x] **SF Fed coverage gap:** no surprises for 2024-2025 yet. Decided 2026-11-17: the main
      sample ends 2023-12-31 (`sample.end`); CRSP/Compustat still pulled to 2025
      (`sample.data_end`). Extend when the SF Fed updates the file
- [x] FRED and Ken French parsers checked against the real files (2026-09 French vintage)
- [x] Chen's posted estimates located: three stock-level SAS files on his HKUST page
      (`data/chen.py`); schema printed by the data report
- [x] Free downloads run end to end on a local machine (2026-11-17)
- [x] CRSP daily 1994-2024 via legacy `crsp.dsf` (pulled 2026-11-17); delisting returns
      attached to the last trading day with Shumway imputation (PR #4)
- [x] **CRSP CIZ migration:** 2025 comes from `crsp.dsf_v2`, renamed to the legacy schema
      (`wrds_pull.normalize_ciz`). First run OK; 2024 overlap: ret corr 1.0, max diff 15bp
- [ ] Compustat annual + CCM link; book equity, profitability, leverage, FF49 industries
- [ ] Monthly stock panel: ME, microcap flag (NYSE 20th pct), lagged z-scored characteristics
- **Check:** a data report (counts by year, missingness, event calendar) in `results/tables/`
  (`make report` writes `data_report.md` and `fomc_calendar.csv`)

## M2. Replication of Chen (2022)
- [ ] Market, size deciles and book-to-market deciles: day-0 durations
- [ ] Chen's posted estimates are stock-level (`data/chen.py`): aggregate to size and BM
      deciles and match against ours
- [ ] Side-by-side table and scatter, ours vs Chen's
- [ ] Orthogonalized-surprise version (Bauer-Swanson) as robustness
- **Check:** signs and ranking match Chen; gaps explainable by daily vs intraday returns

## M3. Stock-level duration model
- [ ] Stock-event panel: day-0 returns on the surprise with lagged z-characteristics
- [ ] Expanding-window pooled fit at each month end (only earlier meetings); store coefficients
- [ ] Fitted durations `D_hat_{i,m}`; check stability of coefficients over time
- [ ] Quintile sorts, VW, ex-microcaps; industry-neutral variant
- **Check:** out-of-sample, the day-0 duration of the top-minus-bottom quintile is positive
  and significant (the sort actually separates rate exposure)

## M4. Horizon durations D(h) (headline)
- [ ] Cumulative HL returns for h = 0..20 after each scheduled meeting
- [ ] D(h) with and without 2y/10y post-event yield controls, with factor controls
- [ ] Pre-specified h = 5 and h = 20; full curve figure with confidence bands
- [ ] Same-day share D(0)/D(20) with event bootstrap CI
- **Check:** headline figure and table in `results/`, written into the paper

## M5. Strategy
- [ ] Post-announcement position (short HL after tightening, long after easing), hold h
- [ ] Sharpe, FF5+Mom alpha, turnover, net of flat costs, by decade
- **Check:** strategy table in `results/tables/`

## M6. Robustness and extensions
- [ ] Orthogonalized surprises throughout (Fed information effect)
- [ ] Equal-weighted; include microcaps; buy-and-hold weights within month
- [ ] Alternative characteristic sets; drop financials vs banks-only
- [ ] Subsamples: pre/post 2008, ZLB periods
- [ ] Optional: large non-FOMC moves in the 2y yield and the 10y-2y slope
- [ ] Placebo: random non-FOMC days with pseudo-surprises

## M7. Paper
- [ ] Move the writeup to LaTeX in `paper/`, tables generated from `results/`
