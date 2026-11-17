# Roadmap

Milestones are in dependency order. Each ends with a concrete check so we know it is done.
Tick boxes as work lands; open a GitHub issue per bullet when someone picks it up.

## M0. Infrastructure (done in the initial commit)
- [x] Package layout, settings, `.gitignore` for data and secrets, CI running tests
- [x] Estimators with synthetic-data tests: event duration, D(h) curve, same-day share
      bootstrap, pooled model, quintile/VW portfolios, event strategy
- [ ] Everyone runs `pip install -e ".[wrds,dev]"` and `pytest` locally
- [ ] Decide where shared data lives (shared drive path for `FOMCDUR_DATA_DIR`)

## M1. Data pipeline
- [ ] SF Fed surprises: confirm the download URL and column names (`data/sffed.py`
      COLUMN_MAP), confirm coverage back to 1994, count scheduled meetings (target ~250)
- [ ] FRED and Ken French downloads run end to end; parsers checked on real files
- [ ] CRSP daily 1994-2024 via legacy `crsp.dsf`; delisting-adjusted returns
- [ ] **CRSP CIZ migration:** legacy SIZ tables stop at 2024. Pull 2025 from the CIZ
      (`_v2`) tables and check the two formats agree on an overlap year
- [ ] Compustat annual + CCM link; book equity, profitability, leverage, FF49 industries
- [ ] Monthly stock panel: ME, microcap flag (NYSE 20th pct), lagged z-scored characteristics
- **Check:** a data report (counts by year, missingness, event calendar) in `results/tables/`

## M2. Replication of Chen (2022)
- [ ] Market, size deciles and book-to-market deciles: day-0 durations
- [ ] Locate and parse Chen's posted estimates (`data/chen.py`); match portfolios
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
