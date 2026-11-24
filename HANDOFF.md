# Handoff (started 2026-11-17, updated 2026-11-24)

Written at the end of the first working session, for Kenzo and Grigorios and for any
future Claude session. The plan is in [ROADMAP.md](ROADMAP.md) and setup reference is
in the [README](README.md). This note covers what changed, what's next, and how to
catch up.

## 0a. Update 2026-11-24: M3 started, sample is 2003+ intraday only

- **Decision (Seth):** the main sample is the 162 scheduled meetings from 2003-09 (first
  Millisecond TAQ day) to 2023-12, and the day-0 return is the intraday [-30, +10] min TAQ
  midpoint change. The daily day-0 measure and 1994-2003 are dropped. Settings:
  `sample.start = 2003-09-10`; `sample.m2_start = 1994-01-01` keeps the M2 scripts as they
  were; `sample.data_start = 2002` for new CRSP/Compustat pulls (rerunning M2 from scratch
  needs `pull_crsp(start_year=1994)`).
- **M3 step 1 done** (`make portfolios`, `scripts/08_portfolio_durations.py`,
  `fomcdur.extension.portfolio_durations`): day-0 durations of the FF49 industries and of
  size, B/M, profitability and leverage deciles, each with the robustness set. Industry
  durations are reliable (split-half 0.79, equal-duration test rejected); characteristic
  deciles are flat except size. Numbers in ROADMAP M3.
- **Next:** M3 step 2, the industry-portfolio panel (duration on lagged VW characteristics),
  then the out-of-sample fitted durations and quintile check.
- The script runs in ~25 s from the raw files (TAQ midpoints, CRSP 2003-2023, Compustat).

## 0. Update 2026-11-20: M2 done, what's next

- **M2 (Chen replication) is finished** (PRs #5 and #6). Results and the reasoning behind
  each step are in ROADMAP M2. Short version:
  - B/M ranking agrees with Chen. The size shape agrees only under his own estimator and
    [1, 300] trim.
  - The Chen-exact mode (`scripts/07_chen_exact.py`) uses a dER_M calibrated to his output
    (0.514 x the 40-min market return): we could not reproduce his VAR. It recovers 77% of
    his stock-level estimates, but the stock-level correlation is only 0.36.
- **Design decision (Seth):** weak stock-level agreement = low signal-to-noise. From M3 on we
  estimate durations on **portfolios** (French 49 industries, plus characteristic deciles),
  by regression on the surprise. See "Design decisions after M2" in ROADMAP.
- **Next:** M3 step 1, day-0 durations for the 49 industry portfolios (done 2026-11-24,
  see 0a).
- **Running heavy jobs:** WRDS and TAQ work runs on Seth's Mac (Remote Control session from
  a Claude thread, or by hand). TAQ pulls are slow (1-11 min per meeting) and resumable.
  Run them detached (`caffeinate`, `nohup`) with retries.

## 1. What we did on 2026-11-17

**Infrastructure (M0).** We set up the repo: the `fomcdur` Python package (src layout),
`config/settings.toml`, one download module per data source, estimators with tests on
synthetic data, and a roadmap with milestones M1 to M7.

**Data pipeline (M1), merged through PRs #1 to #3, with #4 open:**

| PR | What |
|---|---|
| #1 | Loaders checked against the real source files. Added the SF Fed, French, FRED and Chen loaders, CRSP pulls in both the legacy format (to 2024) and the newer CIZ format (2025), the event calendar, and the data report (`make report`) |
| #2 | Shared HTTP downloads with retries; one failing source no longer stops the others |
| #3 | FRED switched to the **official API** (`FRED_API_KEY`), with Treasury.gov as the fallback |
| #4 | `wrds` installed by default; SF Fed date-parsing crash fixed; delisting returns attached to the last trading day and imputed; sharper Compustat section in the report; this note |

**Decisions made (and where they live):**
- **The main sample ends 2023-12-31 with 239 scheduled FOMC meetings.** The SF Fed
  surprise file stops at 2023-12-13 and has nothing for 2024-2025. CRSP and Compustat
  are still pulled through 2025 (`sample.data_end`), so we can extend as soon as the
  SF Fed updates. Set in `config/settings.toml`.
- **Use official APIs with keys wherever one exists** (Seth's rule). FRED goes through
  api.stlouisfed.org because `fred.stlouisfed.org` is behind Akamai bot protection,
  which drops Python downloads. Ken French, the SF Fed and Chen only publish static
  files, so those stay plain downloads.
- **Delisting returns** are compounded into the stock's last trading-day return. Missing
  returns for performance delistings (codes 500 and 520-584) are imputed at -30% for
  NYSE/AMEX and -55% for Nasdaq (Shumway 1997; Shumway and Warther 1999).
- **Units:** SF Fed `MPS` is in percentage points, and `surprise_scale = 0.01` turns
  duration into years.

**What the first real data run showed (Seth's Mac):**
- French factors, NYSE breakpoints, FF49 industries, FRED yields and Chen's files all load.
- CRSP 1994-2025 pulled. The 2024 overlap check between the legacy and CIZ formats gives
  a return correlation of 1.0 and a largest difference of 15bp, so no action is needed.
- Chen's posted estimates are **stock-level** files with columns `permno`, `eventdate`
  and `duration`, plus `dssduration`, `size`, `ret` and `returndate` depending on the
  file. Durations lie between 1 and 300, so they appear to be bounded. Check how in his
  paper before comparing.

## 2. Where we go from here

1. **Merge PR #4**, then on a machine with WRDS: `git pull && make install && make report`.
2. **Read `results/tables/data_report.md`.** Confirm that Compustat missingness for
   CRSP-linked firms is reasonable. Across all of Compustat it rises to about 50% by
   2025, which we think comes from firms never in CRSP and from more negative book
   equity. The new tables split these out. Also check the delisting breakdown by code.
   Then tick the remaining M1 boxes in ROADMAP.md.
3. **Decide where shared data lives** (ROADMAP M0). Put CRSP and Compustat extracts on a
   shared drive and point `FOMCDUR_DATA_DIR` at it, so only one person needs to pull from WRDS.
4. **M2, the Chen replication:** day-0 durations for the market and for size and
   book-to-market deciles. Group Chen's stock-level estimates into the same deciles and
   build the side-by-side table and scatter. Read how Chen bounds estimates to [1, 300] first.
5. **No CI.** The GitHub Actions workflow was removed in PR #4. Jobs never ran because
   of a GitHub account billing lock, and we chose not to pay for it. Tests run locally:
   **run `make test` before every PR** and say in the PR that it passed.

## 3. How Kenzo and Grigorios catch up

1. **Get access:** ask Seth to add you as a collaborator on
   `github.com/sethjpete/fomc-equity-duration`. You'll also need your own WRDS account
   (through Princeton) if you will pull CRSP and Compustat yourself.
2. **Clone and install** (Python 3.11+):
   ```bash
   git clone https://github.com/sethjpete/fomc-equity-duration.git
   cd fomc-equity-duration
   python -m venv .venv && source .venv/bin/activate
   make install            # pip install -e ".[dev]", which includes wrds
   make test               # should pass (28 tests as of PR #4)
   ```
3. **Get a free FRED API key** at https://fred.stlouisfed.org/docs/api/api_key.html.
   It needs a FRED account and takes a minute.
4. **Create `.env`** (git-ignored) from the template:
   ```bash
   cp .env.example .env
   # edit: WRDS_USERNAME=<your wrds username>
   #       FRED_API_KEY=<your key>
   ```
5. **Pull the free data** (FRED, Ken French, SF Fed, Chen): `make data-free`
6. **Pull WRDS data**, if you're the one doing it rather than using the shared drive:
   `make data-wrds`. The first connection asks for your WRDS password and offers to save
   it to `~/.pgpass` in your home directory. Say yes. It may also trigger Duo. A full
   pull (CRSP daily 1994-2025 by year, plus Compustat and the CCM link) takes a while.
7. **Build the report:** `make report`, then open `results/tables/data_report.md`.
8. **Ground rules:**
   - **Never commit data or credentials.** `data/`, `.env`, `.pgpass`, `*.parquet` and
     `*.sas7bdat` are git-ignored. CRSP and Compustat are licensed.
   - Work on a branch, run `make test` (there is no CI), open a PR, and have someone
     review it. Don't push straight to `main`.
   - Small tables and figures in `results/` can be committed. Raw extracts cannot.

## 4. Notes for future Claude sessions

- **Network:** the cloud sandbox blocks `fred.stlouisfed.org`, `mba.tuck.dartmouth.edu`
  and `www.frbsf.org` (`api.stlouisfed.org` is untested), so real downloads run on a
  team member's machine. A personal cloud environment named `fomc-equity-duration` that
  allows the first three hosts was created for future sessions.
- **WRDS** runs only on a team member's machine, with their credentials in `~/.pgpass`.
  Prepare the code and have them run `make data-wrds report`. The CIZ (`crsp.dsf_v2`)
  column names are now confirmed by the first real run.
- **Data layout:** `data/raw/<source>/` holds raw downloads (`crsp/dsf_<year>.parquet`,
  `crsp/dsedelist.parquet`, `crsp/overlap/`, `compustat/funda.parquet`,
  `compustat/ccm_link.parquet`, `sffed/`, `fred/`, `french/`, `chen/`). Also
  `data/interim`, `data/processed`, and `results/tables` / `results/figures`. Paths come
  from `fomcdur/config.py`, and `FOMCDUR_DATA_DIR` overrides the data root.
- **Where things are written down:** milestones and ticked items are in ROADMAP.md,
  settings and the reason behind each are in `config/settings.toml`, facts about each
  source are in the module docstrings in `src/fomcdur/data/`, and this file covers the
  2026-11-17 session.
- **SF Fed file:** use the sheet `FOMC (update 2023)`. Dates are date-typed cells.
  `MPS` is missing for 2001-09-17, and `MPS_ORTH` is NA from March to December 2020 by
  design. The one after-close announcement (2020-03-15, 5pm Sunday) maps to day 0 = 2020-03-16.
- **Chen (2022)** is Zhanhui Chen at HKUST, not Andrew Chen. His files are on
  https://chenzhanhui.people.ust.hk/publications.html.
- **TAQ:** Millisecond TAQ (`taqm_YYYY.complete_nbbo_YYYYMMDD`) from 2003-09-10 only;
  Monthly TAQ is not licensed. Link table `wrdsapps_link_crsp_taqm.taqmclink`. The `wrds`
  package returns nullable Float64 columns: cast to float64 before numpy work.
- **Chen-exact inputs** (`make` targets / `scripts/01_pull_data.py dsi fred_chen taq_chen`):
  `config/chen_table_a1.csv` is his Internet Appendix Table A.1, transcribed by hand.
- **There is no CI** (removed deliberately; Seth's GitHub account has a billing lock and
  he chose not to pay). Run `make test` yourself before pushing and report the result in
  the PR. Don't re-add a workflow unless asked.
