"""Parsers checked against snippets of the real files (vintages noted in each module)."""

import numpy as np
import pandas as pd
import pytest

from fomcdur import config
from fomcdur.clean.events import build_calendar
from fomcdur.data import fred, french, sffed, wrds_pull


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW", tmp_path)
    return tmp_path


# --- SF Fed -----------------------------------------------------------------------

SFFED_HEADER = ["Date", "Time", "Unscheduled", "FF1", "FF2", "ED1", "ED2", "ED3", "ED4", "SP500",
                "SP500 emini", "TNOTE02", "TNOTE05", "TNOTE10", "TBOND", "MPS", "MPS_ORTH",
                "NFP_SURP", "NFP_12M", "SP500_3M", "SLOPE_3M", "BCOM_3M", "TR_SKEW"]
# (Excel serial, time, unscheduled, MPS, MPS_ORTH) copied from the 2023 update
SFFED_ROWS = [
    (32184, "11:30am", 0, 0.00930356, -0.0000851611),   # 1988-02-11, before sample
    (34369, "11:05am", 0, 0.160331, 0.151179),           # 1994-02-04
    (34442, "10:06am", 1, 0.0996491, 0.0728801),         # 1994-04-18 unscheduled
    (37151, "8:20am", 1, "NA", "NA"),                    # 2001-09-17, MPS missing
    (43905, "5:00pm", 1, -0.212882, "NA"),               # 2020-03-15 Sunday 5pm
    (43950, "2:00pm", 0, 0.00290021, "NA"),              # 2020-04-29, ORTH NA (COVID)
    (45273, "2:00pm", 0, -0.0818608, -0.0569218),        # 2023-12-13
]


def write_sffed(path, as_dates=False):
    """as_dates=True stores real date cells, as the SF Fed file does; False stores serials."""
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active.title = "Description"
    ws = wb.create_sheet(sffed.SHEET)
    ws.append(SFFED_HEADER)
    for d, t, u, m, mo in SFFED_ROWS:
        row = dict.fromkeys(SFFED_HEADER, "NA")
        if as_dates:
            d = (pd.Timestamp("1899-12-30") + pd.Timedelta(days=d)).to_pydatetime()
        row.update(Date=d, Time=t, Unscheduled=u, MPS=m, MPS_ORTH=mo)
        ws.append([row[c] for c in SFFED_HEADER])
    wb.save(path)


@pytest.mark.parametrize("as_dates", [True, False])
def test_sffed_load_real_layout(raw, as_dates):
    write_sffed(sffed.raw_dir("sffed") / sffed.FILENAME, as_dates=as_dates)
    assert list(sffed.load()["date"].dt.year) == [2020, 2023]  # main sample from 2003-09
    df = sffed.load(start="1994-01-01")  # scheduled, MPS present
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["1994-02-04", "2020-04-29", "2023-12-13"]
    assert df["mps"].iloc[-1] == pytest.approx(-0.0818608)  # pp
    assert np.isnan(df["mps_orth"].iloc[1])
    full = sffed.load(scheduled_only=False, start="1900-01-01", drop_missing=False)
    assert len(full) == len(SFFED_ROWS) and full["mps"].isna().sum() == 1
    assert sffed.coverage_end() == pd.Timestamp("2023-12-13")


def test_calendar_moves_after_close_and_weekend(raw):
    write_sffed(sffed.raw_dir("sffed") / sffed.FILENAME)
    s = sffed.load(scheduled_only=False, start="1994-01-01", drop_missing=False)
    cal = build_calendar(s, pd.bdate_range("1994-01-03", "2024-01-31")).set_index("date")
    assert cal.loc["2020-03-15", "day0"] == pd.Timestamp("2020-03-16")  # Sunday 5pm -> Monday
    assert cal.loc["1994-04-18", "day0"] == pd.Timestamp("1994-04-18")  # morning, same day
    assert cal.loc["1994-02-04", "hour"] == pytest.approx(11 + 5 / 60)
    assert cal["shifted"].sum() == 1


def test_calendar_after_close_on_weekday_goes_to_next_day():
    s = pd.DataFrame({"date": pd.to_datetime(["2008-01-21"]), "time": ["4:30pm"]})  # Monday
    cal = build_calendar(s, pd.bdate_range("2008-01-01", "2008-02-01"))
    assert cal["day0"].iloc[0] == pd.Timestamp("2008-01-22")


# --- FRED -------------------------------------------------------------------------

def test_fred_parse_blank_holidays():
    y = fred.parse("observation_date,DGS2\n1976-06-01,7.26\n1976-07-05,\n1976-07-06,7.30\n")
    assert y.isna().sum() == 1 and y.iloc[-1] == 7.30


# --- Ken French -------------------------------------------------------------------

MOM = """This file was created by using the 202609 CRSP database.  It
contains a momentum factor, constructed from six value-weight portfolios formed

Missing data are indicated by -99.99 or -999.

,Mom
19261103,   0.55
19261104, -99.99
19261105,   1.17

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""

ME_BP = """This file was created using the 202609 CRSP database.  It contains every 5th NYSE ME percentile (divided by 1000000).

202608,  1069,""" + ",".join(f"{v:10.2f}" for v in range(1, 21)) + """
202609,  1070,""" + ",".join(f"{v:10.2f}" for v in range(21, 41)) + """

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""

SIC49 = """ 1 Agric  Agriculture
          0100-0199 Agricultural production - crops
          0200-0299 Agricultural production - livestock

 2 Food   Food Products
          2000-2009 Food and kindred products
"""


def test_french_momentum_header_and_missing_codes():
    df = french.parse_daily_csv(MOM.replace("\n", "\r\n"))
    assert list(df.columns) == ["Mom"] and len(df) == 3
    assert np.isnan(df.iloc[1, 0]) and df.iloc[2, 0] == pytest.approx(0.0117)


def test_french_breakpoints_and_sic(raw):
    (french.raw_dir("french") / "me_breakpoints.txt").write_text(ME_BP)
    (french.raw_dir("french") / "siccodes49.txt").write_text(SIC49)
    bp = french.load_me_breakpoints()
    assert bp.index[-1] == pd.Timestamp("2026-09-30") and bp["p20"].iloc[-1] == 24
    sic = french.load_siccodes49()
    assert list(sic["ind49"]) == [1, 1, 2]
    ind = french.assign_ff49(pd.Series([150, 2005, 9999, np.nan]), sic)
    assert list(ind[:3]) == [1, 2, 49] and pd.isna(ind[3])


# --- CRSP CIZ -----------------------------------------------------------------------

def test_normalize_ciz_and_overlap():
    ciz = pd.DataFrame({"permno": [1, 1, 2], "dlycaldt": pd.to_datetime(["2024-01-02",
                                                                        "2024-01-03",
                                                                        "2024-01-02"]),
                        "dlyret": [0.01, -0.02, 0.0], "dlyretx": [0.01, -0.02, 0.0],
                        "dlyprc": [10.0, 9.8, 5.0], "shrout": [100, 100, 50], "dlyvol": [1, 2, 3],
                        "siccd": [2000, 2000, 6000], "primaryexch": ["N", "N", "Q"]})
    out = wrds_pull.normalize_ciz(ciz)
    assert list(out["exchcd"]) == [1, 1, 3] and set(out["shrcd"]) == {11}
    leg = out.drop(columns="source").copy()
    leg.loc[1, "ret"] = -0.0201
    stats = wrds_pull.compare_overlap(leg, out)
    assert stats["rows_both"] == 3 and stats["only_ciz"] == 0
    assert stats["max_abs_ret_diff"] == pytest.approx(1e-4)
