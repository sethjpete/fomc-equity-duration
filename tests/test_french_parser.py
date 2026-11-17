from fomcdur.data.french import parse_daily_csv

SAMPLE = """This file was created using the 202509 CRSP database.
The 1-factor ...

,Mkt-RF,SMB,HML,RMW,CMA,RF
19630701,  -0.67,   0.02,  -0.35,   0.03,   0.13,   0.012
19630702,   0.79,  -0.28,   0.28,  -0.08,  -0.21,   0.012

 Annual Factors
,Mkt-RF,SMB
1964, 1.0, 2.0
"""


def test_parse_daily_block_only():
    df = parse_daily_csv(SAMPLE)
    assert list(df.columns) == ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert len(df) == 2 and abs(df.iloc[0, 0] + 0.0067) < 1e-12
