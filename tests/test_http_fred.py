import pandas as pd
import pytest
import requests

from fomcdur import config
from fomcdur.data import fred, http


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def get(self, url, **kw):
        self.calls += 1
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        r = requests.Response()
        r.status_code, r._content = o
        return r


def test_get_retries_timeouts_then_succeeds():
    s = FakeSession([requests.ReadTimeout(), (503, b""), (200, b"ok")])
    waits = []
    r = http.get("https://x", session=s, retries=3, backoff=1, sleep=waits.append)
    assert r.text == "ok" and s.calls == 3 and waits == [1, 2]


def test_get_gives_up_and_does_not_retry_404():
    with pytest.raises(requests.ReadTimeout):
        http.get("https://x", session=FakeSession([requests.ReadTimeout()] * 2), retries=1,
                 sleep=lambda _: None)
    s = FakeSession([(404, b"")])
    with pytest.raises(requests.HTTPError):
        http.get("https://x", session=s, sleep=lambda _: None)
    assert s.calls == 1


TREASURY = """Date,"1 Mo","2 Yr","10 Yr"
01/03/2023,4.17,4.40,3.79
01/02/2023,,,
"""


def test_treasury_to_fred_matches_fred_format():
    text = fred.treasury_to_fred([pd.read_csv(pd.io.common.StringIO(TREASURY))], "DGS2")
    assert text.splitlines()[0] == "observation_date,DGS2"
    y = fred.parse(text)
    assert y.index[0] == pd.Timestamp("2023-01-02") and y.isna().sum() == 1
    assert y.iloc[-1] == 4.40


def test_download_falls_back_to_treasury_and_skips_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW", tmp_path)
    (tmp_path / "fred").mkdir()
    (tmp_path / "fred" / "DGS10.csv").write_text("observation_date,DGS10\n2023-01-03,3.79\n")

    def fred_hangs(series):
        raise requests.ReadTimeout("read timed out")

    monkeypatch.setattr(fred, "_from_fred", fred_hangs)
    monkeypatch.setattr(fred, "_treasury_year",
                        lambda y: pd.read_csv(pd.io.common.StringIO(TREASURY)))
    fred.download()
    df = fred.load()
    assert df.loc["2023-01-03", "DGS2"] == 4.40 and df.loc["2023-01-03", "DGS10"] == 3.79
