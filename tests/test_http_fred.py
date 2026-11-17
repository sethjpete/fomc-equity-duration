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


def test_api_json_to_fred():
    payload = {"observations": [
        {"realtime_start": "x", "realtime_end": "x", "date": "2023-01-02", "value": "."},
        {"realtime_start": "x", "realtime_end": "x", "date": "2023-01-03", "value": "4.40"}]}
    y = fred.parse(fred.api_json_to_fred(payload, "DGS2"))
    assert y.isna().sum() == 1 and y.loc["2023-01-03"] == 4.40
    with pytest.raises(ValueError, match="Bad Request"):
        fred.api_json_to_fred({"error_code": 400, "error_message": "Bad Request. bad key"}, "DGS2")


@pytest.fixture
def fred_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW", tmp_path)
    monkeypatch.setattr(fred, "_treasury_year",
                        lambda y: pd.read_csv(pd.io.common.StringIO(TREASURY)))
    return tmp_path


def test_download_uses_api_when_key_set(fred_env, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")
    seen = []
    monkeypatch.setattr(fred, "_from_api", lambda s, key: (seen.append((s, key)), fred.api_json_to_fred(
        {"observations": [{"date": "2023-01-03", "value": "1.0"}]}, s))[1])
    fred.download()
    assert seen == [("DGS2", "k"), ("DGS10", "k")]
    assert fred.load().loc["2023-01-03", "DGS10"] == 1.0


def test_download_falls_back_to_treasury_and_skips_existing(fred_env, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")
    (fred_env / "fred").mkdir()
    (fred_env / "fred" / "DGS10.csv").write_text("observation_date,DGS10\n2023-01-03,3.79\n")

    def api_fails(series, key):
        raise requests.ConnectionError("dropped")

    monkeypatch.setattr(fred, "_from_api", api_fails)
    fred.download()
    df = fred.load()
    assert df.loc["2023-01-03", "DGS2"] == 4.40 and df.loc["2023-01-03", "DGS10"] == 3.79


def test_download_without_key_uses_treasury(fred_env, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr(fred, "_from_api", lambda *a: pytest.fail("API called without a key"))
    fred.download()
    assert fred.load().loc["2023-01-03", "DGS10"] == 3.79
