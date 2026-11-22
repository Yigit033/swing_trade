"""Gölge koşucu uçtan uca (sahte veri kaynağı, SQLite defter, ağ yok)."""
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from swing_trader.shadow import rules, runner
from swing_trader.shadow.ledger import ShadowLedger

NY = ZoneInfo("America/New_York")


class FakeSrc:
    class SecIndexMissing(Exception):
        pass

    def __init__(self):
        days = rules.sessions_between(date(2025, 1, 2), date(2025, 9, 30))
        n = len(days)
        close = 20 + np.linspace(0, 4, n)             # yavaş yükselen, stop'a değmeyen seri
        self.bars = {}
        for t, mult in (("ABC", 1.0), ("XYZ", 1.5), ("IWM", 10.0)):
            c = close * mult
            self.bars[t] = pd.DataFrame(dict(Date=pd.to_datetime(days), Open=c * 0.999, High=c * 1.01,
                                             Low=c * 0.99, Close=c, Volume=1e6))
        self.filings = {}
        self.vce = {}                                    # asof → [(ticker, dv20)]
        self.champion = []

    def live_vce_hits(self, asof):
        return dict(hits=[dict(ticker=t, dv20=v, close=1.0) for t, v in self.vce.get(asof, [])],
                    stats=dict(universe=2, fetched=2, current_bar=2, vce=len(self.vce.get(asof, []))))

    def champion_trades(self, since):
        return self.champion

    def dilution_flags(self, rows):
        return {str(r["id"]): ([dict(form="424B5", date="2025-06-01")] if r["ticker"] == "DIL" else [])
                for r in rows}

    def candidate_universe(self):
        return {"ABC", "XYZ"}

    def earnings_reporters(self, start, end):
        return pd.DataFrame({"Marketcap": [1e9, 5e12]}, index=["ABC", "MEGA"])

    def earnings_rows(self, t):
        assert t == "ABC", "evren/cap ön filtresi MEGA'yı elemeliydi"
        return [dict(ts="2025-06-12T16:05:00-04:00", est=0.50, rep=0.575, surprise=15.0),
                dict(ts="2025-03-12T16:05:00-04:00", est=0.50, rep=0.60, surprise=20.0)]

    def fetch_bars(self, tickers, start, end):
        return {t: b[(b.Date >= pd.Timestamp(start)) & (b.Date <= pd.Timestamp(end))].reset_index(drop=True)
                for t, b in self.bars.items() if t in set(tickers)}

    def shares_outstanding(self, t):
        return 50e6

    def fetch_form4_filings(self, day):
        if day.weekday() >= 5:
            raise self.SecIndexMissing(str(day))
        return self.filings.get(day, [])


def _buy(acc, cik, sym, day, value):
    return dict(form="4", accession=acc, filing_date=day, symbol=sym,
                owners=[dict(cik=cik, relationship="Director")],
                transactions=[dict(code="P", acquired="A", shares=str(value / 10), price="10")])


def test_h5_signal_lifecycle_pending_open_closed(tmp_path):
    L, src = ShadowLedger(str(tmp_path / "s.db")), FakeSrc()
    out = runner.run_shadow_daily(now=datetime(2025, 6, 16, 7, 0, tzinfo=NY), ledger=L, src=src)
    assert out["H5"]["events"] == 1 and out["H5"]["added"] == 1
    t = L.trades("H5")[0]
    assert (t["ticker"], t["signal_date"], t["status"], t["late_decision"]) == ("ABC", "2025-06-13", "PENDING", 0)

    runner.run_shadow_daily(now=datetime(2025, 6, 17, 7, 0, tzinfo=NY), ledger=L, src=src)
    t = L.trades("H5")[0]
    assert t["status"] == "OPEN" and t["entry_date"] == "2025-06-16"
    assert t["stop"] == pytest.approx(t["entry_price"] - 2.5 * t["atr"], rel=1e-6)

    runner.run_shadow_daily(now=datetime(2025, 7, 25, 7, 0, tzinfo=NY), ledger=L, src=src)
    t = L.trades("H5")[0]
    assert t["status"] == "CLOSED" and t["exit_reason"] == "TIME" and t["bars"] == 20
    assert t["net_pct"] == pytest.approx(t["ret_pct"] - 0.5, abs=1e-6)
    assert t["iwm_ret"] is not None
    rep = L.get_meta("report")
    assert rep["strategies"]["H5"]["closed"] == 1
    assert rep["strategies"]["H5"]["gate"]["status"] == "TOPLANIYOR"


def test_h5_late_decision_flag(tmp_path):
    L, src = ShadowLedger(str(tmp_path / "s.db")), FakeSrc()
    runner.run_shadow_daily(now=datetime(2025, 6, 16, 11, 0, tzinfo=NY), ledger=L, src=src)  # açılıştan sonra
    assert L.trades("H5")[0]["late_decision"] == 1


def test_h4_cluster_recorded_with_warmup_and_universe(tmp_path):
    L, src = ShadowLedger(str(tmp_path / "s.db")), FakeSrc()
    runner.run_shadow_daily(now=datetime(2025, 6, 16, 7, 0, tzinfo=NY), ledger=L, src=src)
    assert L.get_meta("h4_observation_start") == "2025-06-16"
    src.filings = {
        date(2025, 6, 17): [_buy("a1", "1", "XYZ", date(2025, 6, 17), 60_000),
                            _buy("b1", "7", "NOTLISTED", date(2025, 6, 17), 80_000)],
        date(2025, 6, 18): [_buy("a2", "2", "XYZ", date(2025, 6, 18), 50_000),
                            _buy("b2", "8", "NOTLISTED", date(2025, 6, 18), 80_000)],
    }
    seen = []
    for hour in (7, 8, 9):      # koşu başına 1 gün (SEC hız sınırı); birikim sonraki koşularda erir
        out = runner.run_shadow_daily(now=datetime(2025, 6, 19, hour, 0, tzinfo=NY), ledger=L, src=src)
        seen += [p["day"] for p in out["H4"]["processed"]]
    assert seen == ["2025-06-16", "2025-06-17", "2025-06-18"]
    rows = {t["ticker"]: t for t in L.trades("H4")}
    assert rows["XYZ"]["event_date"] == "2025-06-18" and rows["XYZ"]["status"] in ("PENDING", "OPEN")
    assert rows["XYZ"]["warmup"] == 1                       # pencere gözlem başlangıcından önceye taşıyor
    assert rows["NOTLISTED"]["status"] == "SKIPPED" and rows["NOTLISTED"]["reason"] == "NOT_IN_UNIVERSE"


def test_params_change_blocks_new_signals(tmp_path):
    L, src = ShadowLedger(str(tmp_path / "s.db")), FakeSrc()
    L.set_meta("live_start", "2025-06-13")
    L.set_meta("params_sha256", "eski-kurallar")
    out = runner.run_shadow_daily(now=datetime(2025, 6, 16, 7, 0, tzinfo=NY), ledger=L, src=src)
    assert out["error"] == "PARAMS_CHANGED_WITHOUT_PREREGISTRATION"
    assert L.trades() == []


def test_shadow_api_reads_report(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routers import shadow as shadow_router

    db = str(tmp_path / "s.db")
    L, src = ShadowLedger(db), FakeSrc()
    runner.run_shadow_daily(now=datetime(2025, 6, 16, 7, 0, tzinfo=NY), ledger=L, src=src)
    monkeypatch.setattr("swing_trader.shadow.ledger.ShadowLedger.__init__",
                        lambda self, db_path=None: (setattr(self, "_path", db), setattr(self, "_pg", False), None)[-1])
    app = FastAPI()
    app.include_router(shadow_router.router)
    c = TestClient(app)
    r = c.get("/api/shadow")
    assert r.status_code == 200 and r.json()["trades"][0]["ticker"] == "ABC"
    assert r.json()["report"]["strategies"]["H5"]["gate"]["status"] == "TOPLANIYOR"
    monkeypatch.delenv("CRON_SECRET", raising=False)
    assert c.post("/api/shadow/daily-run").status_code == 503
    monkeypatch.setenv("CRON_SECRET", "s3")
    assert c.post("/api/shadow/daily-run", headers={"X-Cron-Secret": "x"}).status_code == 401


def test_lagging_latest_bar_keeps_trade_pending(tmp_path):
    """Son seansın barı yfinance'ta henüz yoksa giriş kalıcı ATLANMAZ, PENDING kalır."""
    L, src = ShadowLedger(str(tmp_path / "s.db")), FakeSrc()
    runner.run_shadow_daily(now=datetime(2025, 6, 16, 7, 0, tzinfo=NY), ledger=L, src=src)
    b = src.bars["ABC"]
    src.bars["ABC"] = b[b.Date != pd.Timestamp("2025-06-16")]        # giriş günü barı gecikti
    runner.run_shadow_daily(now=datetime(2025, 6, 17, 7, 0, tzinfo=NY), ledger=L, src=src)
    assert L.trades("H5")[0]["status"] == "PENDING"
    src.bars["ABC"] = b                                                # bar geldi
    runner.run_shadow_daily(now=datetime(2025, 6, 18, 7, 0, tzinfo=NY), ledger=L, src=src)
    assert L.trades("H5")[0]["status"] == "OPEN"


def test_h6_lifecycle_fixed_hold_and_champion(tmp_path):
    """Ek F: canlı VCE isabeti → PENDING → t+1 açılışta OPEN → 8. seans kapanışında CLOSED (stop yok)."""
    L, src = ShadowLedger(str(tmp_path / "s.db")), FakeSrc()
    src.vce = {date(2025, 6, 13): [("XYZ", 9e7), ("ABC", 5e7)]}
    src.champion = [dict(id=1, ticker="CLN", entry_date="2025-06-16", realized_pnl_pct=2.0),
                    dict(id=2, ticker="DIL", entry_date="2025-06-16", realized_pnl_pct=-1.0)]
    out = runner.run_shadow_daily(now=datetime(2025, 6, 16, 7, 0, tzinfo=NY), ledger=L, src=src)
    assert out["H6"]["added"] == 2 and L.get_meta("h6_live_start") == "2025-06-13"
    assert L.get_meta("params_h6_sha256") == rules.PARAMS_H6_SHA256
    t = {r["ticker"]: r for r in L.trades("H6")}
    assert t["XYZ"]["status"] == "PENDING" and t["XYZ"]["late_decision"] == 0 and t["XYZ"]["key"] == 9e7
    runner.run_shadow_daily(now=datetime(2025, 6, 17, 7, 0, tzinfo=NY), ledger=L, src=src)
    t = L.trades("H6")[0]
    assert t["status"] == "OPEN" and t["entry_date"] == "2025-06-16" and t["stop"] is None
    runner.run_shadow_daily(now=datetime(2025, 6, 30, 7, 0, tzinfo=NY), ledger=L, src=src)
    t = {r["ticker"]: r for r in L.trades("H6")}["XYZ"]
    assert t["status"] == "CLOSED" and t["exit_reason"] == "TIME" and t["bars"] == 8
    assert t["exit_date"] == "2025-06-26"     # 16,17,18,20,23,24,25,26 Haz = 8 seans (19 Haz tatil)
    rep = L.get_meta("report")
    assert rep["strategies"]["H6"]["closed"] == 2 and rep["strategies"]["H6"]["gate"]["need_closed"] == 100
    assert rep["champion"]["closed"] == 2 and rep["champion"]["net_mean"] == pytest.approx(0.0)
    split = rep["champion"]["dilution_split"]                    # Ek H ileri ölçümü
    assert split["dilution"] == dict(closed=1, net_mean=-1.5) and split["clean"] == dict(closed=1, net_mean=1.5)
    assert L.get_meta("champion_dilution")["2"][0]["form"] == "424B5"
    assert rep["params_sha256"] == rules.PARAMS_SHA256          # H4/H5 kilidi etkilenmedi


def test_h6_params_change_blocks_only_h6(tmp_path):
    L, src = ShadowLedger(str(tmp_path / "s.db")), FakeSrc()
    L.set_meta("params_h6_sha256", "eski-h6")
    src.vce = {date(2025, 6, 13): [("XYZ", 9e7)]}
    out = runner.run_shadow_daily(now=datetime(2025, 6, 16, 7, 0, tzinfo=NY), ledger=L, src=src)
    assert out["H6"]["error"] == "H6_PARAMS_CHANGED_WITHOUT_PREREGISTRATION"
    assert L.trades("H6") == [] and out["H5"]["added"] == 1     # H5 normal çalışmaya devam


def test_sec_dilution_window_cache_and_holidays(monkeypatch):
    """Son 30 gün; geçmiş gün önbelleğe alınır, dünün dosyası alınmaz; 404 (tatil) boş sayılır."""
    import json
    from urllib.error import HTTPError
    from swing_trader.data import sec_filings as sf
    monkeypatch.setattr(sf, "_days", {})
    monkeypatch.setattr(sf, "_tickers", {"at": 0.0, "map": {}})
    calls = []
    idx = "\n".join([
        "Form Type   Company Name   CIK   Date Filed   File Name",
        "424B5       ACME CORP      123   20251103     edgar/data/123/0001-25-000001.txt",
        "424B5/A     ACME CORP      123   20251103     edgar/data/123/0001-25-000002.txt",
        "10-Q        OTHER INC      456   20251103     edgar/data/456/0001-25-000003.txt",
    ])

    def fake_get(url):
        calls.append(url)
        if url.endswith("company_tickers.json"):
            return json.dumps({"0": {"cik_str": 123, "ticker": "ACME"}, "1": {"cik_str": 456, "ticker": "OTH"}}).encode()
        if "form.20251103.idx" in url:
            return idx.encode()
        raise HTTPError(url, 404, "nf", None, None)
    monkeypatch.setattr(sf, "_get", fake_get)
    got = sf.recent_dilution(["ACME", "OTH", "NOPE"], date(2025, 11, 20), today=date(2025, 11, 21))
    assert got == {"ACME": [dict(form="424B5", date="2025-11-03")]}      # /A ve 10-Q sayılmaz
    assert date(2025, 11, 3) in sf._days and date(2025, 11, 20) not in sf._days
    n = len(calls)
    sf.recent_dilution(["ACME"], date(2025, 11, 20), today=date(2025, 11, 21))
    assert len(calls) - n == 1                                           # yalnız önbelleksiz son gün
    assert sf.recent_dilution(["ACME"], date(2025, 12, 20), today=date(2025, 12, 22)) == {}   # pencere dışı


def test_sec_missing_file_403_is_treated_as_missing(monkeypatch):
    """SEC tatilinde günlük indeks yok → S3 AccessDenied 403; 404 sayılmalı, gerçek 403 sayılmamalı."""
    import io
    from urllib.error import HTTPError
    from swing_trader.shadow import sources as S
    monkeypatch.setattr(S.time, "sleep", lambda s: None)

    def fake(body):
        def _open(req, timeout=30):
            raise HTTPError(req.full_url, 403, "Forbidden", {}, io.BytesIO(body))
        return _open
    import gzip
    denied = b'<?xml version="1.0"?><Error><Code>AccessDenied</Code></Error>'
    for body in (denied, gzip.compress(denied)):                 # SEC gerçekte gzip'li gönderiyor
        monkeypatch.setattr(S, "urlopen", fake(body))
        with pytest.raises(S.SecIndexMissing):
            S.form4_index(date(2026, 11, 11))
    monkeypatch.setattr(S, "urlopen", fake(b"<html>Request Rate Threshold Exceeded</html>"))
    with pytest.raises(HTTPError) as e:
        S._sec_get("https://www.sec.gov/x", retries=2)
    assert e.value.code == 403                                   # gerçek engel sessizce yutulmaz
