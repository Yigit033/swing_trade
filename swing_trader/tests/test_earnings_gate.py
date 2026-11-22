"""Bilanço kapısı: veri yokken GEÇMEZ (fail-closed) + ikinci kaynak (bilanço takvimi).

Kök neden (2026-11-22): filters.check_earnings yfinance boş/hatalı döndüğünde "geç" diyordu;
canlı SAIC 2026-08-31 bilanço gününde açıldı. Lab v3 Ek D1: kapı işlem başına +0.22 puan,
portföyde +7 puan değerli (kapı açık vs kapalı), yani "bilinmiyor" geçmemeli.
"""
from datetime import date, datetime

import pandas as pd
import pytest

from swing_trader.data import earnings_calendar
from swing_trader.small_cap.filters import SmallCapFilters

SIG = datetime(2026, 8, 31)


@pytest.fixture
def filt():
    return SmallCapFilters()


@pytest.fixture
def sources(monkeypatch):
    """own: hissenin kendi tarihleri (None = bilinmiyor); cal: takvim (None = alınamadı)."""
    state = {"own": None, "cal": None, "own_calls": 0, "cal_calls": []}

    def own(ticker):
        state["own_calls"] += 1
        return state["own"]

    def cal(start, end):
        state["cal_calls"].append((start, end))
        return state["cal"]

    monkeypatch.setattr(SmallCapFilters, "_ticker_earnings", staticmethod(own))
    monkeypatch.setattr(earnings_calendar, "reporters", cal)
    return state


def test_saic_regression_calendar_catches_missing_ticker_dates(filt, sources):
    sources["cal"] = {"SAIC": {date(2026, 8, 31)}}
    ok, reason = filt.check_earnings("SAIC", SIG)
    assert not ok and "calendar" in reason
    assert sources["cal_calls"] == [(date(2026, 8, 31), date(2026, 9, 3))]   # 0..N=3 gün


def test_both_sources_unknown_rejects(filt, sources):
    ok, reason = filt.check_earnings("XYZ", SIG)
    assert not ok and "unverified" in reason


def test_calendar_known_without_ticker_passes(filt, sources):
    sources["cal"] = {"OTHER": {date(2026, 9, 1)}}
    ok, reason = filt.check_earnings("XYZ", SIG)
    assert ok and "calendar" in reason


def test_ticker_dates_known_calendar_down_passes(filt, sources):
    sources["own"] = [pd.Timestamp("2026-08-05 16:05", tz="America/New_York"),
                      pd.Timestamp("2026-11-04 16:05", tz="America/New_York")]
    ok, reason = filt.check_earnings("XYZ", SIG)
    assert ok and reason == "No nearby earnings"


def test_ticker_dates_in_window_reject_without_calendar_call(filt, sources):
    sources["own"] = [pd.Timestamp("2026-09-02 07:00", tz="America/New_York")]
    ok, reason = filt.check_earnings("XYZ", SIG)
    assert not ok and reason.startswith("Earnings in 2 days")
    assert sources["cal_calls"] == []


def test_either_source_rejects(filt, sources):
    """Hissenin tarihleri yakında bilanço göstermese de takvim gösteriyorsa ret (birleşim)."""
    sources["own"] = [pd.Timestamp("2026-05-05")]
    sources["cal"] = {"XYZ": {date(2026, 9, 3)}}
    ok, reason = filt.check_earnings("XYZ", SIG)
    assert not ok and reason.startswith("Earnings in 3 days")


def test_symbol_normalized_for_calendar(filt, sources):
    sources["cal"] = {"BRK-B": {date(2026, 9, 1)}}
    assert filt.check_earnings("BRK.B", SIG)[0] is False


def test_unexpected_error_rejects(filt, monkeypatch):
    def boom(ticker):
        raise RuntimeError("x")
    monkeypatch.setattr(SmallCapFilters, "_ticker_earnings", staticmethod(boom))
    ok, reason = filt.check_earnings("XYZ", SIG)
    assert not ok and "fail-closed" in reason


@pytest.mark.parametrize("earn,expected", [("2026-08-31", False), ("2026-09-03", False),
                                           ("2026-09-04", True), ("2026-08-30", True)])
def test_explicit_dates_window_unchanged(filt, earn, expected):
    """Açık liste (backtest/discovery) olduğu gibi kullanılır: 0..3 gün ret, sonrası/öncesi serbest."""
    assert filt.check_earnings("XYZ", SIG, [pd.Timestamp(earn)])[0] is expected


def test_explicit_empty_list_means_verified_none(filt, monkeypatch):
    """[] = çağıran doğruladı, bilanço yok (discovery VERIFIED). Ağ çağrısı yapılmaz."""
    monkeypatch.setattr(SmallCapFilters, "_ticker_earnings",
                        staticmethod(lambda t: pytest.fail("açık listede ağ çağrısı olmamalı")))
    assert filt.check_earnings("XYZ", "2026-08-31", []) == (True, "No nearby earnings")


def test_apply_all_filters_live_path_rejects_unknown(filt, sources):
    dates = pd.bdate_range(end="2026-08-31", periods=40)
    df = pd.DataFrame(dict(Date=dates, Open=20., High=20.5, Low=19.5, Close=20., Volume=1_000_000.))
    passed, res = filt.apply_all_filters("XYZ", df, {"marketCap": 1e9}, SIG)
    assert not passed and "unverified" in res["filters"]["earnings"]["reason"]


# ── kaynak 1: hissenin kendi tarihleri ───────────────────────────────────────

class _FakeTicker:
    responses = []

    def __init__(self, t):
        pass

    @property
    def earnings_dates(self):
        r = _FakeTicker.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_ticker_earnings_empty_is_unknown_after_one_retry(monkeypatch):
    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", _FakeTicker)
    _FakeTicker.responses = [pd.DataFrame(), pd.DataFrame()]
    assert SmallCapFilters._ticker_earnings("SAIC") is None
    assert _FakeTicker.responses == []                     # iki deneme yapıldı


def test_ticker_earnings_retry_recovers(monkeypatch):
    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", _FakeTicker)
    idx = pd.DatetimeIndex([pd.Timestamp("2026-08-31 07:00", tz="America/New_York")])
    _FakeTicker.responses = [RuntimeError("Too Many Requests"), pd.DataFrame({"x": [1]}, index=idx)]
    assert SmallCapFilters._ticker_earnings("SAIC") == list(idx)


# ── kaynak 2: bilanço takvimi ────────────────────────────────────────────────

class _FakeCalendars:
    calls = []
    pages = []

    def __init__(self, start, end):
        _FakeCalendars.calls.append((start, end))

    def get_earnings_calendar(self, filter_most_active, limit, offset, force):
        return _FakeCalendars.pages[offset // 100] if offset // 100 < len(_FakeCalendars.pages) else None


@pytest.fixture
def fake_calendars(monkeypatch):
    import yfinance as yf
    monkeypatch.setattr(yf, "Calendars", _FakeCalendars)
    _FakeCalendars.calls, _FakeCalendars.pages = [], []
    earnings_calendar.clear_cache()
    yield _FakeCalendars
    earnings_calendar.clear_cache()


def _page(rows):
    return pd.DataFrame({"Event Start Date": [pd.Timestamp(ts, tz="UTC") for _, ts in rows]},
                        index=pd.Index([s for s, _ in rows], name="Symbol"))


def test_calendar_end_inclusive_and_ny_dates(fake_calendars):
    fake_calendars.pages = [_page([("SAIC", "2026-08-31 11:10"),      # 07:10 ET → 08-31
                                   ("LATE", "2026-09-04 00:30"),      # 09-03 20:30 ET → 09-03 (dahil)
                                   ("OUT", "2026-09-04 14:00"),       # 09-04 → pencere dışı
                                   ("BRK.B", "2026-09-01 20:00")])]
    got = earnings_calendar.reporters(date(2026, 8, 31), date(2026, 9, 3))
    assert fake_calendars.calls == [("2026-08-31", "2026-09-04")]   # Yahoo bitişi dahil değil → +1
    assert got == {"SAIC": {date(2026, 8, 31)}, "LATE": {date(2026, 9, 3)}, "BRK-B": {date(2026, 9, 1)}}


def test_calendar_paginates(fake_calendars):
    first = _page([(f"T{i}", "2026-09-01 12:00") for i in range(100)])
    fake_calendars.pages = [first, _page([("LAST", "2026-09-02 12:00")])]
    got = earnings_calendar.reporters(date(2026, 8, 31), date(2026, 9, 3))
    assert len(got) == 101 and "LAST" in got


def test_calendar_empty_or_error_is_unknown_and_cached(fake_calendars, monkeypatch):
    assert earnings_calendar.reporters(date(2026, 8, 31), date(2026, 9, 3)) is None   # boş → bilinmiyor
    assert earnings_calendar.reporters(date(2026, 8, 31), date(2026, 9, 3)) is None
    assert len(fake_calendars.calls) == 1          # hata önbellekte: her hisse için yeniden denenmez

    def boom(*a):
        raise RuntimeError("network")
    monkeypatch.setattr(earnings_calendar, "_fetch", boom)
    assert earnings_calendar.reporters(date(2026, 9, 7), date(2026, 9, 10)) is None


def test_calendar_success_cached_once(fake_calendars):
    fake_calendars.pages = [_page([("SAIC", "2026-08-31 11:10")])]
    for _ in range(5):
        earnings_calendar.reporters(date(2026, 8, 31), date(2026, 9, 3))
    assert len(fake_calendars.calls) == 1


# ── ürünün backtest'i: canlıyla aynı kural ───────────────────────────────────

def test_backtest_rejects_signal_when_earnings_unknown(monkeypatch):
    from swing_trader.small_cap.smallcap_backtest import SmallCapBacktester
    bt = SmallCapBacktester()
    bt._reset_state()
    tape = pd.DataFrame(dict(Date=pd.bdate_range(end="2026-10-02", periods=80), Open=20., High=20.5,
                             Low=19.5, Close=20., Volume=1_000_000.))
    bt._data = {"KNOWN": tape, "UNKNOWN": tape}
    bt._earnings = {"KNOWN": [pd.Timestamp("2026-08-05")], "UNKNOWN": None}
    seen = {}

    def scan(ticker, w, **kw):
        seen[ticker] = kw["earnings_dates"]
        return dict(ticker=ticker, quality_score=90, rank_score=90)

    monkeypatch.setattr(bt.engine, "scan_stock", scan)
    bt._scan_and_track(pd.Timestamp("2026-10-02"), {"regime": "BULL", "confidence": "CONFIRMED"}, 60, 10)
    assert seen == {"KNOWN": [pd.Timestamp("2026-08-05")], "UNKNOWN": []}
    assert bt._diag["skip_earnings_unknown"] == 1
    assert [p["signal"]["ticker"] for p in bt.pending_signals] == ["KNOWN"]
