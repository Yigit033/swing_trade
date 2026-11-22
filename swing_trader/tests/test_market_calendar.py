"""NYSE trading-calendar helper tests — lock in holiday handling."""

from datetime import date, datetime

from swing_trader.utils.market_calendar import (
    _NYSE_TZ,
    entry_window_open,
    is_trading_day,
    next_session_open_et,
    next_trading_day,
    nyse_holidays,
    us_market_session,
)


def test_juneteenth_2026_is_closed():
    # 2026-06-19 is a Friday Juneteenth — the bug that made pending look stuck.
    assert not is_trading_day(date(2026, 6, 19))
    # Signal on Thu 06-18 → next session is Mon 06-22 (skip holiday + weekend).
    assert next_trading_day(date(2026, 6, 18)) == date(2026, 6, 22)


def test_weekend_skip():
    # Friday 2026-06-12 → Monday 2026-06-15
    assert next_trading_day(date(2026, 6, 12)) == date(2026, 6, 15)


def test_normal_weekday():
    # Mon 2026-06-15 → Tue 2026-06-16
    assert next_trading_day(date(2026, 6, 15)) == date(2026, 6, 16)


def test_2026_holiday_set():
    h = nyse_holidays(2026)
    for d in [
        date(2026, 1, 1),    # New Year
        date(2026, 1, 19),   # MLK
        date(2026, 2, 16),   # Presidents
        date(2026, 4, 3),    # Good Friday
        date(2026, 5, 25),   # Memorial
        date(2026, 6, 19),   # Juneteenth
        date(2026, 7, 3),    # Independence (observed, Jul 4 = Sat)
        date(2026, 9, 7),    # Labor
        date(2026, 11, 26),  # Thanksgiving
        date(2026, 12, 25),  # Christmas
    ]:
        assert d in h, f"{d} should be an NYSE holiday"


def test_independence_day_observed_when_saturday():
    # 2026-07-04 is Saturday → observed Friday 07-03
    assert date(2026, 7, 3) in nyse_holidays(2026)
    assert not is_trading_day(date(2026, 7, 3))


def _et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=_NYSE_TZ)


def test_us_market_session_states():
    # Fri 2026-07-17, 05:10 ET — the pre-market scan that surfaced the Finviz bug
    assert us_market_session(_et(2026, 7, 17, 5, 10)) == "pre_market"
    assert us_market_session(_et(2026, 7, 17, 9, 29)) == "pre_market"
    assert us_market_session(_et(2026, 7, 17, 9, 30)) == "regular"
    assert us_market_session(_et(2026, 7, 17, 15, 59)) == "regular"
    assert us_market_session(_et(2026, 7, 17, 16, 0)) == "after_hours"
    assert us_market_session(_et(2026, 7, 18, 12, 0)) == "closed"     # Saturday
    assert us_market_session(_et(2026, 6, 19, 12, 0)) == "closed"     # Juneteenth


def test_entry_window_lifecycle():
    # Signal bar = Thu 2026-07-16 → measured entry = Fri 07-17 09:30 ET open
    bar = date(2026, 7, 16)
    assert next_session_open_et(bar) == _et(2026, 7, 17, 9, 30)
    # After-close scan the same evening: window open (canonical flow)
    assert entry_window_open(bar, _et(2026, 7, 16, 16, 5))
    # Pre-market next morning: still open (entry = today's open, upcoming)
    assert entry_window_open(bar, _et(2026, 7, 17, 5, 10))
    # At/after the open the measured entry is gone → t+2 would be unmeasured
    assert not entry_window_open(bar, _et(2026, 7, 17, 9, 30))
    assert not entry_window_open(bar, _et(2026, 7, 17, 14, 0))


def test_entry_window_weekend_and_holiday():
    # Friday bar → entry Monday open; all weekend the window stays open
    assert entry_window_open(date(2026, 6, 12), _et(2026, 6, 14, 20, 0))
    assert not entry_window_open(date(2026, 6, 12), _et(2026, 6, 15, 10, 0))
    # Thu 2026-06-18 bar → Fri is Juneteenth (closed) → entry Mon 06-22 open
    assert next_session_open_et(date(2026, 6, 18)) == _et(2026, 6, 22, 9, 30)
    assert entry_window_open(date(2026, 6, 18), _et(2026, 6, 19, 12, 0))


def test_us_market_session_converts_timezones():
    # 12:10 Istanbul (UTC+3) on 2026-07-17 == 05:10 ET → pre_market
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    ist = datetime(2026, 7, 17, 12, 10, tzinfo=ZoneInfo("Europe/Istanbul"))
    assert us_market_session(ist) == "pre_market"


def test_special_closure_carter_mourning_2025():
    # 2025-01-09: NYSE ulusal yas günü kapalı. Eksikken açık pozisyonlar
    # execution.evaluate içinde MISSING_SESSION ile REVIEW_REQUIRED'a düşüyordu.
    assert not is_trading_day(date(2025, 1, 9))
    assert next_trading_day(date(2025, 1, 8)) == date(2025, 1, 10)
    assert date(2025, 1, 9) in nyse_holidays(2025)


def test_special_closure_bush_mourning_2018():
    assert not is_trading_day(date(2018, 12, 5))


def test_saturday_new_year_does_not_close_prior_friday():
    # 2022-01-01 ve 2028-01-01 Cumartesi; NYSE önceki Cuma açıktı / açık olacak.
    assert is_trading_day(date(2021, 12, 31))
    assert is_trading_day(date(2027, 12, 31))
    assert date(2021, 12, 31) not in nyse_holidays(2022)
    assert date(2027, 12, 31) not in nyse_holidays(2028)
    # Pazar 1 Ocak → Pazartesi gözlenir (2023-01-02)
    assert not is_trading_day(date(2023, 1, 2))


def test_open_trade_spanning_carter_closure_is_not_frozen():
    """2025-01-09'u kapsayan pozisyon artık MISSING_SESSION'a düşmemeli."""
    import pandas as pd
    from swing_trader.paper_trading import execution

    days = ["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-10", "2025-01-13"]
    hist = pd.DataFrame({"Date": pd.to_datetime(days), "Open": 10.0, "High": 10.2,
                         "Low": 9.9, "Close": 10.1, "Volume": 1e6})
    trade = {"id": 1, "ticker": "TST", "entry_date": "2025-01-06T14:30:00Z", "entry_price": 10.0,
             "position_size": 100, "stop_loss": 9.0, "initial_stop": 9.0, "target": 12.0,
             "target_2": 13.0, "max_hold_days": 10, "atr": 0.4, "status": "OPEN"}
    _updates, result = execution.evaluate(trade, hist, 0.33)
    reason = result[3]
    assert "REVIEW_REQUIRED" not in reason, reason
