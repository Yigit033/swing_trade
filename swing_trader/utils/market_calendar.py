"""
NYSE trading-calendar helper — dependency-free.

Used by the PENDING → next-day-open confirmation flow so the *displayed*
expected entry date accounts for weekends AND US market holidays (the old
code skipped only weekends, so it told users a signal would confirm on
Juneteenth — a closed session — and then looked "stuck").

The confirmation logic itself enters on the first REAL price bar after the
signal date (so it was always correct); this module only makes the
human-facing date/“why pending” accurate.
"""

from __future__ import annotations

from datetime import date, datetime, time as dtime, timedelta
from functools import lru_cache
from typing import Optional, Set

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python < 3.9
    from backports.zoneinfo import ZoneInfo

_NYSE_TZ = ZoneInfo("America/New_York")


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th `weekday` (Mon=0) of month, e.g. 3rd Monday of January."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Last `weekday` of month (e.g. last Monday of May)."""
    if month == 12:
        d = date(year, 12, 31)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


def _easter(year: int) -> date:
    """Anonymous Gregorian computus → Easter Sunday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = ((h + m - 7 * n + 114) % 31) + 1
    return date(year, month, day)


def _observed(d: date) -> date:
    """NYSE rule: holiday on Sat → observed Fri; on Sun → observed Mon."""
    if d.weekday() == 5:        # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:        # Sunday
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=16)
def nyse_holidays(year: int) -> Set[date]:
    """Full-day NYSE market holidays for a given year (observed dates)."""
    h = {
        _observed(date(year, 1, 1)),                 # New Year's Day
        _nth_weekday(year, 1, 0, 3),                 # MLK Jr. Day (3rd Mon Jan)
        _nth_weekday(year, 2, 0, 3),                 # Washington's Birthday (3rd Mon Feb)
        _easter(year) - timedelta(days=2),           # Good Friday
        _last_weekday(year, 5, 0),                   # Memorial Day (last Mon May)
        _nth_weekday(year, 9, 0, 1),                 # Labor Day (1st Mon Sep)
        _nth_weekday(year, 11, 3, 4),                # Thanksgiving (4th Thu Nov)
        _observed(date(year, 12, 25)),               # Christmas
        _observed(date(year, 7, 4)),                 # Independence Day
    }
    if year >= 2022:
        h.add(_observed(date(year, 6, 19)))          # Juneteenth (federal since 2021)
    return h


def is_trading_day(d: date) -> bool:
    """True if `d` is a regular NYSE session (weekday, not a holiday)."""
    if d.weekday() >= 5:
        return False
    return d not in nyse_holidays(d.year)


def next_trading_day(d: date) -> date:
    """First NYSE session strictly AFTER `d` (skips weekends + holidays)."""
    nxt = d + timedelta(days=1)
    while not is_trading_day(nxt):
        nxt += timedelta(days=1)
    return nxt


def next_session_open_et(d: date) -> datetime:
    """09:30 ET open of the first NYSE session strictly AFTER `d`."""
    nxt = next_trading_day(d)
    return datetime(nxt.year, nxt.month, nxt.day, 9, 30, tzinfo=_NYSE_TZ)


def prev_trading_day(d: date) -> date:
    """Last NYSE session strictly BEFORE `d` (skips weekends + holidays)."""
    prev = d - timedelta(days=1)
    while not is_trading_day(prev):
        prev -= timedelta(days=1)
    return prev


def last_completed_session(now: Optional[datetime] = None) -> date:
    """
    En son TAMAMLANMIŞ NYSE seansının tarihi.

    Sistem yalnız tamamlanmış barlarla karar verir; bu fonksiyon "veri hangi
    güne kadar gelmiş olmalı" sorusunun beklenen cevabını verir. Kapanış
    16:00 ET; o saatten önceyse bugünün barı henüz tamamlanmamıştır.

    2026-08-04 arızası bunun için gerekli oldu: yfinance 03 Ağustos barını
    tüm piyasa için OHLC=NaN / hacim dolu döndürdü. NaN-guard barı düşürünce
    sistem 31 Temmuz'a (iki seans eski) düşüyor ve bunu kimse fark etmiyordu —
    tarama, rejim tespiti ve çıkış kontrolü sessizce bayat veriyle koşuyordu.
    Bu fonksiyon "olması gereken" ile "gelen"i karşılaştırmayı mümkün kılar.
    """
    et = (now or datetime.now(tz=_NYSE_TZ)).astimezone(_NYSE_TZ)
    today = et.date()
    if is_trading_day(today) and et.time() >= dtime(16, 0):
        return today
    return prev_trading_day(today)


def sessions_behind(bar_date: date, now: Optional[datetime] = None) -> int:
    """
    `bar_date` beklenen son tamamlanmış seansın KAÇ seans gerisinde?

    0 = güncel, 1 = bir seans eski, ... Negatif olamaz (ileri tarihli bar 0 sayılır).
    """
    expected = last_completed_session(now)
    if bar_date >= expected:
        return 0
    n = 0
    cur = expected
    while cur > bar_date and n < 30:
        cur = prev_trading_day(cur)
        n += 1
    return n


def entry_window_open(bar_date: date, now: Optional[datetime] = None) -> bool:
    """
    True while the measured entry for a signal bar is still takeable.

    The VCE edge was measured with entry at the OPEN of the first session
    AFTER the signal bar (t+1 open). Once that open has passed, entering is
    no longer the validated pattern (it becomes an unmeasured t+2 entry) —
    the paper tracker refuses to create the trade.
    """
    now_et = (now or datetime.now(tz=_NYSE_TZ)).astimezone(_NYSE_TZ)
    return now_et < next_session_open_et(bar_date)


def us_market_session(now: Optional[datetime] = None) -> str:
    """
    Current NYSE session state: 'pre_market' | 'regular' | 'after_hours' | 'closed'.

    Scanner uses this to warn when a scan runs pre-market: Finviz's live
    Change/RelVol columns reset before the open, so the RelVol-based screens
    (incl. the primary VCE breakout-day queries) structurally return 0 rows
    and the composite ranking runs on pre-market noise.

    Half-days (1pm early closes) are treated as regular full sessions — the
    distinction doesn't matter for the scan-quality warning this feeds.
    """
    et = (now or datetime.now(tz=_NYSE_TZ)).astimezone(_NYSE_TZ)
    if not is_trading_day(et.date()):
        return "closed"
    t = et.time()
    if t < dtime(9, 30):
        return "pre_market"
    if t < dtime(16, 0):
        return "regular"
    return "after_hours"
