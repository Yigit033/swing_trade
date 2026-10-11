"""Isolated cash-equity calendar. No changes to production scheduling.

NYSE/Nasdaq published 2025--2028 schedules; fail closed outside that range.
Sources: nyse.com/trade/hours-calendars, nasdaqtrader.com/Trader.aspx?id=Calendar.
Ad-hoc future closures require a versioned update; never silently extrapolate.
"""
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from swing_trader.utils.market_calendar import nyse_holidays

NY = ZoneInfo("America/New_York")
VERSION = "cash-equities-2025-2028-v1"
EARLY = {date.fromisoformat(s) for s in (
    "2025-07-03", "2025-11-28", "2025-12-24", "2026-11-27",
    "2026-12-24", "2027-11-26", "2028-07-03", "2028-11-24")}


def is_session(day: date) -> bool:
    if not 2025 <= day.year <= 2028:
        raise ValueError("CALENDAR_OUTSIDE_VERIFIED_RANGE")
    holidays = set(nyse_holidays(day.year))
    # NYSE does NOT observe Saturday New Year on the prior Friday.
    holidays.discard(date(2027, 12, 31))
    holidays.add(date(2025, 1, 9))  # President Carter national day of mourning
    return day.weekday() < 5 and day not in holidays


def sessions(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)
            if is_session(start + timedelta(days=i))]


def session_close(day: date) -> datetime:
    if not is_session(day):
        raise ValueError("NOT_A_SESSION")
    return datetime.combine(day, time(13 if day in EARLY else 16), NY)


def last_completed(now: datetime, publication_delay_minutes: int = 30) -> date:
    if now.tzinfo is None:
        raise ValueError("AWARE_TIMESTAMP_REQUIRED")
    day = now.astimezone(NY).date()
    while not is_session(day) or now < session_close(day) + timedelta(minutes=publication_delay_minutes):
        day -= timedelta(days=1)
    return day


def next_session(day: date) -> date:
    day += timedelta(days=1)
    while not is_session(day):
        day += timedelta(days=1)
    return day
