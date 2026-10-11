"""Optional bounded, current-only Yahoo metadata for trigger-positive tickers.

Never silently backfill historical earnings/market cap. A future calendar range
is provider evidence, not a guarantee that an event cannot change. Empty events
remain UNKNOWN. This opt-in command is separate from pure offline scanning.
"""
import time
from datetime import date, datetime

from .bars import safe_error
from .calendar import last_completed
from .master import utc_now


def interpret_calendar(calendar: dict, signal: date) -> tuple[str, list[str]]:
    raw = calendar.get("Earnings Date") if isinstance(calendar, dict) else None
    if not isinstance(raw, list) or not raw:
        return "UNKNOWN", []
    dates = sorted({date.fromisoformat(str(d)[:10]).isoformat() for d in raw})
    # Old calendar / range crossing an already elapsed day is not complete
    # evidence of the upcoming event interval.
    if dates[0] < signal.isoformat():
        return "UNKNOWN", dates
    # Expand the provider's estimated range so a boundary isn't mistaken for
    # a guaranteed event-free interval inside a two-date estimate.
    from datetime import timedelta
    first, last = date.fromisoformat(dates[0]), date.fromisoformat(dates[-1])
    if (last-first).days > 60:
        return "UNKNOWN", dates
    return "VERIFIED", [(first+timedelta(days=i)).isoformat() for i in range((last-first).days+1)]


def collect(scan: dict, snapshot: dict, limit: int = 100) -> dict:
    import yfinance as yf
    from datetime import timezone
    session = date.fromisoformat(scan["inputs"]["session"])
    if session != last_completed(datetime.now(timezone.utc)):
        raise ValueError("CURRENT_FACTS_CANNOT_BACKFILL_HISTORICAL_SCAN")
    candidates = sorted(r["ticker"] for r in scan["rows"] if r["trigger_positive"])
    master = {r["symbol"]: r for r in snapshot["records"] if r["discovery_status"] == "CANDIDATE"}
    output, circuit = {}, False
    for index, symbol in enumerate(candidates):
        fact = {"session": str(session), "source": "YAHOO_CURRENT_OBSERVATION_NOT_HISTORICAL",
                "market_cap": None, "float_shares": None, "earnings_coverage": "UNKNOWN", "earnings_dates": [],
                "errors": [], "logical_requests": 0, "observed_at": utc_now()}
        if index >= limit or circuit:
            fact["errors"].append("NOT_REQUESTED_BUDGET" if index >= limit else "NOT_REQUESTED_CIRCUIT")
            output[symbol] = fact
            continue
        ticker = yf.Ticker(master[symbol]["yfinance_symbol_candidate"])
        try:
            time.sleep(1)
            fact["logical_requests"] += 1
            info = ticker.get_info()
            if info.get("symbol") == master[symbol]["yfinance_symbol_candidate"] and info.get("currency") == "USD":
                fact["market_cap"] = info.get("marketCap")
                fact["float_shares"] = info.get("floatShares")
            else:
                fact["errors"].append("IDENTITY_OR_CURRENCY_UNKNOWN")
            time.sleep(1)
            fact["logical_requests"] += 1
            coverage, dates = interpret_calendar(ticker.calendar, session)
            fact["earnings_coverage"], fact["earnings_dates"] = coverage, dates
            fact["earnings_evidence"] = "PROVIDER_NEXT_REPORTED_OR_ESTIMATED_RANGE_NOT_INDEPENDENT_VERIFICATION"
        except Exception as exc:
            error = safe_error(exc)
            fact["errors"].append(error)
            circuit = error.endswith("_STOP")
        fact["observed_at"] = utc_now()
        output[symbol] = fact
    return output
