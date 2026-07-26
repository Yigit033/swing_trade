"""
_drop_incomplete_last_bar NaN-guard regression tests (2026-07-25).

Live incident: yfinance returned a completed-looking row (past trading day,
non-today) with populated Volume but NaN OHLC — a provider glitch, not an
incomplete-session artifact. The date-equality check alone ("is this today?")
missed it because the row's date was a stale Friday during a Sunday scan.
_validate_ohlcv_data's 10% NaN-ratio threshold also let it through (1/62 rows).
The NaN silently propagated into rolling MA/ATR calcs, making every
`close > ma50`-style VCE trigger check evaluate to False — a real market
signal masquerading as a rejection.
"""

import pandas as pd

from swing_trader.data.fetcher import DataFetcher


def _dates(n, start="2026-06-01"):
    return pd.date_range(start, periods=n, freq="D")


def test_drops_trailing_nan_ohlc_row_regardless_of_date():
    # Mirrors the live incident: last row is a STALE date (not "today"), so
    # the old date-equality check alone would never fire — only the NaN
    # check catches it.
    df = pd.DataFrame({
        "Date": _dates(5),
        "Open": [10.0, 10.1, 10.2, 10.3, float("nan")],
        "High": [10.5, 10.6, 10.7, 10.8, float("nan")],
        "Low": [9.8, 9.9, 10.0, 10.1, float("nan")],
        "Close": [10.2, 10.3, 10.4, 10.5, float("nan")],
        "Volume": [1000, 1100, 1200, 1300, 1400],  # volume IS populated — the glitch signature
    })
    out = DataFetcher._drop_incomplete_last_bar(df)
    assert len(out) == 4
    assert not out["Close"].isna().any()
    assert out["Close"].iloc[-1] == 10.5


def test_keeps_clean_trailing_bar():
    df = pd.DataFrame({
        "Date": _dates(3),
        "Open": [10.0, 10.1, 10.2],
        "High": [10.5, 10.6, 10.7],
        "Low": [9.8, 9.9, 10.0],
        "Close": [10.2, 10.3, 10.4],
        "Volume": [1000, 1100, 1200],
    })
    out = DataFetcher._drop_incomplete_last_bar(df)
    assert len(out) == 3
    assert out["Close"].iloc[-1] == 10.4


def test_nan_row_dropping_is_recursive_not_just_last():
    # After dropping one bad trailing row, if the NEW last row is also NaN
    # (pathological double-glitch), it should be caught too since callers
    # may re-invoke; single call only strips one trailing bad row by design
    # (matches the pre-existing intraday-drop semantics — one bar at a time).
    df = pd.DataFrame({
        "Date": _dates(3),
        "Open": [10.0, 10.1, float("nan")],
        "High": [10.5, 10.6, float("nan")],
        "Low": [9.8, 9.9, float("nan")],
        "Close": [10.2, 10.3, float("nan")],
        "Volume": [1000, 1100, 1200],
    })
    out = DataFetcher._drop_incomplete_last_bar(df)
    assert len(out) == 2
    assert not out["Close"].isna().any()


def test_empty_df_after_drop_returns_empty_not_error():
    df = pd.DataFrame({
        "Date": _dates(1),
        "Open": [float("nan")],
        "High": [float("nan")],
        "Low": [float("nan")],
        "Close": [float("nan")],
        "Volume": [500],
    })
    out = DataFetcher._drop_incomplete_last_bar(df)
    assert len(out) == 0


def test_nan_in_middle_of_series_not_touched_by_this_guard():
    # This guard only strips the TRAILING bad bar — a NaN buried mid-series
    # is a separate (existing, unaddressed-by-this-fix) data-quality concern
    # handled by _validate_ohlcv_data's ratio threshold, not this function.
    df = pd.DataFrame({
        "Date": _dates(4),
        "Open": [10.0, float("nan"), 10.2, 10.3],
        "High": [10.5, float("nan"), 10.7, 10.8],
        "Low": [9.8, float("nan"), 10.0, 10.1],
        "Close": [10.2, float("nan"), 10.4, 10.5],
        "Volume": [1000, 1100, 1200, 1300],
    })
    out = DataFetcher._drop_incomplete_last_bar(df)
    assert len(out) == 4  # trailing row is clean, mid-series NaN untouched
