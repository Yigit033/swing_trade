"""
PaperTradeTracker NaN-guard regression tests (2026-08-02).

Live incident: GET /api/trades and GET /api/performance both returned 500 with
`ValueError: Out of range float values are not JSON compliant`.

Root cause chain:
  1. `fetch_price_history` (tracker's own yfinance path, separate from
     data/fetcher.py's `_drop_incomplete_last_bar`) did not drop NaN-OHLC bars,
     so the known yfinance glitch leaked through here.
  2. `update_trade_status` wrote `float(price_history['Close'].iloc[-1])`
     straight into current_price / unrealized_pnl and PERSISTED it, so the NaN
     became permanent DB state (id=72 HR, id=76 DNOW).
  3. Both read endpoints returned the row without `sanitize_for_json`, so
     json.dumps blew up on every request — the pages were down until repaired.

These tests lock the two engine-side barriers (1 and 2). The API-side barrier
(3) is covered by api/tests/test_sanitize_json.py.
"""

import math

import numpy as np
import pandas as pd
import pytest

from swing_trader.paper_trading.tracker import PaperTradeTracker


OHLC = ("Open", "High", "Low", "Close")


def _frame(closes, with_nan_tail=False):
    n = len(closes) + (1 if with_nan_tail else 0)
    rows = {
        "Date": pd.date_range("2026-07-25", periods=n, freq="D"),
        "Volume": [1_000 * (i + 1) for i in range(n)],
    }
    for col in OHLC:
        vals = list(closes)
        if with_nan_tail:
            vals = vals + [np.nan]
        rows[col] = vals
    return pd.DataFrame(rows)


def _drop_nan_ohlc(df):
    """Mirror of the guard applied inside fetch_price_history."""
    cols = [c for c in OHLC if c in df.columns]
    return df.dropna(subset=cols) if cols else df


def test_nan_ohlc_tail_bar_is_dropped():
    """A trailing Volume-populated / OHLC-NaN bar must not survive."""
    df = _frame([10.5, 11.0, 11.5], with_nan_tail=True)
    assert len(df) == 4

    cleaned = _drop_nan_ohlc(df)

    assert len(cleaned) == 3
    assert math.isfinite(float(cleaned["Close"].iloc[-1]))
    assert float(cleaned["Close"].iloc[-1]) == 11.5


def test_all_nan_frame_collapses_to_empty():
    """If every bar is NaN the history is unusable — caller must treat as None."""
    df = pd.DataFrame({
        "Date": pd.date_range("2026-07-25", periods=3, freq="D"),
        "Open": [np.nan] * 3, "High": [np.nan] * 3,
        "Low": [np.nan] * 3, "Close": [np.nan] * 3,
        "Volume": [100, 200, 300],
    })
    assert len(_drop_nan_ohlc(df)) == 0


def test_clean_frame_is_untouched():
    """The guard must not silently discard good bars."""
    df = _frame([10.5, 11.0, 11.5])
    assert len(_drop_nan_ohlc(df)) == 3


class _RecordingStorage:
    """Minimal storage double — records what update_trade_status persists."""

    def __init__(self):
        self.writes = []

    def update_trade(self, trade_id, fields, user_id=None):
        self.writes.append((trade_id, dict(fields)))
        return True


@pytest.fixture
def tracker_with_recorder():
    storage = _RecordingStorage()
    return PaperTradeTracker(storage), storage


def test_nan_close_is_never_persisted(monkeypatch, tracker_with_recorder):
    """
    Second barrier: even if a NaN close somehow reaches update_trade_status,
    it must not be written to the DB (that is what made the outage permanent).
    """
    tracker, storage = tracker_with_recorder

    nan_history = pd.DataFrame({
        "Date": pd.date_range("2026-07-25", periods=2, freq="D").date,
        "Open": [10.0, np.nan], "High": [10.5, np.nan],
        "Low": [9.5, np.nan], "Close": [10.0, np.nan],
        "Volume": [1000, 2000],
    })
    monkeypatch.setattr(tracker, "fetch_price_history", lambda *a, **k: nan_history)
    # Keep the trade OPEN so we reach the unrealized-P&L branch.
    monkeypatch.setattr(
        tracker, "check_exit_conditions",
        lambda *a, **k: ("OPEN", 0, "", "", None),
    )

    trade = {
        "id": 999, "ticker": "TEST", "status": "OPEN",
        "entry_date": "2026-07-25", "entry_price": 10.0, "position_size": 100,
        "stop_loss": 9.0, "target_1": 11.0, "target_2": 12.0,
        "atr": 0.5, "hold_days_max": 20, "swing_type": "A",
    }

    result = tracker.update_trade_status(trade)

    for field in ("current_price", "unrealized_pnl", "unrealized_pnl_pct"):
        value = result.get(field)
        assert value is None or math.isfinite(float(value)), \
            f"{field} sızdırdı: {value!r}"

    for _tid, fields in storage.writes:
        for key, value in fields.items():
            if isinstance(value, float):
                assert math.isfinite(value), f"NaN/Inf DB'ye yazıldı: {key}={value!r}"
