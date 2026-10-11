"""
Trail replay must not use the persisted trailing_stop as an input.

Live incident (KYMR id=83, entry 2026-08-12 @ $110.22):
  A later (corrupt / incomplete) bar ratcheted trailing_stop to $110.17.
  The next tracker run applied that persisted trail to the Aug 14 wick
  (Low $110.05) and closed TRAILED −0.05%. A fresh replay from initial_stop
  stays OPEN — Low $110.05 never tags the honest chandelier ($105.62).

Persisted trailing_stop is a display cache, not a stop input.
"""

import pandas as pd
from unittest.mock import Mock

from swing_trader.paper_trading.tracker import PaperTradeTracker


def _kymr_bars():
    # Entry day + two sessions. Matches yfinance daily OHLC used in the post-mortem.
    return pd.DataFrame({
        "Date": pd.to_datetime(["2026-08-12", "2026-08-13", "2026-08-14"]).date,
        "Open":  [110.22, 114.73, 112.90],
        "High":  [115.00, 114.73, 118.04],
        "Low":   [109.08, 112.44, 110.05],
        "Close": [114.03, 114.42, 117.39],
        "Volume": [961000, 481700, 468600],
    })


def _trade(**overrides):
    t = {
        "entry_price": 110.22,
        "stop_loss": 99.87,
        "initial_stop": 99.87,
        "trailing_stop": 99.87,
        "target": 125.84,
        "target_2": 144.71,
        "atr": 4.1405,
        "entry_date": "2026-08-12T13:30:00Z",
        "position_size": 14,
        "max_hold_days": 8,
        "partial_exit_price": None,
    }
    t.update(overrides)
    return t


def test_kymr_poisoned_trail_must_not_close():
    """Persisted $110.17 trail must be ignored — same bars stay OPEN."""
    tracker = PaperTradeTracker(Mock())
    poisoned = _trade(trailing_stop=110.17)
    status, exit_price, _, _, trail = tracker.check_exit_conditions(
        poisoned, _kymr_bars()
    )
    assert status == "OPEN"
    assert exit_price == 0
    # Honest chandelier: peak 118.04 − 3×4.1405 = 105.62
    assert trail == 105.62


def test_kymr_fresh_replay_stays_open():
    tracker = PaperTradeTracker(Mock())
    status, _, _, _, trail = tracker.check_exit_conditions(_trade(), _kymr_bars())
    assert status == "OPEN"
    assert trail == 105.62


def test_trail_still_fires_when_low_actually_tags_chandelier():
    """Sanity: a later bar that tags the recomputed trail still exits TRAILED.

    Same-bar order: Aug 14 raises the trail to 105.62 AFTER the stop check, so
    that day's low cannot be the tag. The next session can.
    """
    tracker = PaperTradeTracker(Mock())
    bars = _kymr_bars()
    nxt = pd.DataFrame({
        "Date": pd.to_datetime(["2026-08-17"]).date,
        "Open": [107.00], "High": [108.00], "Low": [105.00],
        "Close": [106.00], "Volume": [400000],
    })
    bars = pd.concat([bars, nxt], ignore_index=True)
    status, exit_price, _, reason, _ = tracker.check_exit_conditions(_trade(), bars)
    assert status == "TRAILED"
    assert exit_price == 105.62
    assert "Trailing stop" in reason
