"""Unit tests for PENDING → OPEN / REJECTED confirmation logic."""

import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from swing_trader.paper_trading.tracker import PaperTradeTracker


def _pending_row(**kwargs):
    base = {
        "id": 1,
        "ticker": "TEST",
        "entry_date": "2026-03-24 19:10",
        "entry_price": 100.0,
        "signal_price": 100.0,
        "stop_loss": 94.0,
        "target": 112.0,
        "target_2": 115.0,
        "swing_type": "A",
        "atr": 0,
        "position_size": 100,
        "status": "PENDING",
    }
    base.update(kwargs)
    return base


class ConfirmPendingTradesTest(unittest.TestCase):
    def test_confirms_at_next_session_open(self):
        storage = MagicMock()
        storage.get_open_trades.return_value = [_pending_row()]
        tracker = PaperTradeTracker(storage)

        hist = pd.DataFrame(
            {
                "Date": [date(2026, 3, 24), date(2026, 3, 25), date(2026, 3, 26)],
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.5, 101.5, 102.5],
            }
        )

        with patch.object(tracker, "fetch_price_history", return_value=hist):
            tracker.confirm_pending_trades(None)

        storage.update_trade.assert_called_once()
        call_kw = storage.update_trade.call_args[0][1]
        self.assertEqual(call_kw["status"], "OPEN")
        self.assertEqual(call_kw["entry_price"], 101.0)

    @patch("swing_trader.paper_trading.tracker.date")
    def test_stale_pending_rejects_when_no_price_data(self, mock_date):
        """After MAX_STALE_PENDING_CALENDAR_DAYS, empty history → REJECTED."""
        mock_date.today.return_value = date(2026, 4, 20)
        mock_date.side_effect = date

        storage = MagicMock()
        storage.get_open_trades.return_value = [_pending_row()]
        tracker = PaperTradeTracker(storage)

        with patch.object(tracker, "fetch_price_history", return_value=None):
            tracker.confirm_pending_trades(None)

        storage.close_trade.assert_called_once()
        args = storage.close_trade.call_args[0]
        self.assertEqual(args[3], "REJECTED")

    def test_gap_up_rejects(self):
        storage = MagicMock()
        storage.get_open_trades.return_value = [_pending_row()]
        tracker = PaperTradeTracker(storage)

        hist = pd.DataFrame(
            {
                "Date": [date(2026, 3, 24), date(2026, 3, 25)],
                "Open": [100.0, 110.0],
                "High": [100.0, 111.0],
                "Low": [100.0, 109.0],
                "Close": [100.0, 110.0],
            }
        )

        with patch.object(tracker, "fetch_price_history", return_value=hist):
            tracker.confirm_pending_trades(None)

        storage.close_trade.assert_called_once()
        self.assertIn("Gap-up", storage.close_trade.call_args[0][4])


if __name__ == "__main__":
    unittest.main()
