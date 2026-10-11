"""Raised UI stop_loss must stop the last completed bar (ASST 2026-09).

Kaydet wrote stop_loss=$25.50 to the DB. Exit replay seeds from initial_stop
($16.32) and ignores the user stop on historical bars (a $25.50 stop applied
from entry $19 would close day one). The last completed bar is the stop the
trader sees — last close $24 must STOP, not stay OPEN.
"""

from datetime import date

import pandas as pd

from swing_trader.paper_trading.storage import PaperTradeStorage
from swing_trader.paper_trading.tracker import PaperTradeTracker


def _tracker(tmp_path) -> PaperTradeTracker:
    return PaperTradeTracker(PaperTradeStorage(db_path=str(tmp_path / "paper.db")))


def _bars() -> pd.DataFrame:
    # Entry bar is included. Policy becomes effective after Aug 31 close.
    return pd.DataFrame(
        {
            "Date": [date(2026, 8, d) for d in (25, 26, 27, 28, 31)] + [date(2026, 9, 1)],
            "Open": [19.04, 21., 21., 21., 21., 24.20],
            "High": [19.50, 22., 22., 22., 22., 24.40],
            "Low": [18.80, 20., 20., 20., 20., 23.50],
            "Close": [19.04, 21.5, 21.5, 21.5, 21.5, 24.00],
            "Volume": [1_000_000] * 6,
        }
    )


def _asst(**overrides) -> dict:
    trade = {
        "entry_price": 19.04,
        "position_size": 100,
        "stop_loss": 25.50,
        "initial_stop": 16.32,
        "target": 40.00,
        "max_hold_days": 14,
        "entry_date": "2026-08-25T13:30:00Z",
        "policy_effective_at": "2026-08-31T20:01:00Z",
        "atr": 0.0,
        "partial_exit_price": 0,
    }
    trade.update(overrides)
    return trade


def test_raised_stop_stops_on_last_bar_not_earlier_dip(tmp_path):
    """Aug 26 Low $20 is below $25.50 but is NOT the last bar — must not retro-stop."""
    status, exit_price, _, _, _ = _tracker(tmp_path).check_exit_conditions(_asst(), _bars())
    assert status == "STOPPED"
    # Sep 1 opened below the raised stop → fill at Open (gap through).
    assert exit_price == 24.20


def test_unedited_stop_stays_open_when_price_above_initial(tmp_path):
    status, _, _, _, _ = _tracker(tmp_path).check_exit_conditions(
        _asst(stop_loss=16.32),
        _bars(),
    )
    assert status == "OPEN"


def test_middle_dip_below_raised_stop_does_not_close_if_last_bar_recovered(tmp_path):
    bars = _bars()
    bars.loc[5, ["Open", "High", "Low", "Close"]] = [26.0, 27.0, 25.8, 26.5]
    status, _, _, _, _ = _tracker(tmp_path).check_exit_conditions(_asst(), bars)
    assert status == "OPEN"
