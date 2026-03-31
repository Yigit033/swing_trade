"""Unit tests for SmallCapBacktester P/L close logic (partial + remainder)."""

import pytest

from swing_trader.small_cap.smallcap_backtest import SmallCapBacktester


@pytest.fixture
def bt():
    b = SmallCapBacktester(config=None)
    b.initial_capital = 10_000
    b.capital = 10_000
    return b


def test_close_trade_full_position(bt):
    t = {
        "entry_price": 10.0,
        "exit_price": 11.0,
        "shares": 10,
        "initial_shares": 10,
        "partial_pnl_dollar": 0.0,
        "partial_shares": 0,
        "partial_exit_price": 0.0,
    }
    bt._close_trade(t)
    assert t["pnl_dollar"] == 10.0
    assert t["shares"] == 10
    assert abs(bt.capital - 10_010.0) < 0.01


def test_close_trade_after_partial(bt):
    bt.capital = 10_050.0
    t = {
        "entry_price": 10.0,
        "exit_price": 12.0,
        "shares": 5,
        "initial_shares": 10,
        "partial_pnl_dollar": 5.0,
        "partial_shares": 5,
        "partial_exit_price": 11.0,
    }
    bt._close_trade(t)
    assert t["pnl_dollar"] == 15.0
    assert abs(t["exit_price"] - 11.5) < 0.02
    assert abs(bt.capital - (10_050.0 + 10.0)) < 0.01


def test_fill_prices_slip_zero():
    b = SmallCapBacktester(config=None)
    b.settings = b.settings.model_copy(update={"slippage_bps_per_side": 0})
    assert b._entry_fill_price(100.0) == 100.0
    assert b._exit_fill_price(100.0) == 100.0
