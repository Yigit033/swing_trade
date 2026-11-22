"""SmallCapBacktester kapanış muhasebesi (kısmi bacak + kalan) ve slippage."""

import pandas as pd
import pytest

from swing_trader.small_cap.smallcap_backtest import SmallCapBacktester


def _open(**kw):
    tr = dict(ticker="T", swing_type="A", pathway="VCE", quality_score=80, rank_score=80,
              entry_day=pd.Timestamp("2026-10-01"), entry_price=10.0, size=10, stop=9.0,
              target=11.0, target_2=12.0, max_hold_days=20, initial_risk=10.0, status="TIMEOUT",
              reason="", exit_day=pd.Timestamp("2026-10-05"), exit_price=11.0, bars=3, sold=0,
              partial_price=0.0, partial_day=None, partial_realized=False)
    tr.update(kw)
    return tr


@pytest.fixture
def bt():
    b = SmallCapBacktester(config=None)
    b.settings = b.settings.model_copy(update={"slippage_bps_per_side": 0, "commission_usd_per_side": 0.0})
    b.initial_capital = b.capital = 10_000
    return b


def test_close_trade_full_position(bt):
    tr = _open()
    bt._close_trade(tr, 11.0, tr["exit_day"], "TIMEOUT", "", 3)
    t = bt.trades[0]
    assert t["pnl_dollar"] == 10.0 and t["shares"] == 10
    assert abs(bt.capital - 10_010.0) < 0.01
    assert t["r_multiple"] == pytest.approx(1.0)


def test_close_trade_after_partial_realized_on_t1_day(bt):
    # T1 günü kısmi bacak (5 hisse @11) sermayeye eklenir, kapanışta tekrar eklenmez.
    tr = _open(sold=5, partial_price=11.0, partial_day=pd.Timestamp("2026-10-02"))
    bt.open_trades = [tr]
    bt._settle_exits(pd.Timestamp("2026-10-02"))
    assert abs(bt.capital - 10_005.0) < 0.01 and tr["partial_realized"]
    tr["exit_price"] = 12.0
    bt._settle_exits(pd.Timestamp("2026-10-05"))
    t = bt.trades[0]
    assert t["pnl_dollar"] == 15.0
    assert abs(t["exit_price"] - 11.5) < 0.02
    assert abs(bt.capital - 10_015.0) < 0.01


def test_fill_prices_slip_zero(bt):
    assert bt._entry_fill_price(100.0) == 100.0
    assert bt._exit_fill_price(100.0) == 100.0


def test_slippage_charged_on_both_legs():
    b = SmallCapBacktester(config=None)
    b.settings = b.settings.model_copy(update={"slippage_bps_per_side": 50, "commission_usd_per_side": 0.0})
    b.initial_capital = b.capital = 10_000
    tr = _open(entry_price=100.0, size=10, exit_price=100.0, initial_risk=100.0)
    b._close_trade(tr, 100.0, tr["exit_day"], "TIMEOUT", "", 3)
    # giriş 100.5, çıkış 99.5 → 10 × −1.0
    assert b.trades[0]["pnl_dollar"] == pytest.approx(-10.0)


def test_midas_fee_charged_per_order(bt):
    """Sabit emir ücreti: tam işlemde 2 emir, kısmi satışlı işlemde 3 emir."""
    bt.settings = bt.settings.model_copy(update={"commission_usd_per_side": 1.5})
    tr = _open()
    bt._close_trade(tr, 11.0, tr["exit_day"], "TIMEOUT", "", 3)
    assert bt.trades[0]["pnl_dollar"] == 10.0 - 3.0 and abs(bt.capital - 10_007.0) < 0.01
    tr = _open(sold=5, partial_price=11.0, partial_day=pd.Timestamp("2026-10-02"))
    bt.open_trades = [tr]
    bt._settle_exits(pd.Timestamp("2026-10-02"))
    tr["exit_price"] = 12.0
    bt._settle_exits(pd.Timestamp("2026-10-05"))
    assert bt.trades[1]["pnl_dollar"] == 15.0 - 4.5
    assert abs(bt.capital - (10_007.0 + 10.5)) < 0.01
