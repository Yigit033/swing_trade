"""Ek X (TRD) saf kuralları: haftalık karar, sabit %8 stop, 50 günlük ortalama çıkışı, risk tabanlı boyut."""
import numpy as np
import pandas as pd
import pytest

from lab import pro_rules as pr
from lab import trd_rules as tr

CAL = pd.bdate_range("2023-01-02", periods=160)          # Pazartesi başlar → 99. indeks Cuma, 100. indeks Pazartesi
START, END = CAL[100], CAL[139]


def panel(closes: dict, opens: dict = None, lows: dict = None):
    C = pd.DataFrame({t: np.asarray(v, dtype=float) for t, v in closes.items()}, index=CAL)
    O = C.shift(1).fillna(C.iloc[0])
    for t, m in (opens or {}).items():
        for i, v in m.items():
            O.iat[i, C.columns.get_loc(t)] = v
    Lw = np.minimum(O, C)
    for t, m in (lows or {}).items():
        for i, v in m.items():
            Lw.iat[i, C.columns.get_loc(t)] = v
    return {"Open": O, "High": np.maximum(O, C), "Low": Lw, "Close": C, "Volume": C * 0 + 1e6}


def run(P, eligible=None, key=None, bull=True, **kw):
    C = P["Close"]
    eligible = pd.DataFrame(True, index=CAL, columns=C.columns) if eligible is None else eligible
    key = pd.DataFrame(0.0, index=CAL, columns=C.columns) if key is None else key
    bl = pd.Series(bull, index=CAL) if isinstance(bull, bool) else bull
    return tr.simulate(P, eligible, key, bl, START, END, **kw)


def test_week_last_marks_last_session_of_each_iso_week():
    cal = pd.DatetimeIndex(["2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28",      # Cuma (29 Mart) tatil
                            "2024-04-01", "2024-04-02", "2024-04-05", "2024-04-08"])
    assert tr.week_last(cal).tolist() == [False, False, False, True, False, False, True, False]


def test_trend_mask_needs_price_above_sma50_and_sma50_above_sma200():
    idx = pd.bdate_range("2020-01-01", periods=260)
    up = pd.Series(np.linspace(50, 150, 260), index=idx)
    down = pd.Series(np.linspace(150, 50, 260), index=idx)
    m = tr.trend_mask({"Close": pd.DataFrame({"UP": up, "DOWN": down})})
    assert bool(m["UP"].iloc[-1]) and not bool(m["DOWN"].iloc[-1])
    assert not m["UP"].iloc[:199].any()                                  # SMA200 yokken uygun değil


def test_buys_next_open_after_weekly_decision_with_fixed_size_and_costs():
    P = panel({"AAA": np.full(160, 100.0)})
    out, eq, T = run(P)
    assert len(T) == 1 and T.entry_day[0] == START and T.signal_date[0] == CAL[99] and T.reason[0] == "END"
    assert T.alloc_frac[0] == pytest.approx(0.125)                       # %1 risk ÷ %8 stop
    sh = (1250 - 1.5) / (100 * 1.001)
    assert eq.iloc[-1] == pytest.approx(10_000 - 1250 + sh * 100 * 0.999 - 1.5)
    assert out["taken"] == 1 and out["idle_cash_pct"] == pytest.approx(87.8, abs=0.1)     # son gün kapanışta tamamı nakit


def test_no_buy_when_regime_is_not_bull_or_stock_not_eligible():
    P = panel({"AAA": np.full(160, 100.0)})
    assert len(run(P, bull=False)[2]) == 0
    assert len(run(P, eligible=pd.DataFrame(False, index=CAL, columns=["AAA"]))[2]) == 0
    late = pd.Series(False, index=CAL)
    late.iloc[104:] = True                                               # ikinci karar gününde (104) BULL
    T = run(P, bull=late)[2]
    assert len(T) == 1 and T.entry_day[0] == CAL[105]


def test_intraday_stop_exits_at_stop_price_and_gap_exits_at_open():
    P = panel({"AAA": np.full(160, 100.0)}, lows={"AAA": {102: 91.0}})
    T = run(P, eligible=_only(["AAA"], 99))[2]
    assert (T.reason[0], T.exit_price[0], bool(T.at_open[0]), T.exit_day[0]) == ("STOP", 92.0, False, CAL[102])
    c = np.full(160, 100.0)
    c[102:] = 90.0
    P = panel({"AAA": c}, opens={"AAA": {102: 90.0}})
    T = run(P, eligible=_only(["AAA"], 99))[2]
    assert (T.reason[0], T.exit_price[0], bool(T.at_open[0]), T.exit_day[0]) == ("STOP", 90.0, True, CAL[102])
    assert T.ret_pct[0] == pytest.approx(-10.0)


def test_weekly_exit_only_on_decision_day_close_below_sma50():
    c = np.full(160, 100.0)
    c[101] = 97.0                                                        # hafta içi SMA50 altı, Cuma'ya toparlanır
    c[107:] = 96.0                                                       # ikinci hafta: Çarşamba'dan beri altında
    P = panel({"AAA": c})
    T = run(P, eligible=_only(["AAA"], 99))[2]
    assert (T.reason[0], T.exit_day[0], bool(T.at_open[0])) == ("TRAIL", CAL[110], True)   # Cuma 109 → Pazartesi açılış
    assert T.exit_price[0] == pytest.approx(96.0)


def test_cash_limits_to_eight_positions_in_key_order():
    names = [f"T{i:02d}" for i in range(12)]
    P = panel({t: np.full(160, 50.0) for t in names})
    key = pd.DataFrame({t: float(i) for i, t in enumerate(names)}, index=CAL)
    out, _, T = run(P, key=key)
    first = T[T.entry_day == START]
    assert sorted(first.ticker) == names[4:] and out["taken"] == 8       # en yüksek anahtarlı 8 hisse


def test_cold_streak_halves_risk_after_two_losing_trades():
    c = {t: np.full(160, 100.0) for t in ("L1", "L2", "NEW")}
    for t in ("L1", "L2"):
        c[t][101:] = 80.0
    P = panel(c, opens={"L1": {101: 80.0}, "L2": {101: 80.0}})
    el = _only(["L1", "L2"], 99)
    el.loc[CAL[104], "NEW"] = True                                       # ikinci karar gününde yeni aday
    T = run(P, eligible=el)[2]
    new = T[T.ticker == "NEW"].iloc[0]
    assert (T[T.ticker != "NEW"].net < 0).all() and new.alloc_frac == pytest.approx(0.0625)


def test_stopped_stock_can_be_rebought_on_a_later_decision_day():
    P = panel({"AAA": np.full(160, 100.0)}, lows={"AAA": {102: 91.0}})
    T = run(P)[2]
    assert T.reason.tolist() == ["STOP", "END"] and T.entry_day[1] == CAL[105]


def test_growth_rate_agrees_with_growth_condition():
    rng = np.random.default_rng(3)
    cal = pd.bdate_range("2019-01-01", "2023-12-29")
    for _ in range(20):
        ends = pd.date_range("2018-03-31", periods=18, freq="QE")
        known = ends + pd.to_timedelta(rng.integers(20, 60, len(ends)), unit="D")
        q = pd.DataFrame(dict(end=ends, rev=rng.uniform(50, 400, len(ends)), rev_known=known,
                              ni=rng.uniform(-20, 60, len(ends)), ni_known=known))
        kn, ok = pr.growth_asof(q, cal)
        rate = tr.growth_rate_asof(q, cal)
        assert np.isfinite(rate[kn]).all() and (rate[ok] >= pr.MIN_SALES_GROWTH).all()
        assert not np.isfinite(rate[~kn]).any()


def _only(tickers, i):
    """Yalnız i. gün (karar günü) uygun: tek alım, sonra yeniden alım yok."""
    cols = ["AAA"] if tickers == ["AAA"] else ["L1", "L2", "NEW"]
    el = pd.DataFrame(False, index=CAL, columns=cols)
    el.loc[CAL[i], tickers] = True
    return el
