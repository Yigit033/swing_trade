"""Lab v3 Ek T — MOM (aylık 12-1 momentum rotasyonu) kuralları, sentetik fiyatlarla.

Kural (ön kayıt): ay sonu seansında karar, ertesi seans açılışında işlem; uygun = havuzda, 200 günlük
ortalamanın üstünde; sıra 20'den büyükse ya da uygun değilse sat; yalnız BULL'da al; en çok 10 pozisyon.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab import rotation as rot  # noqa: E402

CAL = pd.bdate_range("2014-10-01", "2016-06-30")
TICKERS = [f"T{i:02d}" for i in range(25)]            # T00 en hızlı yükselen … T24 en yavaş
START, END = "2016-01-04", "2016-04-29"


def _panels(mutate=None):
    n = len(CAL)
    close = pd.DataFrame({t: 50.0 * (1 + 0.0010 - 0.00003 * i) ** np.arange(n) for i, t in enumerate(TICKERS)},
                         index=CAL)
    if mutate:
        mutate(close)
    return {"Open": close * 0.999, "Close": close, "High": close * 1.01, "Low": close * 0.99}


def _pool(value=True):
    return pd.DataFrame(value, index=CAL, columns=TICKERS)


BULL = {d: ("BULL", "CONFIRMED") for d in CAL}


def test_rebalance_days_are_month_ends_plus_the_session_before_the_period():
    days, last = rot.rebalance_days(CAL, pd.Timestamp(START), pd.Timestamp(END))
    # Sentetik takvimde 1 Ocak iş günü: dönem başından (4 Ocak) önceki son seans odur (gerçek takvimde 31 Aralık).
    assert [str(d.date()) for d in days] == ["2016-01-01", "2016-01-29", "2016-02-29", "2016-03-31"]
    assert str(last.date()) == "2016-04-29"            # dönemin son seansı karar günü değil


def test_buys_the_ten_strongest_at_the_next_open_and_holds_them():
    P = _panels()
    T = rot.mom_trades(P, _pool(), BULL, START, END)
    assert sorted(T.ticker) == TICKERS[:10]            # 12-1 momentum sırası = büyüme hızı sırası
    assert (T.entry_day == pd.Timestamp("2016-01-04")).all()
    assert (T.signal_date == pd.Timestamp("2016-01-01")).all()        # dönem başından önceki son seans
    for r in T.itertuples():
        assert r.entry_price == pytest.approx(P["Open"].at[pd.Timestamp("2016-01-04"), r.ticker])
        assert r.exit_price == pytest.approx(P["Close"].at[pd.Timestamp(END), r.ticker])
    assert set(T.reason) == {"END"} and not T.at_open.any()


def test_score_skips_the_last_month():
    # Son 21 seansta fırlayan hisse 12-1 skorunda öne geçmez (son ay bilerek atlanır).
    def spike(close):
        i = close.index.get_loc(pd.Timestamp("2015-12-31"))
        close.iloc[i - 15:, close.columns.get_loc("T24")] *= 3.0
    T = rot.mom_trades(_panels(spike), _pool(), BULL, START, END)
    assert "T24" not in set(T.ticker)


def test_trend_break_is_sold_at_next_rebalance_open_and_replaced_only_in_bull():
    def crash(close):                                  # T00 Ocak ortasında çöker → 200 günlük ortalamanın altı
        i = close.index.get_loc(pd.Timestamp("2016-01-15"))
        close.iloc[i:, close.columns.get_loc("T00")] *= 0.5
    P = _panels(crash)
    T = rot.mom_trades(P, _pool(), BULL, START, END)
    sold = T[T.ticker == "T00"].iloc[0]
    assert sold.reason == "TREND" and sold.at_open
    assert sold.exit_day == pd.Timestamp("2016-02-01")                 # 29 Ocak kararı → 1 Şubat açılışı
    assert sold.exit_price == pytest.approx(P["Open"].at[pd.Timestamp("2016-02-01"), "T00"])
    repl = T[T.ticker == "T10"].iloc[0]                                 # sıradaki en güçlü hisse
    assert repl.entry_day == pd.Timestamp("2016-02-01")
    caution = dict(BULL)
    caution[pd.Timestamp("2016-01-29")] = ("CAUTION", "CONFIRMED")
    T2 = rot.mom_trades(P, _pool(), caution, START, END)
    assert (T2.ticker == "T00").any() and T2[T2.ticker == "T00"].iloc[0].reason == "TREND"   # satış rejime bakmaz
    late = T2[T2.ticker == "T10"]
    assert len(late) == 1 and late.iloc[0].entry_day == pd.Timestamp("2016-03-01")           # alım BULL'u bekler


def test_no_buys_outside_bull_and_pool_exit_is_sold():
    assert rot.mom_trades(_panels(), _pool(), {d: ("BEAR", "CONFIRMED") for d in CAL}, START, END).empty
    pool = _pool()
    pool.loc["2016-02-29", "T03"] = False                              # havuzdan çıktı (ör. piyasa değeri < 10 mlr $)
    T = rot.mom_trades(_panels(), pool, BULL, START, END)
    out = T[T.ticker == "T03"].iloc[0]
    assert out.reason == "POOL" and out.exit_day == pd.Timestamp("2016-03-01")


def test_rank_beyond_twenty_is_sold():
    # T00'ın uzun vadeli getirisi (son ay hariç) en alta düşerse ama trendi bozulmazsa neden RANK olmalı.
    def fade(close):
        c = close.columns.get_loc("T00")
        i = close.index.get_loc(pd.Timestamp("2015-12-31"))
        base = close.iloc[i, c]
        close.iloc[i + 1:, c] = base * (1 - 0.0004) ** np.arange(1, len(close) - i)   # yavaş düşüş, 12-1 skoru erir
    P = _panels(fade)
    T = rot.mom_trades(P, _pool(), BULL, "2016-01-04", "2016-06-30")
    t00 = T[T.ticker == "T00"].iloc[0]
    assert t00.reason in ("RANK", "TREND") and t00.at_open and t00.exit_day > pd.Timestamp("2016-02-01")


def test_pool_return_uses_the_same_timing_as_the_trade():
    P = _panels()
    T = rot.mom_trades(P, _pool(), BULL, START, END)
    X = rot.attach_pool(T, P, _pool())
    a, b = pd.Timestamp("2016-01-04"), pd.Timestamp(END)
    want = float(((P["Close"].loc[b] / P["Open"].loc[a] - 1) * 100).mean())       # END: açılış → kapanış
    assert X.pool_ret.iloc[0] == pytest.approx(want)
    assert X.excess.mean() > 0                                                    # en güçlü 10 > havuz ortalaması
    T2 = T.assign(at_open=True)
    want_open = float(((P["Open"].loc[b] / P["Open"].loc[a] - 1) * 100).mean())
    assert rot.attach_pool(T2, P, _pool()).pool_ret.iloc[0] == pytest.approx(want_open)
