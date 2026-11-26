"""Lab v3 Ek V — PRO kuralları (büyüyen lider + VCE + dar stop + kazananı tutma + değişken boyut), sentetik veriyle.

Ön kayıt: lab/PREREGISTRATION.md Ek V. Kurallar lab/pro_rules.py (saf fonksiyonlar).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab import pro_rules as pr  # noqa: E402


# ── Büyüme verisi ────────────────────────────────────────────────────────────

def _fact(start, end, val, filed, form="10-Q"):
    return dict(start=start, end=end, val=val, filed=filed, form=form, accn=f"a-{filed}")


def _facts(rev=(), ni=(), rev_tag="Revenues", extra=None):
    gaap = {rev_tag: {"units": {"USD": list(rev)}}, "NetIncomeLoss": {"units": {"USD": list(ni)}}}
    gaap.update(extra or {})
    return {"facts": {"us-gaap": gaap}}


Q = [("2018-01-01", "2018-03-31"), ("2018-04-01", "2018-06-30"), ("2018-07-01", "2018-09-30"),
     ("2019-01-01", "2019-03-31"), ("2019-04-01", "2019-06-30"), ("2019-07-01", "2019-09-30")]
FILED = ["2018-05-05", "2018-08-04", "2018-11-03", "2019-05-04", "2019-08-03", "2019-11-02"]


def test_direct_quarters_use_the_first_filed_value_not_the_restated_one():
    rev = [_fact(*Q[0], 100, FILED[0]),
           _fact(*Q[0], 90, FILED[3]),                 # bir yıl sonra karşılaştırma sütununda düzeltilmiş değer
           _fact(*Q[3], 130, FILED[3])]
    ni = [_fact(*Q[0], 5, FILED[0]), _fact(*Q[3], 9, FILED[3])]
    q = pr.quarters_table(_facts(rev, ni))
    assert list(q.end.dt.strftime("%Y-%m-%d")) == ["2018-03-31", "2019-03-31"]
    assert q.rev.tolist() == [100.0, 130.0]            # 90 değil: ilk dosyalanan
    assert q.rev_known.iloc[0] == pd.Timestamp(FILED[0])


def test_year_end_quarter_is_annual_minus_nine_months_known_at_the_later_filing():
    rev = [_fact("2018-01-01", "2018-09-30", 300, "2018-11-03"),
           _fact("2018-01-01", "2018-12-31", 420, "2019-02-20", "10-K")]
    ni = [_fact("2018-01-01", "2018-09-30", 30, "2018-11-03"),
          _fact("2018-01-01", "2018-12-31", 50, "2019-02-20", "10-K")]
    q = pr.quarters_table(_facts(rev, ni))
    assert len(q) == 1
    r = q.iloc[0]
    assert (r.end, r.rev, r.ni) == (pd.Timestamp("2018-12-31"), 120.0, 20.0)
    assert r.rev_known == pd.Timestamp("2019-02-20")


def test_revenue_takes_the_largest_tag_in_the_first_filing_and_profit_follows_tag_priority():
    rev = [_fact(*Q[0], 100, FILED[0])]
    extra = {"SalesRevenueGoodsNet": {"units": {"USD": [_fact(*Q[0], 60, FILED[0])]}},      # alt kalem
             "ProfitLoss": {"units": {"USD": [_fact(*Q[0], 7, FILED[0]), _fact(*Q[1], 8, FILED[1])]}}}
    ni = [_fact(*Q[0], 5, FILED[0])]
    rev.append(_fact(*Q[1], 110, FILED[1]))
    q = pr.quarters_table(_facts(rev, ni, extra=extra))
    assert q.rev.tolist() == [100.0, 110.0]
    assert q.ni.tolist() == [5.0, 8.0]                 # 1. çeyrek NetIncomeLoss, 2. çeyrek yalnız ProfitLoss


def _growth_case(rev_now=130, ni_now=9, drop_prev=False):
    rev = [_fact(*Q[i], v, FILED[i]) for i, v in ((0, 100), (3, rev_now))]
    ni = [_fact(*Q[i], v, FILED[i]) for i, v in ((0, 5), (3, ni_now))]
    if drop_prev:
        rev, ni = rev[1:], ni[1:]
    cal = pd.bdate_range("2019-04-01", "2020-03-31")
    known, ok = pr.growth_asof(pr.quarters_table(_facts(rev, ni)), cal)
    return cal, pd.Series(known, index=cal), pd.Series(ok, index=cal)


def test_growth_is_known_only_from_the_session_after_the_filing_date():
    cal, known, ok = _growth_case()                    # dosya 2019-05-04 (Cumartesi) → ilk seans 2019-05-06
    assert not ok.loc[:"2019-05-03"].any() and not known.loc[:"2019-05-03"].any()
    assert ok.loc["2019-05-06"] and known.loc["2019-05-06"]


def test_growth_goes_stale_200_days_after_the_period_end():
    cal, known, ok = _growth_case()                    # dönem sonu 2019-03-31 → 200 gün: 2019-10-17
    assert ok.loc["2019-10-17"] and not ok.loc["2019-10-18"]
    assert not known.loc["2019-10-18"]


@pytest.mark.parametrize("rev_now,ni_now", [(124, 9), (130, 5), (130, 4)])
def test_growth_needs_25_percent_sales_and_higher_profit(rev_now, ni_now):
    cal, known, ok = _growth_case(rev_now, ni_now)
    assert known.loc["2019-06-03"] and not ok.any()


def test_growth_accepts_a_shrinking_loss_and_rejects_missing_year_ago_quarter():
    assert _growth_case(ni_now=-1)[2].loc["2019-06-03"] is not None
    rev = [_fact(*Q[0], 100, FILED[0]), _fact(*Q[3], 130, FILED[3])]
    ni = [_fact(*Q[0], -5, FILED[0]), _fact(*Q[3], -1, FILED[3])]
    cal = pd.bdate_range("2019-04-01", "2019-08-30")
    assert pr.growth_asof(pr.quarters_table(_facts(rev, ni)), cal)[1][cal.get_loc("2019-06-03")]
    cal, known, ok = _growth_case(drop_prev=True)
    assert not known.any() and not ok.any()


def test_newer_quarter_replaces_the_older_one():
    rev = [_fact(*Q[0], 100, FILED[0]), _fact(*Q[1], 100, FILED[1]), _fact(*Q[3], 130, FILED[3]),
           _fact(*Q[4], 105, FILED[4])]                # 2. çeyrek büyüme %5 → şart düşer
    ni = [_fact(*Q[0], 5, FILED[0]), _fact(*Q[1], 5, FILED[1]), _fact(*Q[3], 9, FILED[3]), _fact(*Q[4], 9, FILED[4])]
    cal = pd.bdate_range("2019-04-01", "2019-12-31")
    ok = pd.Series(pr.growth_asof(pr.quarters_table(_facts(rev, ni)), cal)[1], index=cal)
    assert ok.loc["2019-05-06":"2019-08-02"].all() and not ok.loc["2019-08-05":].any()


# ── VCE paritesi ─────────────────────────────────────────────────────────────

def _random_frames(n_tickers=40, n=400, seed=3):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2018-01-01", periods=n)
    P = {f: {} for f in ("Open", "High", "Low", "Close", "Volume")}
    for i in range(n_tickers):
        vol = np.where(rng.random(n) < 0.15, 0.004, 0.02)             # sakin ve oynak dönemler → sıkışma
        vol = pd.Series(vol).rolling(12, min_periods=1).mean().values
        c = 50 * np.exp(np.cumsum(rng.normal(0.001, vol)))
        o = c * (1 + rng.normal(0, vol / 3))
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2)))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2)))
        v = rng.lognormal(13, 0.6, n)
        for f, x in zip(("Open", "High", "Low", "Close", "Volume"), (o, h, lo, c, v)):
            P[f][f"T{i:02d}"] = x
    return {f: pd.DataFrame(d, index=cal) for f, d in P.items()}


def test_vectorized_vce_matches_the_production_trigger():
    from swing_trader.small_cap.signals import SmallCapSignals
    sig = SmallCapSignals()
    P = _random_frames()
    mask = pr.vce_mask(P)
    cal = mask.index
    checked = hits = 0
    for t in mask.columns[:12]:
        df = pd.DataFrame({f: P[f][t].values for f in ("Open", "High", "Low", "Close", "Volume")})
        df.insert(0, "Date", cal)
        for i in range(140, len(cal)):
            w = df.iloc[i - 125: i + 1].reset_index(drop=True)        # canlının ~6 aylık penceresi
            want = sig.check_vce_breakout(w)[0]
            assert bool(mask.iat[i, mask.columns.get_loc(t)]) == bool(want), (t, cal[i])
            checked += 1
            hits += int(want)
    assert checked > 3000 and hits >= 5                                 # test gerçekten isabet içeriyor


# ── İşlem yürütme ────────────────────────────────────────────────────────────

CAL = pd.bdate_range("2019-01-01", periods=140)


def _one(close, open_=None, low=None):
    c = pd.DataFrame({"X": np.asarray(close, dtype=float)}, index=CAL)
    o = pd.DataFrame({"X": np.asarray(open_ if open_ is not None else close, dtype=float)}, index=CAL)
    lo = pd.DataFrame({"X": np.asarray(low if low is not None else np.minimum(c["X"].values, o["X"].values), dtype=float)},
                      index=CAL)
    return {"Open": o, "Low": lo, "Close": c, "High": c * 1.0}


def _signal(*idx):
    s = pd.DataFrame(False, index=CAL, columns=["X"])
    for i in idx:
        s.iat[i, 0] = True
    return s


KEY = pd.DataFrame(1.0, index=CAL, columns=["X"])
START, END = CAL[60], CAL[-1]


def test_stop_is_the_signal_day_low_capped_at_eight_percent():
    close = np.full(140, 100.0)
    low = close.copy()
    low[69] = 97.0                                     # sinyal günü dibi → stop 97 (uzaklık %3)
    low[72] = 96.0                                     # stop gün içinde çalışır
    T = pr.pro_trades(_one(close, low=low), _signal(69), KEY, START, END)
    r = T.iloc[0]
    assert (r.entry_day, r.exit_day, r.reason) == (CAL[70], CAL[72], "STOP")
    assert r.exit_price == 97.0 and r.stop_dist == pytest.approx(0.03) and not r.at_open
    low[69] = 80.0                                     # dip çok uzak → stop girişin %8 altı
    low[72] = 91.0
    r = pr.pro_trades(_one(close, low=low), _signal(69), KEY, START, END).iloc[0]
    assert r.exit_price == pytest.approx(92.0) and r.stop_dist == pytest.approx(0.08)


def test_gap_through_the_stop_fills_at_the_open_and_gap_below_signal_low_is_skipped():
    close = np.full(140, 100.0)
    low, open_ = close.copy(), close.copy()
    low[69] = 97.0
    open_[73], low[73], close[73] = 90.0, 89.0, 90.0   # boşlukla stop altı açılış
    r = pr.pro_trades(_one(close, open_, low), _signal(69), KEY, START, END).iloc[0]
    assert (r.exit_day, r.exit_price, r.reason, r.at_open) == (CAL[73], 90.0, "STOP", True)
    open_[70] = 96.5                                   # giriş açılışı sinyal günü dibinin altında → işlem yok
    assert len(pr.pro_trades(_one(close, open_, low), _signal(69), KEY, START, END)) == 0


def test_winner_is_held_until_the_open_after_the_first_close_below_sma50():
    close = np.concatenate([np.full(70, 100.0), 100.0 * 1.01 ** np.arange(1, 41), np.full(30, 100.0)])
    low = close * 0.999
    low[69] = 99.0
    P = _one(close, low=low)
    r = pr.pro_trades(P, _signal(69), KEY, START, END).iloc[0]
    s50 = P["Close"]["X"].rolling(50).mean()
    first_below = next(i for i in range(70, 140) if close[i] < s50.iloc[i])
    assert first_below == 110                          # yükseliş bitince kapanış ortalamanın altına iner
    assert (r.exit_day, r.reason, r.at_open) == (CAL[111], "TRAIL", True)
    assert r.bars == 42 and r.ret_pct == pytest.approx(100 * (100.0 / 101.0 - 1))     # giriş 101, çıkış 100


def test_open_position_blocks_new_signals_and_period_end_closes_at_the_last_close():
    close = 100.0 * 1.004 ** np.arange(140)            # kesintisiz yükseliş: çıkış yok
    low = close * 0.999
    T = pr.pro_trades(_one(close, low=low), _signal(69, 80, 90), KEY, START, END)
    assert len(T) == 1 and T.iloc[0].reason == "END"
    assert T.iloc[0].exit_day == CAL[-1] and T.iloc[0].exit_price == pytest.approx(close[-1])


def test_time_exit_closes_at_the_eighth_session():
    close = np.full(140, 100.0)
    low = close * 0.999
    low[69] = 95.0                                     # stop 95: sekiz seansta çalışmaz
    r = pr.pro_trades(_one(close, low=low), _signal(69), KEY, START, END, exit_mode="time8").iloc[0]
    assert (r.entry_day, r.exit_day, r.reason, r.bars) == (CAL[70], CAL[77], "TIME", 8)


# ── Portföy ──────────────────────────────────────────────────────────────────

def _trades(rows):
    T = pd.DataFrame(rows)
    T["strategy"], T["key"], T["at_open"] = "PRO", 1.0, False
    T["ret_pct"] = 100 * (T.exit_price / T.entry_price - 1)
    T["bars"] = 5
    return T


def _closes(tickers):
    return pd.DataFrame(100.0, index=CAL, columns=tickers)


def test_position_size_is_one_percent_risk_capped_at_a_quarter_of_equity():
    T = _trades([dict(ticker="A", entry_day=CAL[1], exit_day=CAL[5], entry_price=100.0, exit_price=100.0, stop_dist=0.08),
                 dict(ticker="B", entry_day=CAL[1], exit_day=CAL[5], entry_price=100.0, exit_price=100.0, stop_dist=0.02)])
    out, eq, Tk = pr.portfolio_risk(T, _closes(["A", "B"]), cost=0.0, fee=0.0)
    # A: %1 / %8 = %12.5; B: %1 / %2 = %50 → tavan %25. Ortalama %18.75.
    assert out["taken"] == 2 and out["mean_position_pct"] == pytest.approx(18.8, abs=0.06)
    assert out["total_pct"] == 0.0


def test_two_losses_in_a_row_halve_the_risk_until_a_win():
    rows = [dict(ticker=t, entry_day=CAL[a], exit_day=CAL[a + 2], entry_price=100.0, exit_price=x, stop_dist=0.10)
            for t, a, x in (("A", 1, 95.0), ("B", 5, 95.0), ("C", 10, 110.0), ("D", 15, 100.0))]
    out, eq, Tk = pr.portfolio_risk(_trades(rows), _closes(list("ABCD")), cost=0.0, fee=0.0)
    e = [float(eq.loc[CAL[i]]) for i in (0 + 1, 4, 9, 14)]
    # A ve B: %10 pozisyon; iki zarardan sonra C: %5 pozisyon (+%10 → +%0.5); C kârlı kapandı → D yine %10
    assert eq.loc[CAL[3]] == pytest.approx(10_000 * 0.995)
    assert eq.loc[CAL[7]] == pytest.approx(10_000 * 0.995 ** 2)
    assert eq.loc[CAL[12]] == pytest.approx(10_000 * 0.995 ** 2 * 1.005)
    assert out["mean_position_pct"] == pytest.approx((10 + 10 + 5 + 10) / 4, abs=0.06)
    assert e[0] == 10_000


def test_trade_is_skipped_when_cash_is_short_and_fees_reduce_equity():
    rows = [dict(ticker=t, entry_day=CAL[1], exit_day=CAL[9], entry_price=100.0, exit_price=100.0, stop_dist=0.01)
            for t in "ABCDE"]                           # her biri %25 → beşincisine nakit kalmaz
    out, eq, Tk = pr.portfolio_risk(_trades(rows), _closes(list("ABCDE")), cost=0.0, fee=1.5)
    assert out["taken"] == 4
    assert eq.iloc[-1] == pytest.approx(10_000 - 4 * 2 * 1.5, abs=0.01)
